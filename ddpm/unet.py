import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.normalization import GroupNorm


# 根据指定的归一化类型
def get_norm(norm, num_channels, num_groups):
    # 如果归一化类型为'in'，则返回实例归一化层（InstanceNorm2d），并设置可学习的仿射参数（affine=True）
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    # 如果归一化类型为'bn'，则返回批归一化层（BatchNorm2d）
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    # 如果归一化类型为'gn'，则返回组归一化层（GroupNorm），传入组数量和通道数量
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    # 如果归一化类型为None，则返回恒等映射层（Identity），相当于不做归一化操作
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


# 用于计算时间步（timesteps）的位置嵌入（Positional Embedding）的类
# 所谓的位置嵌入，其实就是系数，一个递减系数序列，然后乘以输入x（x的分布是正太分布），完成对x的位置嵌入。
# 其实更直白的将就是将int类型x序列，变成(-1,1)之间的double类型序列。方便训练。
class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        # 确保嵌入维度是偶数，这是后续计算正弦和余弦位置嵌入的常见要求
        assert dim % 2 == 0  # dim==128
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        """
        x: (N,). N == base_channels，
        Output: (N, dim).
        """
        device = x.device
        half_dim = self.dim // 2
        # 计算用于生成位置嵌入的指数缩放因子，使得不同位置的嵌入在数值上有合适的分布
        emb = math.log(10000) / half_dim  # 0.1439...
        # 生成一个[0,half_dim - 1]的等差序列，作为指数的底数，然后将其转换为指定设备上的张量，并乘以 -emb 得到指数序列
        # 每一行对应一个样本在部分维度上的位置嵌入情况，不同样本（不同行）由于时间步不同，其位置嵌入也会有所差异，体现了时间步对位置嵌入的影响(递减)。
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # (half_dim,)

        # 至此生成固定系数emb. shape==(half_dim,), val是[1,0)之前的递减序列

        # 将输入的时间步张量x（形状为 (N)）与缩放因子self.scale相乘，再与指数序列emb进行外积运算，得到一个形状为 (N, half_dim) 的中间嵌入张量
        # 对应公式里面的σt*z，外积运算（叉积）：同时包含时间步的变化以及位置编码的相对位置信息。
        emb = torch.outer(x * self.scale, emb)  # (128,) (64,) -> (128,64)

        # 将中间嵌入张量按照正弦和余弦函数分别计算两部分嵌入，然后在最后一维上进行拼接，得到最终的位置嵌入张量，形状为 (N, dim)
        # 使用正弦和余弦函数组合来生成位置嵌入是参考了 Transformer 架构中位置编码的经典方法，它能够有效地为不同位置（在这里对应不同时间步）
        # 的元素赋予不同且具有周期性变化规律的编码信息，使得模型在处理序列或时间相关数据时，能够更好地捕捉顺序和相对位置关系，有助于提高模型的性能和学习效果。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (128,64) -> (128,128)
        return emb


# conv2d 用于对输入张量进行下采样（Downsample）的类，通过步长卷积实现下采样，下采样因子为2
class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):
        """
        x: (N, in_channels, H, W)
        output:  (N, in_channels, H // 2, W // 2)
        """
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


# upSample + conv2d: 用于对输入张量进行上采样（Upsample）的类
class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        """
        x: (N, in_channels, H, W)
        output: (N, in_channels, H * 2, W * 2)
        """
        return self.upsample(x)


# 卷积注意力模块（AttentionBlock）类，实现了带有残差连接的QKV自注意力机制
class AttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        # 根据指定的归一化类型
        self.norm = get_norm(norm, in_channels, num_groups)

        # 创建一个1x1卷积层，将输入通道数转换为输入通道数的3倍（用于生成Q、K、V三个向量）
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)

        # 创建一个1x1卷积层，用于将注意力计算后的结果转换回原始的输入通道数，以便进行残差连接
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """
        x: (N, in_channels, H, W)
        output: (N, in_channels, H, W)
        """
        b, c, h, w = x.shape
        # 通过1x1卷积层将归一化后的输入x转换为Q、K、V三个向量，然后按照通道维度进行分割，每个向量的通道数为原始输入通道数in_channels
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        # 将Q向量的维度进行重排，变为 (b, h * w, c) 的形状，以便后续进行批量矩阵乘法计算注意力权重
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        # 将K向量的维度调整为 (b, c, h * w) 的形状，用于与Q向量进行矩阵乘法
        k = k.view(b, c, h * w)
        # 将V向量的维度重排为 (b, h * w, c) 的形状，以便在计算注意力权重后与权重相乘得到最终输出
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        # 计算Q和K向量的点积，得到注意力得分，并乘以一个缩放因子（通道数的负平方根），以稳定训练过程，形状为 (b, h * w, h * w)
        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        # 对注意力得分进行softmax归一化，得到注意力权重，维度为 -1 表示在最后一维（h * w）上进行归一化操作
        attention = torch.softmax(dot_products, dim=-1)
        # 将注意力权重与V向量进行批量矩阵乘法，得到经过注意力加权后的输出，形状为 (b, h * w, c)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        # 将输出的维度调整回 (b, c, h, w) 的形状，与输入x的维度一致
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        # 将经过注意力模块处理后的输出与原始输入x进行残差连接，并返回结果
        return self.to_out(out) + x


# 残差块（ResidualBlock）类，包含两个卷积块以及时间和类别条件添加机制，并带有残差连接，可选择是否应用注意力模块
class ResidualBlock(nn.Module):
    __doc__ = r"""Applies two conv blocks with resudual connection. Adds time and class conditioning by adding bias after first convolution.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            dropout,
            time_emb_dim=None,
            num_classes=None,
            activation=F.relu,
            norm="gn",
            num_groups=32,
            use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        # 根据指定的归一化类型
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # 如果指定了时间嵌入维度（time_emb_dim不为None），则创建一个线性层，
        # 用于将时间嵌入向量转换为与输出通道数匹配的偏置项，以便添加到卷积结果中实现时间条件添加
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        # 如果指定了类别数量（num_classes不为None），则创建一个嵌入层，?
        # 用于将类别标签转换为与输出通道数匹配的偏置项，以便添加到卷积结果中实现类别条件添加
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        # 创建一个1x1卷积层，用于调整输入通道数与输出通道数不一致时的残差连接；如果输入和输出通道数相同，则返回恒等映射层（Identity）
        self.residual_connection = nn.Conv2d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        # 如果使用注意力机制（use_attention为True），则创建一个AttentionBlock实例；否则返回恒等映射层（Identity）
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)

    def forward(self, x, time_emb=None, y=None):
        """
        x:
        time_emb: 反向推理，额外加的噪声向量σt*z，(base_channels==128, time_emb_dim==512)
        output:
        """
        # 对输入x进行归一化后，应用激活函数，再通过第一个卷积层进行特征提取
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        # 1 如果存在时间偏置项（self.time_bias不为None），则需要传入时间嵌入向量time_emb，
        # 将其通过激活函数和线性层转换后，添加到卷积结果中作为时间条件
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            # time_bias变通道，为了加到out: (base_channels==128, time_emb_dim==512) -> (time_emb_dim, out_channels)
            # out.shape[128,128,32,32], * 噪声矩阵[128,128,1,1]
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]  # 全连接

        # 2 如果存在类别偏置项（self.class_bias不为None），则需要传入类别标签y，将其通过嵌入层转换后，添加到卷积结果中作为类别条件
        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        # 3 加上残差连接的输入x（经过通道数调整，如果需要的话）
        out = self.conv_2(out) + self.residual_connection(x)
        # 如果使用了注意力机制，则将输出通过注意力模块进行处理
        out = self.attention(out)  # 部分残差块有attention

        return out


# UNet模型类，用于估计噪声，是整个网络结构的核心部分，包含下采样、中间层和上采样等多个模块，并集成了时间和类别条件处理机制
class UNet(nn.Module):
    __doc__ = """UNet model used to estimate noise.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        img_channels (int): number of image channels
        base_channels (int): number of base channels (after first convolution)
        channel_mults (tuple): tuple of channel multiplers. Default: (1, 2, 4, 8)
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        time_emb_scale (float): linear scale to be applied to timesteps. Default: 1.0
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        dropout (float): dropout rate at the end of each residual block
        attention_resolutions (tuple): list of relative resolutions at which to apply attention. Default: ()
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """

    def __init__(
            self,
            img_channels,
            base_channels,  # 128
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,  # 每个layer有多少个残差块
            time_emb_dim=None,  # 512
            time_emb_scale=1.0,
            num_classes=None,
            activation=F.relu,
            dropout=0.1,
            attention_resolutions=(),  # 有4个layer层，取值0-3，该值决定是在哪个layer上添加attention
            norm="gn",
            num_groups=32,
            initial_pad=0,
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad

        self.num_classes = num_classes

        # 全连接mlp
        # 如果指定了时间嵌入维度（time_emb_dim不为None），创建一个时间多层感知机（time MLP），用于处理时间嵌入信息
        # 它首先通过PositionalEmbedding层将输入的时间步信息转换为位置嵌入
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),  # (base_channels,) -> (base_channels,base_channels)
            nn.Linear(base_channels, time_emb_dim),   # (base_channels,base_channels) -> (base_channels,time_emb_dim)
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]  # 记录下采样过程中，每个特征的通道数
        now_channels = base_channels

        # 构建下采样部分的网络结构
        for i, mult in enumerate(channel_mults):  # (1, 2, 4, 8)
            out_channels = base_channels * mult  # 以128为基础，乘以倍数，得到每个layer输出通道数

            # num_res_blocks==2,每个layer有2两个残差块
            for _ in range(num_res_blocks):
                # 为每个下采样阶段添加多个残差块（ResidualBlock），可以根据配置决定是否应用注意力机制
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,  # 残差块内部有：时间嵌入、类别嵌入、注意力机制
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,  # 有4个layer层，i取值0-3，该值决定是在哪个layer上添加attention
                ))
                now_channels = out_channels
                channels.append(now_channels)

            # 在每组残差块之后（除了最后一组），添加下采样层
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # 构建中间层部分的网络结构，包含两个残差块，通常在这里进行更复杂的特征交互和处理，第一个残差块应用注意力机制
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,  # 应用注意力机制
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])

        # 构建上采样部分的网络结构
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            # num_res_blocks==2+1,每个layer有3两个残差块
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels

            # 在每组残差块之后（除了第一组），添加上采样层
            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        # 创建最终的输出卷积层，将特征图的通道数转换回输入图像的通道数（img_channels）
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, time=None, y=None):
        """
        x: (b,3,h,w)
        time: (base_channels,)。公式里面的z,每次随机从标准正态分布区间[0,1000)里面随机采样base_channels个随机数
        output: (b,3,h,w)
        """
        # 输入图片长宽不是2的倍数，则需要padding成2的倍数
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)  # 填充值在四个方向上都为ip

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            # 对输入的时间信息进行处理(加上系数σt)，得到时间嵌入向量σt*z
            time_emb = self.time_mlp(time)  # (base_channels,) -> (base_channels==128, time_emb_dim==512)
        else:
            time_emb = None

        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")

        x = self.init_conv(x)

        skips = [x]

        # 每一“残差层”都有噪声嵌入
        # 依次通过下采样部分的各个层（残差块和下采样层），保存中间特征图到skips列表中
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)

        # 通过中间层部分的两个残差块进行特征处理
        for layer in self.mid:
            x = layer(x, time_emb, y)

        # 依次通过上采样部分的各个层（残差块和上采样层）
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):  # 对于残差块，将对应下采样阶段保存的特征图与当前特征图进行拼接后作为输入
                x = torch.cat([x, skips.pop()], dim=1)  # 如果是上采样层，就直接up(x)即可。
            x = layer(x, time_emb, y)

        # 对最终的特征图进行归一化处理，应用激活函数，再通过最终的输出卷积层得到输出结果
        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        # 如果初始填充值不为0，去除填充部分后返回最终输出
        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x
