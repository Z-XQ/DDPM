import argparse
import torchvision
import torch.nn.functional as F

from .unet import UNet
from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,  # 512
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    # 后向，去噪模型，由符合正态分布的随机噪声图像，去噪生成原图
    model = UNet(
        img_channels=3,
        base_channels=args.base_channels,  # 128 第一个卷积输出通道数
        channel_mults=args.channel_mults,  # (1,2,2,2) 每个layer输出的通道数的倍率. Default: (1, 2, 4, 8)
        time_emb_dim=args.time_emb_dim,  # 512 time embedding dimension or None if the block doesn't use time conditioning. Default: None
        norm=args.norm,  # gn which normalization to use (instance, group, batch, or none). Default: "gn"
        dropout=args.dropout,  # 0.1
        activation=activations[args.activation],  # silu
        attention_resolutions=args.attention_resolutions,  # 1 表示第1个layer设置了attention. Default: ()

        num_classes=None if not args.use_labels else 10,  # train: false/true
        initial_pad=0,  # 对开始输入图片的宽度或者高度不是2的倍数的，进行padding. Default: 0
    )

    # 根据传入参数args中指定的调度方式（schedule）来生成扩散过程中使用的betas值。
    if args.schedule == "cosine":  # 生成基于余弦调度的betas值
        # 这种调度方式会根据余弦函数的变化来动态调整扩散过程中不同时间步的噪声添加强度，使得扩散过程更符合某些理想的特性。
        betas = generate_cosine_schedule(args.num_timesteps)  # num_timesteps==1000，生成1000个数字
    else:
        # 基于给定的起始值、结束值以及总时间步数，线性地确定每个时间步对应的噪声添加强度。
        betas = generate_linear_schedule(
            T=args.num_timesteps,  # 个数
            low=args.schedule_low * 1000 / args.num_timesteps,
            high=args.schedule_high * 1000 / args.num_timesteps,
        )

    # 创建GaussianDiffusion实例，它是整个高斯扩散模型的核心类，将前面构建的UNet去噪模型以及生成的betas等参数整合在一起，
    # 用于实现完整的扩散模型的正向扩散、反向去噪、损失计算等功能，以下是对其初始化参数的详细解释：
    diffusion = GaussianDiffusion(
        model, (32, 32), 3, 10,  # model, input image size, input_c, num_classes(这里与前面UNet模型中根据是否使用标签确定的类别数量相关联。)
        betas,  # 决定了扩散过程中每个时间步添加噪声的强度，是扩散模型正向和反向操作的关键参数依据。
        # 指定指数移动平均（EMA）的衰减率，用于在训练过程中平滑模型的权重更新，使得模型参数不会过于剧烈地变化，从参数args中获取，默认值为0.9999。
        ema_decay=args.ema_decay,
        # 指定EMA的更新频率，即每隔多少个训练步骤更新一次EMA模型的权重，从参数args中获取，默认值为1，表示每个训练步骤都进行更新。
        ema_update_rate=args.ema_update_rate,
        # 指定开始进行EMA操作的训练步数，在训练前期可能不需要进行EMA，达到该步数后才开始更新EMA模型权重，这里设置为2000步。
        ema_start=2000,
        # 指定损失类型，从参数args中获取，可选 "l1"（使用平均绝对误差损失）或 "l2"（使用均方误差损失），用于在训练过程中计算模型预测噪声与真实噪声之间的损失。
        loss_type=args.loss_type,
    )

    return diffusion
