import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract


# 定义高斯扩散（Gaussian Diffusion）模型类
class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x:tensor of shape (N, img_channels, *img_size)
        y:tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module):model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int):number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(
            self,
            model,  # 估计生成，去噪模型unet
            img_size,  # (H, W)
            img_channels,  # 3
            num_classes,  # 10  没有用到
            betas,    # (1000,)
            loss_type="l2",  # 均方误差损失
            ema_decay=0.9999,  # 指定指数移动平均（EMA）的衰减率，用于在训练过程中平滑模型的权重更新，使得模型参数不会过于剧烈地变化，从参数args中获取，默认值为0.9999。
            ema_start=5000,  # 指定开始进行EMA操作的训练步数，在训练前期可能不需要进行EMA，达到该步数后才开始更新EMA模型权重，默认5000步。
            ema_update_rate=1,  # 指定EMA的更新频率，即每隔多少个训练步骤更新一次EMA模型的权重，从参数args中获取，默认值为1，表示每个训练步骤都进行更新。
    ):
        super().__init__()
        # 保存传入的用于估计扩散噪声的模型（通常是像UNet这样的网络结构），后续在正向扩散、反向去噪等操作中会用到该模型进行计算
        self.model = model
        # 通过深拷贝传入的模型，创建一个用于指数移动平均（EMA）的模型副本，EMA可以让模型权重在训练过程中更平滑地更新，避免剧烈波动
        self.ema_model = deepcopy(model)

        # 创建EMA实例，用于管理模型权重的指数移动平均更新过程，传入ema_decay参数来指定权重更新的衰减率
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        # 检查传入的损失类型是否是合法的（"l1" 或 "l2"）
        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type

        # 获取扩散过程的总时间步数，由传入的betas数组的长度确定，代表整个扩散过程被划分成了多少个时间阶段
        self.num_timesteps = len(betas)  # (1000,)

        alphas = 1.0 - betas  # (1000,)
        alphas_cumprod = np.cumprod(alphas)  # eg.[1, 2, 3, 4] -> [ 1,  2,  6, 24]

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 缓冲区中的数据会随着模型一起保存和加载，且不需要计算梯度
        # 因为这些参数在训练过程中是固定的或者是按照预定规则变化的，不需要通过反向传播来更新
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))  # 用于计算不同时间步下的噪声添加、去除等操作
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    # 更新扩散模型的指数移动平均（Exponential Moving Average，EMA）相关参数（如果模型中有此机制）
    def update_ema(self):
        self.step += 1
        # 按照设定的更新频率（ema_update_rate）来更新指数移动平均模型的权重
        if self.step % self.ema_update_rate == 0:
            # 在开始阶段（步数小于ema_start），直接将当前模型的权重复制给ema_model
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                # 否则，使用EMA实例来更新ema_model的权重，使其更平滑地接近当前模型的权重
                self.ema.update_model_average(self.ema_model, self.model)

    # 推理
    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        if use_ema:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    # 推理
    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        # 从标准正态分布中采样生成初始的噪声数据，形状为(batch_size, img_channels, *img_size)
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)  # (b,3,h,w)

        # 从总时间步数（num_timesteps - 1）开始，倒序循环到0，逐步进行去噪采样过程
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                # 当时间步大于0时，添加一定的噪声，模拟扩散过程的反向操作，使得采样更符合扩散模型的分布特性
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    # 推理
    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    # 加噪声noise
    def perturb_x(self, x, t, noise):
        """
        x: shape(b,3,h,w)
        t: shape(b,)
        noise: shape(b,3,h,w)
        """
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, y):
        """
        x: x0, shape(b,3,h,w), float32. 输入原始图像
        t: 时间步数t. shape(b,) int.
        """
        # 从一个标准正态分布随机采样得到(b,3,h,w)
        noise = torch.randn_like(x)

        # 加噪noise
        perturbed_x = self.perturb_x(x, t, noise)

        # 估计噪声
        estimated_noise = self.model(perturbed_x, t, y)

        # 计算估计噪声和真实噪声的l2距离
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None):  # x.shape(b,3,h,w). y=None
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")

        # 随机从标准正态分布区间[0,1000)里面随机采样base_channels个整数作为时间步数
        t = torch.randint(0, self.num_timesteps, (b,), device=device)  # (128,)
        return self.get_losses(x, t, y)


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)
