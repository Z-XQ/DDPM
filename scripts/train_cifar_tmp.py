import argparse
import datetime
import os

import torch
# import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
from ddpm import script_utils


def main():
    # 解析出参数，将参数保存到args变量中
    args = create_argparser().parse_args()
    device = args.device

    # 1 model 通过script_utils模块中的函数，根据传入的参数args创建扩散模型实例
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

    # 加载预训练模型和优化器
    if args.model_checkpoint is not None:
        diffusion.load_state_dict(torch.load(args.model_checkpoint))
    if args.optim_checkpoint is not None:
        optimizer.load_state_dict(torch.load(args.optim_checkpoint))

    batch_size = args.batch_size

    # 2 dataset
    train_dataset = datasets.CIFAR10(
        root='./cifar_train',
        train=True,
        download=True,  # 数据集不存在自动下载 download=True
        transform=script_utils.get_transform(),
    )

    # test_dataset = datasets.CIFAR10(
    #     root='./cifar_test',
    #     train=False,
    #     download=True,  # 数据集不存在自动下载 download=True
    #     transform=script_utils.get_transform(),
    # )

    # 3 dataloader
    train_loader = script_utils.cycle(DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ))

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)

    acc_train_loss = 0

    for iteration in range(1, args.iterations + 1):
        diffusion.train()   # 将扩散模型设置为训练模式

        x, y = next(train_loader)
        x = x.to(device)
        y = y.to(device)

        # 计算扩散模型的损失
        if args.use_labels:
            loss = diffusion(x, y)
        else:
            loss = diffusion(x)

        acc_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新扩散模型的指数移动平均（Exponential Moving Average，EMA）相关参数
        diffusion.update_ema()

        if iteration % args.checkpoint_rate == 0:
            model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
            optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

            os.makedirs(args.log_dir, exist_ok=True)
            torch.save(diffusion.state_dict(), model_filename)
            torch.save(optimizer.state_dict(), optim_filename)


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,

        # log_rate=1000,
        checkpoint_rate=1000,   # 控制保存权重频率
        log_dir="~/ddpm_logs",

        model_checkpoint=None,  # 预训练权重路径
        optim_checkpoint=None,  # 预训练优化器路径

        schedule_low=1e-4,  # 控制beta范围： low=args.schedule_low * 1000 / args.num_timesteps,
        schedule_high=0.02,  # high=args.schedule_high * 1000 / args.num_timesteps,

        device=device,
    )
    # 将script_utils模块中的扩散模型相关默认参数更新到defaults字典中
    defaults.update(script_utils.diffusion_defaults())

    # 将defaults字典中的参数添加到命令行参数解析器中，使得可以通过命令行传入参数来覆盖默认值
    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()