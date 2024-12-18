import argparse
import torch
import torchvision

from ddpm import script_utils


def main():
    args = create_argparser().parse_args()  # 设置测试模型参数，比如权重路径、测试多少张图片
    device = args.device

    try:
        # 加载模型和权重
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path))

        # 生成图片
        # 训练时没有使用label
        if args.use_labels:
            for label in range(10):
                y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                samples = diffusion.sample(args.num_images // 10, device, y=y)

                for image_id in range(len(samples)):
                    image = ((samples[image_id] + 1) / 2).clip(0, 1)
                    torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
        else:  # 生成num_images个数的图片
            samples = diffusion.sample(args.num_images, device)
            # 保存生成的图片
            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(
        num_images=10,  # 生成多少张图片
        device=device,
        # 后面参数没啥用处
        schedule_low=1e-4,  # 控制beta范围： low=args.schedule_low * 1000 / args.num_timesteps,
        schedule_high=0.02,  # high=args.schedule_high * 1000 / args.num_timesteps,
    )
    defaults.update(script_utils.diffusion_defaults())  # 添加扩散模型的默认参数

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)  # 合并命令行参数
    return parser


if __name__ == "__main__":
    main()