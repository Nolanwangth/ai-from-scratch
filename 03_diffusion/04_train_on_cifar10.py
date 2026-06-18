"""
训练 Diffusion 模型生成 CIFAR-10 图像
======================================
完整流程: 数据 → 模型 → 训练 → 采样 → 可视化

这个脚本会:
    1. 加载 CIFAR-10 数据集 (32×32 彩色图像)
    2. 训练 U-Net 预测噪声
    3. 生成新图像并可视化
    4. 保存模型

M5 GPU: 约 2 分钟跑完初步训练 (10 epochs, 32x32)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
NoiseSchedule = importlib.import_module("01_forward_diffusion").NoiseSchedule
SimpleUNet = importlib.import_module("02_unet_denoiser").SimpleUNet
DDPM = importlib.import_module("03_ddpm_trainer").DDPM
from utils.mps_utils import get_device, print_device_info


# ============================================================================
# 1. 数据准备
# ============================================================================
def get_cifar10_dataloaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """
    加载 CIFAR-10 数据集。

    CIFAR-10: 60,000 张 32×32 彩色图像, 10 个类别
    - 训练集: 50,000 张
    - 测试集: 10,000 张

    预处理:
        1. Resize → 32×32 (CIFAR-10 本来就是这个大小)
        2. ToTensor → [0, 1]
        3. Normalize → [-1, 1] (Diffusion 模型的标准输入范围)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                          # [0,255] → [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
        #                     均值          标准差
        # Normalize 的公式: (x - mean) / std
        # (x - 0.5) / 0.5 会把 [0,1] 映射到 [-1,1]
    ])

    train_dataset = datasets.CIFAR10(
        root="/tmp/cifar10",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="/tmp/cifar10",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                                   # 训练时打乱
        num_workers=0,                                  # MPS 上设为 0 避免 bug
        drop_last=True,                                 # 丢掉不完整的最后一个 batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    return train_loader, test_loader


# ============================================================================
# 2. 可视化生成的图像
# ============================================================================
def visualize_samples(samples: torch.Tensor, title: str, save_path: str):
    """
    把生成的一批图像排列成网格并保存。

    make_grid: 把 (N, C, H, W) → (C, Grid_H, Grid_W) 排列成网格
    """
    # 排列成 4×4 网格
    grid = make_grid(samples, nrow=4, normalize=False)
    # grid: (3, 4*32 + 3*2, 4*32 + 3*2) = (3, 134, 134) [padding=2]

    # 保存
    save_image(grid, save_path)
    print(f"  图像已保存: {save_path}")

    # 显示 (在 notebook 环境里)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())    # (C,H,W) → (H,W,C)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    display_path = save_path.replace(".png", "_display.png")
    plt.savefig(display_path, dpi=100)
    plt.close()


# ============================================================================
# 3. Main
# ============================================================================
def main():
    print("=" * 60)
    print("Diffusion 训练 CIFAR-10")
    print("=" * 60)

    device = get_device()
    print_device_info()

    # ── 超参数 (可调) ──
    BATCH_SIZE = 64          # 批量大小
    EPOCHS = 20              # 训练轮数 (20 轮约 5 分钟在 M5 上)
    LEARNING_RATE = 2e-4     # 学习率
    T = 1000                 # 扩散步数
    BASE_CHANNELS = 64       # U-Net 基础通道
    CHANNEL_MULT = [1, 2, 2] # 通道倍数 (下采样 2 次 → 8x 分辨率缩减)

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ── 数据 ──
    print(f"\n1. 加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10_dataloaders(BATCH_SIZE)
    print(f"   训练 batch 数: {len(train_loader)}")
    print(f"   测试 batch 数: {len(test_loader)}")

    # 看一眼真实数据
    real_images, _ = next(iter(train_loader))
    print(f"   图像形状: {real_images.shape}")          # (64, 3, 32, 32)
    print(f"   值范围: [{real_images.min():.2f}, {real_images.max():.2f}]")  # [-1, 1]

    # ── 模型 ──
    print(f"\n2. 构建模型...")
    schedule = NoiseSchedule(T=T)
    model = SimpleUNet(
        in_channels=3,
        base_channels=BASE_CHANNELS,
        channel_mult=CHANNEL_MULT,
    ).to(device)

    ddpm = DDPM(model, schedule, device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数总量: {total_params:,}")
    print(f"   扩散步数: {T}")

    # ── 训练 ──
    print(f"\n3. 开始训练 ({EPOCHS} epochs)...")
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            # 一步训练 (DDPM 封装的)
            loss = ddpm.train_step(images, optimizer)
            epoch_loss += loss

            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} — Avg Loss: {avg_loss:.4f}")

        # ── 每 5 个 epoch 生成一批样本看效果 ──
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  生成样本...")
            samples = ddpm.sample(
                n_samples=16,
                channels=3,
                height=32,
                width=32,
                show_progress=False,
            )
            visualize_samples(
                samples,
                f"CIFAR-10 Generated — Epoch {epoch + 1}",
                str(output_dir / f"cifar10_samples_epoch_{epoch + 1:03d}.png"),
            )

    # ── 保存模型 ──
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    save_path = checkpoint_dir / "ddpm_cifar10.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "schedule_T": T,
        "base_channels": BASE_CHANNELS,
        "channel_mult": CHANNEL_MULT,
        "train_losses": train_losses,
    }, save_path)
    print(f"\n4. 模型已保存: {save_path}")

    # ── 最终生成 ──
    print(f"\n5. 最终生成 64 张图像...")
    final_samples = ddpm.sample(n_samples=64, channels=3, height=32, width=32)
    visualize_samples(
        final_samples,
        "CIFAR-10 Final Generation",
        str(output_dir / "cifar10_final_samples.png"),
    )

    # ── Draw loss curve ──
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss — DDPM on CIFAR-10")
    plt.grid(True, alpha=0.3)
    loss_path = output_dir / "cifar10_loss_curve.png"
    plt.savefig(loss_path, dpi=100)
    plt.close()
    print(f"   Loss 曲线已保存: {loss_path}")

    print("\n" + "=" * 60)
    print("✅ Diffusion CIFAR-10 训练完成！")
    print(f"生成的图像在: {output_dir}")
    print("你已经从零实现了 Diffusion Model!")
    print("=" * 60)

    return model, ddpm, train_losses


if __name__ == "__main__":
    main()
