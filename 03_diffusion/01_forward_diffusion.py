"""
Forward Diffusion — 逐步把真实图像变成纯噪声
==================================================
Diffusion Model 的核心思想:
    1. 正向过程 (Forward): 逐步给图像加噪声 → 最终变成纯噪声 (公式化, 不需要学习)
    2. 反向过程 (Reverse): 训练一个网络从噪声中恢复图像 (需要学习)

本文件实现正向过程 — DDPM (Denoising Diffusion Probabilistic Models) 的数学。

直观理解:
    就像在一杯清水里逐渐滴入墨水:
        t=0:    清水 (真实图像)
        t=10:   浅灰 (少许噪声)
        t=50:   灰 (中等噪声)
        t=100:  深灰 (大量噪声)
        t=T:    纯墨 (纯噪声)

    Diffusion 就是学习 "从墨水回到清水" 的过程!

数学 (DDPM):
    q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)
    即: 每一步在上一步的基础上加一点高斯噪声

    关键优化 (重参数化技巧):
    q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)
    可以直接从 x_0 算出 x_t，不用一步步迭代!
    其中: α_t = 1 - β_t, ᾱ_t = Π_{s=1}^{t} α_s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. Noise Schedule — 定义每一步加多少噪声
# ============================================================================
class NoiseSchedule:
    """
    噪声调度: 控制 β_t (每一步的噪声量)。

    β_t 的设计:
        - 线性: β_t 从 small 线性增加到 large (原始 DDPM)
        - 余弦: β_t 按余弦曲线变化 (改进版，效果更好)

    直觉:
        前几步: 加很少噪声 (β 小)，保留图像结构
        后几步: 加很多噪声 (β 大)，快速走向纯噪声
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        """
        Args:
            T: 总步数 (DDPM 用 1000)
            beta_start: 初始噪声量 (T=0 时)
            beta_end:   最终噪声量 (T=T 时)
        """
        self.T = T

        # Beta schedule (线性)
        self.betas = torch.linspace(beta_start, beta_end, T)       # (T,)

        # Alpha: α_t = 1 - β_t
        self.alphas = 1.0 - self.betas                             # (T,)

        # Alpha bar (累积): ᾱ_t = Π_{s=1}^{t} α_s
        # cumprod = cumulative product (累积乘积)
        # 例如: [0.9, 0.8, 0.7] → cumprod = [0.9, 0.72, 0.504]
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)        # (T,)

        # 用得上的预计算值
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

    def __repr__(self):
        return (f"NoiseSchedule(T={self.T}, "
                f"β: [{self.betas[0].item():.4f}, {self.betas[-1].item():.4f}], "
                f"ᾱ: [{self.alphas_bar[0].item():.4f}, {self.alphas_bar[-1].item():.4f}])")


# ============================================================================
# 2. Forward Diffusion — 把图像变噪声
# ============================================================================
def forward_diffusion(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    正向扩散: 给定干净图像 x_0 和时间步 t，返回加了噪声的图像 x_t 和噪声 ε。

    公式 (重参数化):
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε

    这步是 O(1) 的 — 不需要循环 T 步，直接算!

    Args:
        x_0:      (B, C, H, W) — 干净图像
        t:        (B,) — 每条数据的时间步
        schedule: noise schedule

    Returns:
        x_t: (B, C, H, W) — 加了噪声的图像
        noise: (B, C, H, W) — 加入的噪声 (训练目标是预测这个!)
    """
    # 采样标准正态噪声
    noise = torch.randn_like(x_0)

    # 从 schedule 取对应的 ᾱ_t 值
    # t 是 (B,) → 需要 reshape 成 (B, 1, 1, 1) 才能做广播
    # ⚠️ schedule 的 tensor 在 CPU 上，t 可能在 GPU 上 → t 先搬回 CPU 做索引
    device = x_0.device
    t_cpu = t.cpu()
    sqrt_alpha_bar = schedule.sqrt_alphas_bar[t_cpu].to(device).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alphas_bar[t_cpu].to(device).view(-1, 1, 1, 1)

    # 重参数化: 一步到位
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    return x_t, noise


# ============================================================================
# 3. 可视化: 看噪声怎么一步步吞噬图像
# ============================================================================
def visualize_forward_diffusion():
    """
    用一张假图像展示正向扩散的过程。
    从 t=0 (干净) 到 t=T (纯噪声)。
    """
    T = 1000
    schedule = NoiseSchedule(T)

    # 创建一张 "图像" (32x32 的彩色条带，方便观察)
    H, W = 32, 32
    x_0 = torch.zeros(3, H, W)
    # 颜色条带: 红 → 绿 → 蓝
    for i in range(H):
        if i < H // 3:
            x_0[0, i, :] = 1.0                      # 红色条带
        elif i < 2 * H // 3:
            x_0[1, i, :] = 1.0                      # 绿色条带
        else:
            x_0[2, i, :] = 1.0                      # 蓝色条带

    x_0 = x_0.unsqueeze(0)                           # (1, 3, 32, 32)

    # 在不同的 t 时刻做 forward diffusion
    t_steps = [0, 10, 50, 100, 300, 500, 800, 999]
    fig, axes = plt.subplots(1, len(t_steps), figsize=(14, 3))

    for ax, t_val in zip(axes, t_steps):
        t = torch.tensor([t_val])
        x_t, _ = forward_diffusion(x_0, t, schedule)

        # 把 tensor 转成图像格式
        img = x_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(f"t={t_val}\nᾱ={schedule.alphas_bar[t_val]:.3f}")
        ax.axis("off")

    fig.suptitle("Forward Diffusion: 图像 → 噪声\n(从干净图像逐步加噪到纯噪声)",
                 fontsize=14)
    plt.tight_layout()
    save_path = "/tmp/forward_diffusion.png"
    plt.savefig(save_path, dpi=100)
    print(f"  图片保存到: {save_path}")
    plt.close()


# ============================================================================
# 4. 验证: 噪声预测 → 还原 x_0 (理论验证)
# ============================================================================
def verify_reconstruction():
    """
    验证: 如果我们知道真实噪声，能否从 x_t 完美还原 x_0？

    从 forward 公式反推:
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        → x_0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)

    这说明: 如果网络能准确预测 ε，就能完美还原图像!
    DDPM 的训练目标: 让网络预测 ε (噪声)，而不是直接预测 x_0
    (预测噪声比预测图像更容易 — 这是 DDPM 的核心洞察)
    """
    T = 1000
    schedule = NoiseSchedule(T)
    x_0 = torch.randn(1, 3, 32, 32)                 # 随机 "图像"
    t = torch.tensor([500])                           # 中间一步

    # 正向: 加噪声
    x_t, noise = forward_diffusion(x_0, t, schedule)

    # 反向: 如果知道 noise，能不能还原 x_0?
    sqrt_alpha_bar = schedule.sqrt_alphas_bar[500]
    sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alphas_bar[500]
    x_0_reconstructed = (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar

    mse = F.mse_loss(x_0, x_0_reconstructed)
    print(f"  还原误差 (MSE): {mse.item():.8f} — 应该接近 0")
    print(f"  ✅ 如果知道噪声，可以完美还原图像!")
    print(f"  → 所以训练目标是让网络学会预测噪声")


if __name__ == "__main__":
    print("=" * 60)
    print("Forward Diffusion 演示")
    print("=" * 60)

    # 创建 noise schedule
    schedule = NoiseSchedule(T=1000)
    print(f"\n{schedule}")

    # 验证: 从 x_0 到 x_t
    print(f"\n1. Forward Diffusion 测试:")
    x_0 = torch.randn(4, 3, 32, 32)                 # 4 张 "图像"
    t = torch.randint(0, 1000, (4,))                # 随机时间步
    x_t, noise = forward_diffusion(x_0, t, schedule)
    print(f"   x_0 shape: {x_0.shape}")
    print(f"   x_t shape: {x_t.shape}")
    print(f"   noise shape: {noise.shape}")
    print(f"   ✅ Forward OK")

    # 验证: 极限情况
    print(f"\n2. 极限验证:")
    # t=0: x_t 应该接近 x_0
    t0 = torch.zeros(4, dtype=torch.long)
    x_t0, _ = forward_diffusion(x_0, t0, schedule)
    print(f"   t=0: MSE(x_t, x_0) = {F.mse_loss(x_t0, x_0).item():.6f} (应该 ≈ 0)")

    # t=T-1: x_t 应该接近纯噪声 N(0,1)
    tT = torch.full((4,), schedule.T - 1, dtype=torch.long)
    x_tT, _ = forward_diffusion(x_0, tT, schedule)
    print(f"   t=T: mean(x_t)={x_tT.mean().item():.4f} (应该 ≈ 0), "
          f"std(x_t)={x_tT.std().item():.4f} (应该 ≈ 1)")

    # 重构验证
    print(f"\n3. 噪声→图像 还原验证:")
    verify_reconstruction()

    # 可视化 (可选)
    print(f"\n4. 可视化 Forward Diffusion:")
    try:
        visualize_forward_diffusion()
    except Exception as e:
        print(f"   可视化跳过: {e}")

    print("\n" + "=" * 60)
    print("✅ Forward Diffusion 完成！")
    print("核心公式: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε")
    print("训练目标: 预测噪声 ε (不是直接预测 x_0)")
    print("下一步: 运行 02_unet_denoiser.py")
    print("=" * 60)
