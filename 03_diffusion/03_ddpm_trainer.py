"""
DDPM Trainer — 训练 + 采样
============================
DDPM (Denoising Diffusion Probabilistic Models) 的完整训练和采样流程。

训练:
    1. 取真实图像 x_0
    2. 随机选时间步 t
    3. 加噪声: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
    4. 网络预测: ε_θ(x_t, t)
    5. Loss = MSE(ε_θ, ε) ← 让预测的噪声尽量接近真实噪声

采样 (生成):
    1. 从纯噪声开始: x_T ~ N(0, I)
    2. 逐步去噪: for t = T, T-1, ..., 1:
        x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
    3. 最后 x_0 就是一个新生成的图像!

为什么预测噪声而不是预测图像?
    预测噪声: 只需要预测一个 N(0,I) 的噪声(= 回归任务)
    预测图像: 需要想象完整的图像内容(= 难度大得多)
    实验证明: 预测噪声的效果显著更好!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import importlib
NoiseSchedule = importlib.import_module("01_forward_diffusion").NoiseSchedule
forward_diffusion = importlib.import_module("01_forward_diffusion").forward_diffusion
SimpleUNet = importlib.import_module("02_unet_denoiser").SimpleUNet

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
# 1. DDPM Trainer — 训练循环
# ============================================================================
class DDPM:
    """
    DDPM 的完整实现，包含训练和采样。

    用法:
        ddpm = DDPM(model, schedule, device)
        ddpm.train_step(x_0)           # 一步训练
        samples = ddpm.sample(n=16)    # 生成 16 张新图像
    """

    def __init__(self, model: nn.Module, schedule: NoiseSchedule, device: torch.device):
        self.model = model
        self.schedule = schedule
        self.T = schedule.T
        self.device = device

        # 预计算采样时需要反复用到的系数 (全部搬到 GPU/MPS)
        self.betas = schedule.betas.to(device)
        self.alphas = schedule.alphas.to(device)
        self.alphas_bar = schedule.alphas_bar.to(device)

        # 采样公式里的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/√α_t

        # x_t → x_{t-1} 的方差 (DDPM 论文中 σ_t^2 = β_t * (1-ᾱ_{t-1})/(1-ᾱ_t))
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)  # 前面补一个 1
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )

    def train_step(self, x_0: torch.Tensor, optimizer: optim.Optimizer) -> float:
        """
        一步训练:
            1. 随机采时间步 t
            2. 加噪声 → x_t
            3. 网络预测噪声 → ε_θ
            4. loss = MSE(ε_θ, ε)

        Args:
            x_0: (B, C, H, W) — 干净图像
            optimizer: 优化器

        Returns:
            loss 值 (float)
        """
        B = x_0.shape[0]

        # 随机采样时间步 t ∈ [0, T)
        t = torch.randint(0, self.T, (B,), device=self.device)

        # 正向扩散: 加噪声
        x_t, noise = forward_diffusion(x_0, t, self.schedule)

        # 网络预测噪声
        predicted_noise = self.model(x_t, t)

        # Loss: 预测噪声 vs 真实噪声
        loss = F.mse_loss(predicted_noise, noise)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        采样 (生成新图像) — DDPM 的终极目标。

        算法 (DDPM Sampling):
            x_T ~ N(0, I)                          # 从纯噪声开始
            for t = T-1, ..., 0:                   # 逐步去噪
                z ~ N(0, I) if t > 0 else 0       # 随机噪声 (最后一步不加)
                x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t))
                          + √(σ_t^2) * z          # 加一点随机噪声
            return x_0                              # 生成的图像!

        直观理解:
            从一团噪声开始，逐步去掉噪声，每次都加一点新的小噪声
            → 就像雕塑家: 从石头里"释放"出图像

        Args:
            n_samples: 生成几张图
            channels:  图像通道数
            height:    图像高度
            width:     图像宽度

        Returns:
            (n_samples, C, H, W) — 生成的图像
        """
        self.model.eval()

        # 从纯噪声开始
        x_t = torch.randn(n_samples, channels, height, width, device=self.device)

        # 逐步去噪 (从 T-1 降到 0)
        iterator = range(self.T - 1, -1, -1)
        if show_progress:
            iterator = tqdm(iterator, desc="采样/生成中", leave=False)

        for t in iterator:
            # 当前时间步的 batch
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.model(x_t, t_batch)

            # DDPM 去噪公式 (Algorithm 2 from DDPM paper):
            # x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ) + σ_t * z

            # 第一项: 预测的均值
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_bar[t]
            beta_t = self.betas[t]

            # 1/√α_t
            coef1 = 1.0 / torch.sqrt(alpha_t)

            # (1-α_t)/√(1-ᾱ_t) — 噪声的系数
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

            # 均值
            mean = coef1 * (x_t - coef2 * predicted_noise)

            # 噪声项: σ_t * z (t=0 时没有噪声)
            if t > 0:
                noise = torch.randn_like(x_t)
                var_t = self.posterior_variance[t]
                x_t = mean + torch.sqrt(var_t) * noise
            else:
                x_t = mean

        # 把值域映射到 [0, 1] (像素范围)
        x_t = x_t.clamp(-1, 1).add(1).div(2)        # [-1,1] → [0,1]

        return x_t

    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples: int,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        ddim_steps: int = 50,                        # 只用 50 步而不是 1000 步!
    ) -> torch.Tensor:
        """
        ⚡ DDIM 采样 — 加速版 (Denoising Diffusion Implicit Models)

        普通 DDPM 采样需要 1000 步，很慢。
        DDIM 通过数学推导，发现可以跳步采样:
            - 只用 50 步就能达到接近 1000 步的质量
            - 50x 加速!

        原理: DDIM 把采样变成了一个确定性的 ODE 求解过程
              → 可以用更大的步长 → 不需要每一步都加随机噪声

        这是 Stable Diffusion 等现代扩散模型使用的加速方法。
        """
        self.model.eval()

        # 时间步: 均匀间隔取 ddim_steps 个
        # 例如 T=1000, ddim_steps=50 → t = [980, 960, 940, ..., 0]
        step_size = self.T // ddim_steps
        timesteps = list(range(self.T - 1, -1, -step_size))

        x_t = torch.randn(n_samples, channels, height, width, device=self.device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x_t, t_batch)

            # DDIM 确定性的更新 (不加随机噪声)
            alpha_bar_t = self.alphas_bar[t]
            alpha_bar_next = self.alphas_bar[t_next]

            # 预测 x_0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            # 预测方向
            dir_xt = torch.sqrt(1 - alpha_bar_next) * predicted_noise

            # DDIM 更新
            x_t = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt

        return x_t.clamp(-1, 1).add(1).div(2)


# ============================================================================
# 2. 训练器 — 封装完整训练流程
# ============================================================================
class DDPMTrainer:
    """封装训练循环，方便调用"""

    def __init__(
        self,
        ddpm: DDPM,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.ddpm = ddpm
        self.optimizer = optimizer
        self.device = device

    def train_epochs(
        self,
        dataloader,
        epochs: int = 10,
        sample_every: int = 2,
        sample_dir: str = None,
    ) -> list[float]:
        """
        完整训练循环。

        Returns:
            losses: 每个 epoch 的平均 loss
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                # dataloader 返回 (images, labels) 或只是 images
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                loss = self.ddpm.train_step(images, self.optimizer)
                epoch_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs} avg loss: {avg_loss:.4f}")

            # 每几个 epoch 生成样本看看效果
            if sample_dir and (epoch % sample_every == 0 or epoch == epochs - 1):
                sample_dir_path = Path(sample_dir)
                sample_dir_path.mkdir(parents=True, exist_ok=True)
                samples = self.ddpm.sample(n_samples=16, show_progress=False)
                torch.save(samples.cpu(), sample_dir_path / f"samples_epoch_{epoch + 1}.pt")

        return losses


if __name__ == "__main__":
    print("=" * 60)
    print("DDPM Trainer 演示")
    print("=" * 60)

    device = get_device()

    # ── 构建模型 ──
    schedule = NoiseSchedule(T=1000)
    model = SimpleUNet(in_channels=3, base_channels=64, channel_mult=[1, 2, 4])
    model = model.to(device)

    ddpm = DDPM(model, schedule, device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"DDPM 步数:  {schedule.T}")

    # ── 训练一步 (快速验证) ──
    print(f"\n测试一步训练:")
    x_0 = torch.randn(8, 3, 32, 32, device=device)   # 假数据
    loss = ddpm.train_step(x_0, optimizer)
    print(f"  假数据 loss: {loss:.4f}")
    print(f"  ✅ 训练步骤 OK")

    # ── 采样 (生成一批纯噪声的 "图像"，因为模型没训练) ──
    print(f"\n测试采样 (未训练的模型):")
    samples = ddpm.sample(n_samples=4, show_progress=True)
    print(f"  生成形状: {samples.shape}")
    print(f"  值范围: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  ℹ️ 因为是未训练的模型，输出会是随机的")

    # ── DDIM 采样 ──
    print(f"\n测试 DDIM 加速采样:")
    samples_ddim = ddpm.sample_ddim(n_samples=4, ddim_steps=50)
    print(f"  生成形状: {samples_ddim.shape}")
    print(f"  DDIM 只用 50 步 (vs DDPM 的 1000 步)")
    print(f"  加速比: 1000/50 = 20x!")

    print("\n" + "=" * 60)
    print("✅ DDPM Trainer 完成！")
    print("训练: x_0 → x_t(加噪) → predict ε → loss = MSE(pred_ε, ε)")
    print("采样: x_T(N(0,I)) → denoise step by step → x_0 (新图像!)")
    print("下一步: 运行 04_train_on_cifar10.py 训练真实图像生成!")
    print("=" * 60)
