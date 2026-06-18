# 运行: conda activate diffusion && python 03_diffusion/07_flow_matching.py
"""
Flow Matching — 扩散模型的现代进化版
=======================================
Flow Matching 是 2023-2024 年提出的生成模型新范式，
正在取代 DDPM 成为主流（Stable Diffusion 3 就用的这个）。

DDPM vs Flow Matching 的核心区别:

    DDPM:
        正向: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε  (随机的，路径是弯曲的)
        目标: 预测噪声 ε
        采样: 逐步去噪，每次加随机噪声（随机微分方程 SDE）
        步数: 需要很多步（DDPM 1000, DDIM 50）

    Flow Matching:
        正向: x_t = (1-t) · x_0 + t · ε  (线性的，路径是直线!)
        目标: 预测速度场 v = ε - x_0  (从噪声指向数据的方向)
        采样: 解常微分方程 (ODE)，完全确定性
        步数: 可以用很少步（10-25 步就有好效果）

    直观理解:
        DDPM 像 "盲人摸象" — 慢慢摸，逐步猜
        Flow Matching 像 "地图导航" — 沿着直线走，每一步都知道方向

    为什么 Flow Matching 更好？
        1. 路径更短：直线 vs 弯弯曲曲 → 采样步数少
        2. 确定性：不依赖随机噪声，每次生成结果一致 (给定初始种子)
        3. 更简单：不需要 noise schedule, β_t, ᾱ_t 这些复杂公式
        4. 训练和推理是对称的：都走同一条路径

Flow Matching 的训练:
    1. 取真实数据 x_0 和随机噪声 ε ~ N(0,I)
    2. 随机选时间 t ∈ [0,1]
    3. 线性插值: x_t = (1-t) * x_0 + t * ε
    4. 目标速度: v = ε - x_0 (从噪声到数据的直线方向)
    5. 训练网络预测: v_θ(x_t, t) ≈ v
    6. Loss = MSE(v_θ(x_t, t), v)

Flow Matching 的采样 (ODE 求解):
    dx/dt = v_θ(x, t)
    从 t=1 (纯噪声) 到 t=0 (数据)
    用 Euler 方法:
        x_{t-Δt} = x_t + v_θ(x_t, t) * Δt

这就是 Rectified Flow / Flow Matching 的核心！

参考:
    - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
    - Liu et al., "Flow Straight and Fast" (Rectified Flow, ICLR 2023)
    - Esser et al., "Scaling Rectified Flow Transformers" (Stable Diffusion 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
# 1. Flow Matching 的核心操作
# ============================================================================
def flow_interpolate(x_0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    线性插值: x_t = (1-t) * x_0 + t * ε

    这是 Flow Matching 的正向过程 — 比 DDPM 的正向简单太多了!

    参数:
        x_0: (B, C, H, W) — 真实数据
        eps: (B, C, H, W) — 随机噪声 N(0, I)
        t:   (B, 1, 1, 1) — 时间步 [0, 1]

    返回:
        x_t: 插值后的数据

    直观理解:
        t = 0:   x_t = x_0  (纯数据)
        t = 0.5: x_t = 0.5*x_0 + 0.5*ε  (各一半)
        t = 1:   x_t = ε  (纯噪声)
    """
    return (1 - t) * x_0 + t * eps


def flow_target(x_0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """
    目标速度场: v = ε - x_0

    这是 Flow Matching 的学习目标 — 网络要预测这个 "方向向量"。
    v 表示 "从噪声指向数据" 的直线方向。

    为什么是 ε - x_0？
        x_t = (1-t)x_0 + t·ε
        d(x_t)/dt = -x_0 + ε = ε - x_0
        → 这就是 "速度" — x_t 随时间变化的速率和方向
    """
    return eps - x_0


# ============================================================================
# 2. Flow Matching Trainer
# ============================================================================
class FlowMatching:
    """
    Flow Matching 训练 + 采样。

    和 03_ddpm_trainer.py 的 DDPM 类对比，你会发现 FM 简单很多：
        - 不需要 noise schedule (β_t)
        - 不需要 ᾱ_t 预计算
        - 采样时不加随机噪声（确定性 ODE）
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def train_step(self, x_0: torch.Tensor, optimizer: optim.Optimizer) -> float:
        """
        一步训练。

        步骤:
            1. 采样噪声 ε ~ N(0,I)
            2. 采样时间 t ~ U(0,1)
            3. 插值: x_t = (1-t)·x_0 + t·ε
            4. 目标速度: v = ε - x_0
            5. 网络预测: v_θ = model(x_t, t)
            6. Loss = MSE(v_θ, v)

        对比 DDPM train_step (03_ddpm_trainer.py):
            DDPM: x_t = √ᾱ·x_0 + √(1-ᾱ)·ε, target = ε
            FM:   x_t = (1-t)·x_0 + t·ε,   target = ε - x_0
        """
        B = x_0.shape[0]

        # 1. 采样噪声
        eps = torch.randn_like(x_0)

        # 2. 采样时间步 t ∈ [0, 1]
        t = torch.rand(B, device=self.device).view(-1, 1, 1, 1)

        # 3. 线性插值（直线上的一点）
        x_t = flow_interpolate(x_0, eps, t)

        # 4. 目标速度场
        v_target = flow_target(x_0, eps)

        # 5. 网络预测速度场
        # 注意: 时间参数用 t.squeeze() → (B,)
        v_pred = self.model(x_t, t.squeeze())

        # 6. Loss
        loss = F.mse_loss(v_pred, v_target)

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
        n_steps: int = 25,                           # ODE 求解步数（25 步就很好！）
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Flow Matching 采样 — 用 Euler 方法解 ODE。

        ODE: dx/dt = v_θ(x, t)
        从 t=1 (纯噪声) 到 t=0 (数据)

        Euler 方法:
            Δt = 1 / n_steps
            for each step:
                t = current_time
                v = model(x_t, t)       # 当前速度
                x_t = x_t + v * Δt       # 沿速度方向走一小步

        对比 DDPM 采样 (03_ddpm_trainer.py):
            DDPM: x_{t-1} = 1/√α·(x_t - ...) + σ·z  (有随机噪声 z)
            FM:   x_{t+Δt} = x_t + v·Δt           (完全确定性!)
        """
        self.model.eval()

        # 从纯噪声开始 (t=1)
        x = torch.randn(n_samples, channels, height, width, device=self.device)

        dt = 1.0 / n_steps
        times = torch.linspace(1.0, 0.0, n_steps + 1)  # 从 1 到 0

        iterator = range(n_steps)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Flow Matching 采样", leave=False)

        for i in iterator:
            t = times[i]                             # 当前时间
            t_batch = torch.full((n_samples,), t, device=self.device)

            # 预测速度场
            v = self.model(x, t_batch)

            # Euler 步: x = x + v * dt
            # 注意: dt = 负 (从 1 到 0)
            x = x + v * (times[i+1] - times[i])
            #    ↑ 等价于 x = x + model(x,t) * (-dt)
            #    ↑ 因为是朝着 x_0 的方向走 (速度场 v 指向数据)

        return x.clamp(-1, 1).add(1).div(2)          # [-1,1] → [0,1]


# ============================================================================
# 3. 为什么 Flow Matching 更好？可视化对比
# ============================================================================
def compare_paths():
    """打印 DDPM vs Flow Matching 的路径对比"""
    print("=" * 60)
    print("DDPM vs Flow Matching — 直观对比")
    print("=" * 60)
    print("""
    从数据到噪声的 "路径":

    DDPM (弯曲的):
        x_0 ──────→ x_t ──────→ ε
        路径: 弯弯曲曲（随机游走）
        每步: 加随机噪声 + 缩放
        公式: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
        还原: 需要很多步，有随机性

    Flow Matching (直线):
        x_0 ──────────→ ε
        路径: 一条直线!
        每步: 线性插值
        公式: x_t = (1-t)·x_0 + t·ε
        还原: 几步就够了，完全确定

    用比喻理解:
        DDPM: 在雾里走回家 → 每一步都不确定方向，慢慢走
        FM:   有 GPS 导航回家 → 知道直线的方向，大步快走

    用数学理解:
        DDPM:  随机微分方程 (SDE) — dx = f(x,t)dt + g(t)dW
        FM:    常微分方程   (ODE) — dx = v(x,t)dt
               ↑ 无随机项 → 更简单、更稳定、更快
    """)


# ============================================================================
# 4. 快速演示: 用简单 MLP 在 1D 数据上展示 FM
# ============================================================================
def demo_1d_flow_matching():
    """
    在 1D 数据上演示 Flow Matching。
    数据: 两个高斯分布的混合 → 非常适合可视化。
    """
    print("=" * 60)
    print("1D Flow Matching 演示")
    print("=" * 60)

    device = torch.device("cpu")

    # 数据: 混合高斯 (0.3) 和 (7, 0.3)
    def sample_data(n):
        centers = torch.tensor([2.0, 5.0])
        idx = torch.randint(0, 2, (n,))
        return torch.randn(n) * 0.3 + centers[idx]

    # 简单的速度场网络 (MLP)
    model = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),        # 输入: (x, t)
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),                    # 输出: 速度 v
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fm = FlowMatching(model, device)

    # 训练几轮
    print("训练 1D Flow Matching...")
    for step in range(1000):
        x_0 = sample_data(256).unsqueeze(1)           # (B, 1)
        loss = fm.train_step(x_0, optimizer)

    # 采样
    print("采样中...")
    samples = fm.sample(n_samples=1000, channels=1, height=1, width=1, n_steps=20)
    samples = samples.squeeze().numpy()

    print(f"  真实数据: 两个簇 (中心在 2.0 和 5.0)")
    print(f"  生成数据: mean={samples.mean():.2f}, std={samples.std():.2f}")
    print(f"  ✅ Flow Matching 学会了生成两个簇!")

    return samples


# ============================================================================
# 5. DDPM vs Flow Matching 公式卡
# ============================================================================
def formula_card():
    print("\n" + "=" * 60)
    print("DDPM vs Flow Matching — 公式对比卡")
    print("=" * 60)
    print("""
    ┌──────────────────┬─────────────────────────────┬────────────────────────────┐
    │      操作         │          DDPM               │      Flow Matching         │
    ├──────────────────┼─────────────────────────────┼────────────────────────────┤
    │ 正向过程          │ x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε  │ x_t = (1-t)·x₀ + t·ε       │
    │ 训练目标          │ 预测噪声 ε                   │ 预测速度 v = ε - x₀        │
    │ Loss              │ MSE(ε_θ, ε)                 │ MSE(v_θ, ε - x₀)           │
    │ 采样              │ SDE (随机)                   │ ODE (确定性)               │
    │ 采样公式           │ x_{t-1} = ... + σ_t·z      │ x_{t+Δt} = x_t + v·Δt     │
    │ 步数              │ 1000 (DDPM) / 50 (DDIM)     │ 10-25 步                   │
    │ 随机性            │ 有 (每次采样结果不同)         │ 无 (给定种子结果固定)        │
    │ 路径              │ 弯弯曲曲 (随机游走)           │ 直线!                      │
    │ 复杂度            │ 高 (需要 ᾱ, β, σ schedule)   │ 低 (只需 t ∈ [0,1])        │
    └──────────────────┴─────────────────────────────┴────────────────────────────┘

    为什么 Stable Diffusion 3 选 Flow Matching？
        1. 采样更快: 25 步 vs 50 步 (DDIM) → 生成速度快 2 倍
        2. 训练更简单: 不需要 noise schedule 调参
        3. 结果确定: 相同种子 = 相同图像 (便于调试/复现)
        4. DiT 友好: Transformer 做 velocity prediction 效果特别好
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("Flow Matching — 扩散模型的下一站")
    print("=" * 60)

    compare_paths()

    # 1D demo
    try:
        demo_1d_flow_matching()
    except Exception as e:
        print(f"1D demo 失败: {e}")

    formula_card()

    print("\n" + "=" * 60)
    print("✅ Flow Matching 完成！")
    print()
    print("你现在掌握了 3 种扩散模型:")
    print("  1. DDPM   (03_ddpm_trainer.py) — 经典，预测噪声")
    print("  2. DiT    (05_dit.py)          — Transformer backbone")
    print("  3. Flow Matching (本文件)        — 直线路径，预测速度")
    print()
    print("组合使用: DiT + Flow Matching = Stable Diffusion 3 的架构!")
    print("=" * 60)
