"""
World Model — 学习预测世界的未来
===================================
世界模型 = 给定当前观测 + 动作，预测下一帧观测。

    当前帧 o_t + 动作 a_t → [World Model] → 预测下一帧 ô_{t+1}

为什么需要世界模型？
    1. 规划: "如果我做这个动作，世界会变成什么样？" → 选最好的动作
    2. Imagination: 不需要真实环境，在 "想象" 中训练策略
    3. 样本效率: 真环境交互很贵 → 世界模型里 "做梦" 训练 → 零成本

核心组件 (Dreamer/RSSM 简化版):
    Encoder:     o_t → z_t (观测→潜状态)         ← 你 01_vae.py 的 encoder
    Dynamics:    z_t, a_t → ẑ_{t+1} (预测下一潜状态) ← 核心!
    Decoder:     z_t → ô_t (潜状态→重建观测)      ← 你 01_vae.py 的 decoder

损失:
    L = ||o_t - ô_t||² + ||z_{t+1} - ẑ_{t+1}||²
        └─重建观测─┘   └────预测下一状态────┘

这和 VLA 的关系:
    VLA:  vision+lang → action
    WM:   vision+action → next_vision
    → 合在一起: 完整的感知→决策→预测 闭环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
class WorldModel(nn.Module):
    """
    极简世界模型。3 个组件:
        Encoder:  图像 → 潜状态 z
        Dynamics: 潜状态 z + 动作 a → 下一潜状态 ẑ_next
        Decoder:  潜状态 z → 重建图像

    这是 Dreamer / RSSM 的极简版（去掉了 stochastic state，只留确定性）。
    """

    def __init__(self, obs_dim: int = 64, act_dim: int = 3, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder: o_t → z_t ──
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # ── Dynamics: z_t, a_t → ẑ_{t+1} ──
        # 这是世界模型的核心: 学会 "动作如何改变世界"
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # ── Decoder: z_t → ô_t ──
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, obs_dim),
        )

        # ── Reward Predictor (可选): z_t → r̂_t ──
        self.reward_head = nn.Linear(latent_dim, 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)                        # (B, latent)

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, action], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, obs, action, next_obs):
        """训练一步: 编码→预测→算 loss"""
        z = self.encode(obs)
        z_next_pred = self.predict_next(z, action)
        obs_recon = self.decode(z)

        # 3 个 loss
        recon_loss = F.mse_loss(obs_recon, obs)                  # 重建当前
        pred_loss = F.mse_loss(z_next_pred,
                               self.encode(next_obs).detach())    # 预测下一帧潜状态
        #  ^ .detach(): 不让 next_obs 的编码参与梯度（和 Dreamer 一致）

        return recon_loss + pred_loss, z

    @torch.no_grad()
    def imagine(self, init_obs: torch.Tensor, actions: torch.Tensor):
        """
        在想象中 rollout: 给初始观测 + 动作序列 → 预测未来的观测序列。

        这是 model-based RL 的核心: 不需要环境，纯"想象"中规划!
        """
        B, T = actions.shape[:2]
        z = self.encode(init_obs)
        imagined_obs = []

        for t in range(T):
            a = actions[:, t]
            z = self.predict_next(z, a)                # 用预测的 z 自回归 roll-out
            imagined_obs.append(self.decode(z))

        return torch.stack(imagined_obs, dim=1)         # (B, T, obs_dim)


# ============================================================================
if __name__ == "__main__":
    device = get_device()
    print(f"设备: {device}")

    obs_dim = 64       # 观测特征 (如 8×8 的 grid world 编码)
    act_dim = 3        # 动作 (如 xyz 速度)
    latent_dim = 32

    wm = WorldModel(obs_dim, act_dim, latent_dim).to(device)
    print(f"World Model: {obs_dim}d obs → {latent_dim}d latent → {obs_dim}d recon")

    # 训练
    B = 16
    obs = torch.randn(B, obs_dim, device=device)
    act = torch.randn(B, act_dim, device=device)
    next_obs = torch.randn(B, obs_dim, device=device)

    loss, z = wm(obs, act, next_obs)
    print(f"\n单步训练:")
    print(f"  o_t:     {obs.shape}")
    print(f"  z_t:     {z.shape}     ← 压缩 {obs_dim/latent_dim:.0f}x")
    print(f"  ô_t:     {wm.decode(z).shape}  ← 重建")
    print(f"  â_{latent_dim}: {z.square().mean():.3f} ← 潜状态 L2 范数")
    print(f"  Loss: {loss.item():.4f}")

    # 想象模式
    init_obs = torch.randn(1, obs_dim, device=device)
    actions = torch.randn(1, 20, act_dim, device=device)     # 20 步动作
    imagined = wm.imagine(init_obs, actions)
    print(f"\nImagination rollout:")
    print(f"  输入: 初始观测 + 20步随机动作")
    print(f"  输出: {imagined.shape}  ← 预测的20步未来观测序列")
    print(f"  这就是 Model-Based RL 的 '做梦' 训练!")

    print(f"\n✅ World Model OK")
    print(f"  完整闭环: 感知→规划(在想象中)→行动→更新世界模型")
