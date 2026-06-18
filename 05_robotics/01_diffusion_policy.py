"""
Diffusion Policy — 用扩散模型生成机器人动作
================================================
你之前学的 Diffusion 是 去噪图像（x ∈ R^{C×H×W}）。
Diffusion Policy 的创新: 去噪动作轨迹（a ∈ R^{T×D}）

    DDPM:  x_T(纯噪声) → ... → x_0(图像)
    DP:    a_T(纯噪声) → ... → a_0(动作序列)

为什么用 Diffusion 生成动作？
    1. 多模态: 机器人可以有很多种方式完成任务 → Diffusion 天然多模态
    2. 时序一致性: 生成的 T 步动作天然平滑（因为是同时去噪的）
    3. 和图像 Diffusion 完全一样的训练: forward噪声 + predict噪声

架构:
    观测 o_t(图像/状态) → 条件
    动作序列 a_{t:t+H}  → 扩散目标
    网络: 输入 (a_noisy, t, o_cond) → 输出 去噪动作

训练和推理:
    训练: a_noisy = √ᾱ·a + √(1-ᾱ)·ε,  predict ε → 和 DDPM 一样!
    推理: 从纯噪声开始，逐步去噪 → 得到动作序列 → 执行前几步 → re-plan

参考:
    Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
class ActionDenoiser(nn.Module):
    """
    动作去噪网络: 输入(噪声动作, t, 观测), 输出(预测的噪声)

    和 U-Net 的关系:
        U-Net 输入 (噪声图像, t) → 预测图像噪声 (空间卷积)
        这个输入 (噪声动作, t, 观测) → 预测动作噪声 (时序1D卷积)
        本质一样，只是把 2D 卷积换成了 1D 卷积处理时序
    """

    def __init__(self, action_dim: int = 7, obs_dim: int = 128, horizon: int = 16, hidden: int = 256):
        super().__init__()
        self.horizon = horizon

        # 观测编码
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # 时间步编码
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )

        # 1D UNet-style for action sequence
        # 动作序列 (horizon, action_dim) → 当成 1D "图像" 用 Conv1d
        self.embed = nn.Linear(action_dim, hidden)

        self.enc1 = nn.Conv1d(hidden, hidden, 5, padding=2)    # 5=时序感受野
        self.enc2 = nn.Conv1d(hidden, hidden*2, 5, padding=2, stride=2)  # 下采样

        self.bottleneck = nn.Conv1d(hidden*2 + hidden, hidden*2, 5, padding=2)  # enc2通道 + cond通道
        #                            ↑ action特征 + obs特征拼起来

        self.dec1 = nn.ConvTranspose1d(hidden*2, hidden, 5, padding=2, stride=2, output_padding=1)
        self.dec2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.output = nn.Linear(hidden, action_dim)

    def forward(self, a_noisy: torch.Tensor, t: torch.Tensor, obs: torch.Tensor):
        """
        a_noisy: (B, horizon, action_dim)  — 噪声动作序列
        t:       (B,)                      — 去噪步骤
        obs:     (B, obs_dim)              — 当前观测

        返回:  (B, horizon, action_dim) — 预测的噪声
        """
        B, H, A = a_noisy.shape

        # 1. 编码动作
        a = self.embed(a_noisy).transpose(1, 2)          # (B, hidden, H)

        # 2. 编码条件
        t_emb = self.time_mlp(t.float().unsqueeze(-1))   # (B, hidden)
        obs_emb = self.obs_encoder(obs)                   # (B, hidden)
        cond = t_emb + obs_emb                            # 两个条件加起来

        # 3. Encoder (1D conv)
        e1 = self.enc1(a)                                 # (B, hidden, H)
        e2 = self.enc2(e1)                                # (B, hidden*2, H/2)

        # 4. Bottleneck: 注入条件
        cond_expand = cond.unsqueeze(-1).repeat(1, 1, e2.shape[-1])  # (B, H, H/2)
        b = torch.cat([e2, cond_expand], dim=1)           # (B, 2H+2H, H/2)
        b = self.bottleneck(b)

        # 5. Decoder
        d = self.dec1(b)                                  # (B, hidden, H)
        d = self.dec2(d + e1[:, :, :d.shape[-1]])         # skip connection!

        return self.output(d.transpose(1, 2))              # (B, H, A)


# ============================================================================
class DiffusionPolicy(nn.Module):
    """
    完整的 Diffusion Policy: 从观测 → 生成动作序列

    和 DDPM 对比:
        DDPM: image = denoise(noisy_image, t)      → 无条件/条件生成
        DP:   actions = denoise(noisy_actions, t, obs)  → 条件生成
    """

    def __init__(self, action_dim=7, obs_dim=128, horizon=16, T=100):
        super().__init__()
        self.action_dim = action_dim
        self.denoiser = ActionDenoiser(action_dim, obs_dim, horizon)
        self.horizon = horizon
        self.T = T

        # DDPM noise schedule (复用你 03_diffusion 的逻辑)
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        self.register_buffer('alphas_bar', torch.cumprod(alphas, dim=0))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('sqrt_recip_alphas', 1.0 / alphas.sqrt())

    def forward(self, a_clean, obs):
        """训练: 加噪→预测噪声→loss"""
        B = a_clean.shape[0]
        t = torch.randint(0, self.T, (B,), device=a_clean.device)
        noise = torch.randn_like(a_clean)

        a_bar_t = self.alphas_bar[t].view(-1, 1, 1)
        a_noisy = a_bar_t.sqrt() * a_clean + (1 - a_bar_t).sqrt() * noise

        noise_pred = self.denoiser(a_noisy, t, obs)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def act(self, obs, n_steps=50):
        """
        推理: 从纯噪声去噪得到动作序列，返回前几步执行。
        工业实践: 生成 H=16 步动作，只执行前 4 步，然后 re-plan。
        """
        B = obs.shape[0]
        a = torch.randn(B, self.horizon, self.action_dim, device=obs.device)

        step_times = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=obs.device)
        for i in range(len(step_times) - 1):
            t = step_times[i]; t_next = step_times[i + 1]
            eps = self.denoiser(a, t.expand(B), obs)

            at = self.alphas[t]; at_bar = self.alphas_bar[t]; at_bar_next = self.alphas_bar[t_next]
            # DDIM step (确定性)
            pred_a0 = (a - (1 - at_bar).sqrt() * eps) / at_bar.sqrt()
            dir_xt = (1 - at_bar_next).sqrt() * eps
            a = at_bar_next.sqrt() * pred_a0 + dir_xt

        return a


# ============================================================================
if __name__ == "__main__":
    device = torch.device("cpu")

    horizon = 16       # 生成未来 16 步动作
    action_dim = 7     # 7DoF
    obs_dim = 128      # 观测特征

    dp = DiffusionPolicy(action_dim, obs_dim, horizon)
    print(f"Diffusion Policy: 去噪 {horizon}×{action_dim} 动作序列")

    # 模拟训练
    obs = torch.randn(4, obs_dim)
    actions = torch.randn(4, horizon, action_dim)
    loss = dp(actions, obs)
    print(f"训练 loss: {loss.item():.4f}")

    # 推理
    with torch.no_grad():
        actions_out = dp.act(obs[:1], n_steps=20)
    print(f"生成动作: {actions_out.shape}  ← {horizon}步×{action_dim}DoF")
    print(f"前3步: {actions_out[0, :3].detach()}")

    print(f"\n✅ Diffusion Policy OK")
    print(f"  扩散去噪的不只是图像，也可以是动作/轨迹/任何连续信号")
