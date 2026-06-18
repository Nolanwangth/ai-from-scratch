# 运行: conda activate diffusion && python 05_robotics/01_diffusion_policy.py
"""
Diffusion Policy — DiT + Flow Matching 两种去噪方式
====================================================
把原来的 1D Conv U-Net 换成 DiT (Diffusion Transformer)。
同时提供两种训练/推理选项:

Option 1 — DDPM (离散扩散, 你从 03_ddpm 就熟悉的):
    a_noisy = √ᾱ·a + √(1-ᾱ)·ε  →  predict ε
    推理: 50-100 步 DDIM 去噪

Option 2 — Flow Matching (连续流, 你从 07_flow_matching 知道的):
    a_t = (1-t)·a + t·ε  →  predict v = ε - a
    推理: 10-50 步 ODE Euler 求解

区别:
    DDPM:   需要噪声 schedule, 更多步数
    Flow:   更简单 (无 schedule), 步数更少

架构变化:
    原来: ActionDenoiser = 1D Conv Encoder → Bottleneck → 1D Conv Decoder
    现在: DiTActionDenoiser = Self-Attention → Cross-Attention → FFN (×N层)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# DiT 基础组件
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码 — 把标量 t 映射成高维向量"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TransformerBlock(nn.Module):
    """
    DiT Block: Self-Attention → Cross-Attention → FFN

    和标准 Transformer Decoder 一样的结构:
        1. 动作序列自己做 Self-Attention (捕捉时序关系)
        2. 每个动作步通过 Cross-Attention 看条件 (t + 观测)
        3. FFN 做非线性变换
    """

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        # Pre-norm: 在每个子层之前做 LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, cond):
        """
        x:    (B, horizon, dim)  — 动作 token 序列
        cond: (B, 1, dim)        — 条件 (t + obs 编码)
        """
        # Self-attention: 每个动作步看其他动作步
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Cross-attention: 每个动作步看条件
        x = x + self.cross_attn(self.norm2(x), cond, cond)[0]
        # FFN
        x = x + self.ffn(x)
        return x


# ============================================================================
# DiT 去噪器 (替代原来的 Conv1D UNet)
# ============================================================================

class DiTActionDenoiser(nn.Module):
    """
    基于 DiT 的动作去噪器。

    和原来 ActionDenoiser 的对比 (Conv1D vs Transformer):
        原来: 1D Conv 编码 → 下采样 → Bottleneck + 条件注入 → 上采样 → Skip 连接
             好: 卷积的局部感受野适合短时序
             坏: 不好处理长序列, 不好 scale

        DiT:  Self-Attention 全局建模时序关系
             Cross-Attention 注入条件
             好: 可处理任意长度序列, 可堆更多层
             坏: 需要更多数据, 训练更慢

    输入:
        a_noisy: (B, horizon, action_dim)  — 噪声动作序列
        t:       (B,)                      — 时间步 (DDPM传int, Flow传float)
        obs:     (B, obs_dim)              — 观测
    """

    def __init__(self, action_dim=7, obs_dim=128, horizon=16, hidden=256, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon

        # 1. 动作嵌入: 把每个动作步 (action_dim,) → (hidden,)
        self.action_embed = nn.Linear(action_dim, hidden)

        # 2. 位置编码: 让 Transformer 知道每个动作步的顺序
        #    用可学习的 position embedding (和 BERT 一样)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)

        # 3. 时间编码: 把标量 t 映射到高维
        #    Sinusoidal 比简单 MLP 更适合表示连续的时间步
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # 4. 观测编码: 把观测特征映射到和动作一样的维度
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # 5. Transformer 层
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, n_heads, dropout) for _ in range(n_layers)
        ])

        # 6. 输出投影: (hidden,) → (action_dim,)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, action_dim)

    def forward(self, a_noisy, t, obs):
        B, H, A = a_noisy.shape

        # 动作嵌入 + 位置编码
        a = self.action_embed(a_noisy)                           # (B, H, hidden)
        a = a + self.pos_embed[:, :H, :]                         # +位置编码

        # 条件编码 → (B, 1, hidden) 作为 cross-attention 的 key/value
        t_emb = self.time_embed(t).unsqueeze(1)                  # (B, 1, hidden)
        obs_emb = self.obs_encoder(obs).unsqueeze(1)             # (B, 1, hidden)
        cond = t_emb + obs_emb                                   # 直接把条件加在一起

        # 过 Transformer
        for block in self.blocks:
            a = block(a, cond)

        return self.output(self.norm(a))                         # (B, H, action_dim)


# ============================================================================
# Option 1: DDPM 调度 + DiT 去噪器
# ============================================================================

class DiffusionPolicy(nn.Module):
    """
    DDPM + DiT 的 Diffusion Policy。

    和原来相比, 只有去噪器从 Conv1D UNet 换成了 DiT Transformer。
    训练和推理的 DDPM 逻辑不变:
        训练: a_noisy = √ᾱ·a + √(1-ᾱ)·ε,  predict ε → MSE loss
        推理: DDIM 从纯噪声逐步去噪
    """

    def __init__(self, action_dim=7, obs_dim=128, horizon=16, T=100, hidden=256):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.T = T
        self.denoiser = DiTActionDenoiser(action_dim, obs_dim, horizon, hidden)

        # DDPM noise schedule (和 03_diffusion/03_ddpm_trainer.py 一样)
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        self.register_buffer('alphas_bar', torch.cumprod(alphas, dim=0))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)

    def forward(self, a_clean, obs):
        """训练: 加噪 → DiT 预测噪声 → MSE loss"""
        B = a_clean.shape[0]
        t = torch.randint(0, self.T, (B,), device=a_clean.device)
        noise = torch.randn_like(a_clean)

        a_bar_t = self.alphas_bar[t].view(-1, 1, 1)
        a_noisy = a_bar_t.sqrt() * a_clean + (1 - a_bar_t).sqrt() * noise

        noise_pred = self.denoiser(a_noisy, t.float(), obs)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def act(self, obs, n_steps=50):
        """推理: DDIM 去噪 (确定性)"""
        B = obs.shape[0]
        a = torch.randn(B, self.horizon, self.action_dim, device=obs.device)

        step_times = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=obs.device)
        for i in range(len(step_times) - 1):
            t = step_times[i]
            t_next = step_times[i + 1]
            eps = self.denoiser(a, t.float().expand(B), obs)

            at = self.alphas[t]
            at_bar = self.alphas_bar[t]
            at_bar_next = self.alphas_bar[t_next]

            pred_a0 = (a - (1 - at_bar).sqrt() * eps) / at_bar.sqrt()
            dir_xt = (1 - at_bar_next).sqrt() * eps
            a = at_bar_next.sqrt() * pred_a0 + dir_xt

        return a


# ============================================================================
# Option 2: Flow Matching 调度 + DiT 去噪器
# ============================================================================

class FlowMatchingPolicy(nn.Module):
    """
    Flow Matching + DiT 的 Diffusion Policy。

    和 DDPM 对比:
        DDPM: 需要 ᾱ_t, β_t, α_t 这些 schedule
              推理 100 步, 每次都要算复杂的去噪公式

        Flow Matching: 不需要任何 schedule!
              训练: a_t = (1-t)·a + t·ε  (线性插值)
                   预测 v = ε - a  (速度场, 指向数据的方向)
              推理: Euler ODE 从 t=1→0, 10-50 步
                      a = a - v·dt  (沿 ODE 向后推)

    为什么叫 "Flow":
        数据分布的"流"从噪声流向真实数据。
        训练学的是"每个点的运动方向" (速度场),
        推理就是"沿着速度场往前走"。
    """

    def __init__(self, action_dim=7, obs_dim=128, horizon=16, hidden=256):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.denoiser = DiTActionDenoiser(action_dim, obs_dim, horizon, hidden)

    def forward(self, a_clean, obs):
        """
        训练: 线性插值 → 预测速度 → MSE loss

        Flow Matching 比 DDPM 简单的核心原因:
            DDPM: α_t = 1 - β_t,  ᾱ_t = ∏α_i,  a_t = √ᾱ_t·a + √(1-ᾱ_t)·ε
                  噪声加多少不是线性的, 每一步的 SNR 都不同

            Flow: a_t = (1-t)·a + t·ε,  t ∈ [0,1]
                  噪声匀速加入, 路径是直线
        """
        B = a_clean.shape[0]

        # 采样 t ~ Uniform(0, 1)
        t = torch.rand(B, device=a_clean.device)

        # 从数据分布采样噪声
        noise = torch.randn_like(a_clean)

        # 线性插值: a_t = (1-t)·a_0 + t·ε
        # t=0: a_t = a_0 (干净数据)
        # t=1: a_t = ε  (纯噪声)
        a_t = (1 - t.view(-1, 1, 1)) * a_clean + t.view(-1, 1, 1) * noise

        # 预测速度场 v(a_t, t) = d(a_t)/dt = ε - a_0
        v_pred = self.denoiser(a_t, t, obs)
        v_target = noise - a_clean

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def act(self, obs, n_steps=10):
        """
        推理: Euler ODE 求解器

        从纯噪声 a_1 开始, 沿着 ODE d(a_t)/dt = v(a_t, t) 向后推:
            a_{t-dt} = a_t - v(a_t, t) · dt

        理解为什么是减号:
            速度场 v 指向"从噪声到数据"的方向。
            我们要从噪声反推到数据, 所以逆着速度场走。
            也可以想象成: v 告诉你"去噪声的方向", 你要反向走。

        为什么 n_steps=10 就够了 (DDPM 要 50-100):
            Flow Matching 的路径是直线, Euler 只要很少步就能近似。
            DDPM 的路径是弯的 (因为有噪声 schedule), 需要更多步。
        """
        B = obs.shape[0]
        a = torch.randn(B, self.horizon, self.action_dim, device=obs.device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            # t 从 1 到 0 (从噪声到数据)
            t = torch.ones(B, device=obs.device) * (1.0 - i * dt)
            v = self.denoiser(a, t, obs)
            a = a - v * dt  # Euler 步: 逆着速度场走

        return a


# ============================================================================
# Demo — 展示两种选项
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cpu")

    horizon = 16        # 生成未来 16 步动作
    action_dim = 7      # 7DoF
    obs_dim = 128       # 观测特征

    print("=" * 60)
    print("Option 1: DDPM + DiT")
    print("=" * 60)
    dp = DiffusionPolicy(action_dim, obs_dim, horizon)
    print(f"Diffusion Policy: DiT denoiser + DDPM")

    obs = torch.randn(4, obs_dim)
    actions = torch.randn(4, horizon, action_dim)
    loss = dp(actions, obs)
    print(f"训练 loss: {loss.item():.4f}")

    with torch.no_grad():
        actions_out = dp.act(obs[:1], n_steps=20)
    print(f"生成动作: {actions_out.shape}")
    print(f"前3步: {actions_out[0, :3].detach()}")

    print()

    print("=" * 60)
    print("Option 2: Flow Matching + DiT")
    print("=" * 60)
    fm = FlowMatchingPolicy(action_dim, obs_dim, horizon)
    print(f"Flow Matching Policy: DiT denoiser + Flow")

    loss = fm(actions, obs)
    print(f"训练 loss: {loss.item():.4f}")

    with torch.no_grad():
        actions_out = fm.act(obs[:1], n_steps=10)
    print(f"生成动作 (10步ODE): {actions_out.shape}")
    print(f"前3步: {actions_out[0, :3].detach()}")

    with torch.no_grad():
        actions_out = fm.act(obs[:1], n_steps=50)
    print(f"生成动作 (50步ODE): {actions_out.shape}")
    print(f"前3步: {actions_out[0, :3].detach()}")

    print()

    # =========================================================================
    # 对比总结
    # =========================================================================
    print("=" * 60)
    print("两种方式对比")
    print("=" * 60)
    print(f"""
    架构:  两者都用 DiTActionDenoiser (Transformer)

    ┌──────────────┬─────────────────────┬──────────────────────┐
    │              │ DDPM                │ Flow Matching        │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 正向过程     │ √ᾱ·a + √(1-ᾱ)·ε    │ (1-t)·a + t·ε        │
    │              │ (曲线路径, 复杂)     │ (线性路径, 简单)      │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 预测目标     │ ε (噪声)             │ v = ε - a (速度场)    │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 调度参数     │ β_t, α_t, ᾱ_t       │ 无! 只需要 t ∈ [0,1] │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 推理步骤     │ 50-100 (DDIM)        │ 10-50 (Euler ODE)    │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 确定性       │ 否 (可加DDIM变确定)  │ 是 (纯ODE, 确定)      │
    └──────────────┴─────────────────────┴──────────────────────┘

    原本的 1D Conv U-Net → DiT Transformer:
        Conv:     局部感受野, 先下采样再上采样
        Transformer: 全局 Self-Attention, 所有动作步同时看
    """)


# ============================================================================
# 附: 原来的 Conv1D ActionDenoiser (已废弃, 留作参考)
# ============================================================================
"""
class ActionDenoiser(nn.Module):
    原本的 1D Conv U-Net 去噪器, 已被 DiTActionDenoiser 替代。

    如果你想回到 Conv1D 版本:
        1. 取消下面这段注释
        2. 在 DiffusionPolicy 里把 denoiser 改回 ActionDenoiser
        3. 删除 DiTActionDenoiser

    结构:
        Linear 嵌入 → Conv1D Encoder(×2, 含下采样)
        → Bottleneck + 条件注入(通过expand然后cat)
        → ConvTranspose1D Decoder(×2, 上采样) + Skip连接
        → Linear 输出

    def __init__(self, action_dim=7, obs_dim=128, horizon=16, hidden=256):
        super().__init__()
        self.horizon = horizon
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.embed = nn.Linear(action_dim, hidden)
        self.enc1 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.enc2 = nn.Conv1d(hidden, hidden*2, 5, padding=2, stride=2)
        self.bottleneck = nn.Conv1d(hidden*2 + hidden, hidden*2, 5, padding=2)
        self.dec1 = nn.ConvTranspose1d(hidden*2, hidden, 5, padding=2, stride=2, output_padding=1)
        self.dec2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.output = nn.Linear(hidden, action_dim)

    def forward(self, a_noisy, t, obs):
        B, H, A = a_noisy.shape
        a = self.embed(a_noisy).transpose(1, 2)
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        obs_emb = self.obs_encoder(obs)
        cond = t_emb + obs_emb
        e1 = self.enc1(a)
        e2 = self.enc2(e1)
        cond_expand = cond.unsqueeze(-1).repeat(1, 1, e2.shape[-1])
        b = torch.cat([e2, cond_expand], dim=1)
        b = self.bottleneck(b)
        d = self.dec1(b)
        d = self.dec2(d + e1[:, :, :d.shape[-1]])
        return self.output(d.transpose(1, 2))
"""
