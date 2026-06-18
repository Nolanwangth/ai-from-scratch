# 运行: conda activate diffusion && python 05_robotics/02_world_model.py
"""
World Model — 生成式世界模型 (视频预测 + 文本条件)
==================================================
现代世界模型 = 视频生成模型 = 看了过去 + 读了指令 → 预测未来

    past_frames_{t-3, t-2, t-1} + text_prompt → [World Model] → next_frame_t

和原来 Dreamer/RSSM 风格的区别:

    RL 风格 (Dreamer, 原来的代码):
        state_t + action_t → [Dynamics MLP] → next_state_t
        ↑ "做这个动作, 世界会怎样?"
        需要显式动作输入, 只能处理低维状态

    生成式风格 (Cosmos, 现在的代码):
        past_frames + text_prompt → [Transformer] → next_frame
        ↑ "根据指令和看到的变化, 下一帧应该是什么?"
        不需要动作, 直接从观测序列中学习世界规律

    Cosmos (NVIDIA): 视频作为世界模型
        - 输入: 视频帧序列 + 文本描述
        - 输出: 预测的未来视频帧
        - 世界模型 = 视频生成模型

架构:
    过去帧 (T 帧) → FrameEncoder → frame tokens
    文本描述      → PromptEncoder → prompt token
    [frame_tokens, prompt_token, PREDICT_token]
    → Transformer Encoder
    → PREDICT_token 的输出 → OutputDecoder → 预测下一帧

参考:
    NVIDIA Cosmos: https://developer.nvidia.com/cosmos
    WAN + Qwen: 视频生成模型 + 大语言模型作为世界模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
class SinusoidalPosEmb(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ============================================================================
class WorldModel(nn.Module):
    """
    生成式世界模型。

    和视频生成模型的对应关系:
        视频生成:   噪声 + 文本 → 逐帧生成
        世界模型:   过去帧 + 文本 → 下一帧

    核心思路:
        1. 把过去 N 帧观测编码成 N 个 token
        2. 把文本 prompt 编码成 1 个 token
        3. 加一个 [PREDICT] token 表示"要预测的位置"
        4. Transformer 处理所有 token
        5. [PREDICT] token 的输出就是预测的下一帧

    和 Diffusion Policy 的配合:
        DP:  观测 → 生成动作序列
        WM:  过去帧 + 描述 → 预测下一帧
        合起来: 感知→决策→预测闭环
    """

    def __init__(self, obs_dim: int = 64, prompt_dim: int = 512, latent_dim: int = 128,
                 n_past: int = 4, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.n_past = n_past
        self.latent_dim = latent_dim

        # ── Frame Encoder: 每帧观测 → 1 个 latent token ──
        # 如果是图像, 这里可以用 CNN/ViT 编码器
        # 这里是低维观测 (如机器人关节角度 + 物体位置)
        self.frame_encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # ── Prompt Encoder: 文本描述 → 1 个 latent token ──
        # 真实场景: 用 LLM 的 embedding (如 Qwen, Llama)
        # 这里: 模拟 prompt_embedding, 假设 prompt 已经被编码了
        self.prompt_encoder = nn.Sequential(
            nn.Linear(prompt_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # ── 位置编码: 让 Transformer 知道每个 token 的顺序 ──
        # 布局: [past_frame_0, past_frame_1, ..., prompt, PREDICT]
        self.pos_embed = nn.Parameter(torch.randn(1, n_past + 2, latent_dim) * 0.02)

        # ── Transformer Encoder ──
        # 用 PyTorch 自带的 TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (和 DiT 一样)
            enable_nested_tensor=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Output Decoder: latent → 预测的下一帧观测 ──
        self.output_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, obs_dim),
        )

    def forward(self, past_obs: torch.Tensor, prompt: torch.Tensor):
        """
        训练: 用过去 N 帧 + prompt → 预测下一帧

        参数:
            past_obs: (B, n_past, obs_dim)  — 过去 N 帧观测
            prompt:   (B, prompt_dim)       — 文本描述编码

        返回:
            next_obs_pred: (B, obs_dim)     — 预测的下一帧
        """
        B = past_obs.shape[0]

        # 1. 编码过去 N 帧 → N 个 token
        frame_tokens = self.frame_encoder(past_obs)                 # (B, N, latent)

        # 2. 编码 prompt → 1 个 token
        prompt_token = self.prompt_encoder(prompt).unsqueeze(1)     # (B, 1, latent)

        # 3. 创建 [PREDICT] token (初始化为全 0, 由 Transformer 填充信息)
        predict_token = torch.zeros(B, 1, self.latent_dim, device=past_obs.device)

        # 4. 拼接所有 token
        tokens = torch.cat([frame_tokens, prompt_token, predict_token], dim=1)

        # 5. 加位置编码
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # 6. Transformer 编码
        out = self.transformer(tokens)                               # (B, N+2, latent)

        # 7. [PREDICT] token 的输出 = 预测的下一帧
        return self.output_decoder(out[:, -1, :])                     # (B, obs_dim)

    @torch.no_grad()
    def rollout(self, init_past: torch.Tensor, prompt: torch.Tensor, n_future: int = 10):
        """
        自回归 rollout: 从初始序列开始, 不断用预测的下一帧当输入。

        这是世界模型的核心能力: "如果当前趋势继续, 未来会怎样?"

        参数:
            init_past: (B, n_past, obs_dim) — 初始的连续 N 帧
            prompt:    (B, prompt_dim)      — 文本描述
            n_future:  预测多少步

        返回:
            future: (B, n_future, obs_dim) — 预测的未来观测序列
        """
        B = init_past.shape[0]
        past_queue = list(init_past.unbind(dim=1))                   # [ (B,obs) × N ]

        future = []
        for _ in range(n_future):
            # 取最近的 N 帧
            past = torch.stack(past_queue[-self.n_past:], dim=1)     # (B, N, obs)
            next_pred = self(past, prompt)                           # (B, obs)
            future.append(next_pred)
            past_queue.append(next_pred)                             # 把预测结果加入队列

        return torch.stack(future, dim=1)                            # (B, T, obs)


# ============================================================================
if __name__ == "__main__":
    device = get_device()
    print(f"设备: {device}")

    obs_dim = 64            # 观测维度 (模拟低维状态)
    prompt_dim = 512        # 文本描述编码维度 (如 Qwen 的 embedding 维度)
    latent_dim = 128        # Transformer 隐层维度
    n_past = 4              # 用过去 4 帧预测下一帧

    wm = WorldModel(obs_dim, prompt_dim, latent_dim, n_past).to(device)
    print(f"World Model: {n_past}帧过去 + 文本 → 预测下一帧 ({obs_dim}d)")

    # ────────────────
    # 训练
    # ────────────────
    B = 8
    past_obs = torch.randn(B, n_past, obs_dim, device=device)
    prompt = torch.randn(B, prompt_dim, device=device)        # 模拟 LLM embedding
    next_obs = torch.randn(B, obs_dim, device=device)

    pred = wm(past_obs, prompt)
    loss = F.mse_loss(pred, next_obs)
    print(f"\n单步训练:")
    print(f"  past_obs ({n_past}帧): {past_obs.shape}")
    print(f"  prompt:               {prompt.shape}     ← 文本描述编码")
    print(f"  pred next_obs:        {pred.shape}")
    print(f"  target next_obs:      {next_obs.shape}")
    print(f"  Loss:                 {loss.item():.4f}")

    # ────────────────
    # 自回归 rollout
    # ────────────────
    init_past = torch.randn(1, n_past, obs_dim, device=device)
    n_future = 20
    future = wm.rollout(init_past, prompt[:1], n_future)
    print(f"\n自回归 Rollout (预测未来 {n_future} 帧):")
    print(f"  初始: {n_past}帧 过去观测")
    print(f"  预测: {future.shape} ← 从过去趋势推断的未来")

    # ────────────────
    # 世界模型 vs RL 风格对比
    # ────────────────
    print(f"\n{'='*60}")
    print(f"生成式世界模型 vs RL 世界模型")
    print(f"{'='*60}")
    print(f"""
    RL 风格 (经典 Dreamer/RSSM):
        input:  state_t + action_t
        output: next_state_t
        → 需要显式动作标签, 动作空间固定
        → 适合: 格子世界, 机器人低维控制

    生成式风格 (Cosmos / 本文件):
        input:  past_frames_{{t-3,t-2,t-1}} + text_prompt
        output: next_frame_{{t}}
        → 不需要动作, 从时序中学习世界规律
        → 文本 prompt 指导"世界应该往哪个方向变"
        → 适合: 视频预测, 自动驾驶, 机器人视觉推理

    NVIDIA Cosmos:
        "World Foundation Model" = 视频生成模型
        输入: N帧视频 + 文本描述
        输出: 未来 M 帧视频
        世界模型 = "如果你继续这样, 世界会变成..."

    WAN + Qwen:
        WAN 负责: 视频帧的生成/变换
        Qwen 负责: 理解文本描述 → 指导世界走向
        → 文本作为"世界应该往哪个方向变化"的条件
    """)

    print(f"\n✅ 生成式 World Model OK")
    print(f"  核心: 不需要动作, 从观测序列中自己学会世界的规律")


# ============================================================================
# 附: 原来的 RL 风格 World Model (已废弃, 留作参考)
# ============================================================================
"""
class RLWorldModel(nn.Module):
    原本的 RL 风格世界模型 (Dreamer/RSSM), 已被生成式 WorldModel 替代。

    如果你想回到 state+action → next state 版本:
        1. 取消下面这段注释
        2. 重命名当前 WorldModel → GenerativeWorldModel
        3. 恢复 RLWorldModel 的名字

    def __init__(self, obs_dim=64, act_dim=3, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim),
        )
        self.dynamics = nn.Sequential(
            # z_t + a_t → z_next (需要动作标签!)
            nn.Linear(latent_dim + act_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, obs_dim),
        )

    def forward(self, obs, action):
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        return self.decoder(z_next)
"""
