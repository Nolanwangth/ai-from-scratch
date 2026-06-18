"""
Cross-Attention — VLA 的多模态桥梁
=====================================
Self-Attention: Q,K,V 来自同一个序列    → token 看 token
Cross-Attention: Q 来自 A, K/V 来自 B   → 模态 A 看模态 B

VLA 里这就是 "语言指令 → 视觉特征" 的融合:
    Q = 视觉特征  (当前看到的)
    K,V = 语言指令 (任务描述)
    → 视觉 "查询" 语言: "我看到的东西里，哪些和任务相关？"

    例: 语言 "拿起红色杯子" → 视觉特征中 "红色杯子区域" 的 attention 权重变高

同时也有 视觉→动作 的 Cross-Attention:
    Q = action queries (动作槽位)
    K,V = visual features (视觉特征)
    → "需要做哪些动作来操作看到的物体？"

你 02_transformer 里有 Self-Attention，这个就是换一下 Q 的来源。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
class CrossAttention(nn.Module):
    """
    Q 来自 query_seq，K/V 来自 context_seq。

    例: query_seq=视觉patch×64, context_seq=语言token×10
        → 输出(64,d) = 每个视觉 patch 融合了所有语言 token 的信息
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q 来自 query 序列
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        # K, V 来自 context 序列
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query:   (B, T_q, d)  ← 视觉 patch
        context: (B, T_c, d)  ← 语言 token
        返回:    (B, T_q, d)  ← 视觉=语言融合后的表示
        """
        B, T_q, _ = query.shape
        _, T_c, _ = context.shape

        Q = self.W_Q(query).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(context).view(B, T_c, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(context).view(B, T_c, self.n_heads, self.d_head).transpose(1, 2)

        # Q·K^T: 每个视觉 patch 对所有语言 token 的 attention
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T_q, self.d_model)
        return self.out_proj(out)


# ============================================================================
class VLAMini(nn.Module):
    """
    极简 VLA (Vision-Language-Action) 模型。演示 3 步融合：
        vision → + language (cross-attn) → action
    """

    def __init__(self, vision_dim=256, lang_dim=256, action_dim=7, n_patches=64):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, 256)      # 对齐维度
        self.lang_proj = nn.Linear(lang_dim, 256)

        # 语言指令 → 视觉特征
        self.cross_attn = CrossAttention(256, n_heads=8)

        # 动作解码: 融合特征 → action head
        self.action_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )
        # action_dim = 7: xyz位置 + rpy姿态 + gripper开合

    def forward(self, vision: torch.Tensor, lang: torch.Tensor):
        """
        vision: (B, 64, 256)   ← 64 个 ViT patch
        lang:   (B, 10, 256)   ← 10 个语言 token
        输出:   (B, 7)          ← 机器人动作
        """
        v = self.vision_proj(vision)
        l = self.lang_proj(lang)

        # 关键: 视觉 query 语言 → 视觉特征被语言 "调制"
        fused = self.cross_attn(v, l)

        # 池化所有 patch → 单个动作向量
        pooled = fused.mean(dim=1)
        return self.action_head(pooled)


# ============================================================================
if __name__ == "__main__":
    device = torch.device("cpu")

    # 模拟 VLA 场景
    B = 2
    vision_patches = torch.randn(B, 64, 256)          # ViT 输出, 64个patch
    lang_tokens = torch.randn(B, 10, 256)             # 指令 "拿起红色杯子"

    # 1. Cross-Attention 单元测试
    ca = CrossAttention(256, n_heads=8)
    fused = ca(vision_patches, lang_tokens)
    print(f"Vision: {vision_patches.shape}")
    print(f"Lang:   {lang_tokens.shape}")
    print(f"Fused:  {fused.shape}  ← 每个视觉patch融合了全部语言信息")

    # 2. 完整 VLA
    vla = VLAMini()
    action = vla(vision_patches, lang_tokens)
    print(f"\nAction: {action.shape}  ← 7DoF 机器人动作")
    print(f"  (xyz={[f'{x:.2f}' for x in action[0,:3].detach().tolist()]}, "
          f"rpy={[f'{x:.2f}' for x in action[0,3:6].detach().tolist()]}, "
          f"gripper={action[0,6].detach():.2f})")

    print(f"\n✅ Cross-Attention OK — VLA 的多模态核心")
