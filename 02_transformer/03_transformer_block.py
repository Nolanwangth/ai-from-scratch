# 运行: conda activate diffusion && python 02_transformer/03_transformer_block.py
"""
Transformer Block — 把组件拼成积木
=====================================
一个 Transformer Block = 两层:

    x ──→ [Multi-Head Attention] ──→ (+) ──→ [LayerNorm] ──→ [FFN] ──→ (+) ──→ [LayerNorm] ──→ output
              ↑                      ↑                          ↑
              └── residual ──────────┘                          └── residual ──────────┘

关键设计:
    1. Residual Connection (残差连接): x + f(x) — 让梯度直接流回去，防止梯度消失
    2. LayerNorm BEFORE each sublayer (Pre-LN, 现代做法)
    3. FFN = 两个线性层中间夹一个激活函数: Linear → GELU → Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
FlashMultiHeadAttention = importlib.import_module("01_multi_head_attention").FlashMultiHeadAttention


# ============================================================================
# 1. Feed-Forward Network (FFN) — 位置独立的全连接层
# ============================================================================
class FeedForward(nn.Module):
    """
    FFN: 对每个 token 独立做同样的变换
    结构: Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)

    为什么中间维度要放大 4 倍?
        给 Attention 的输出一个"思考空间"。
        Attention 负责 "token 之间的信息交互"
        FFN 负责 "每个 token 内部的信息处理"
        扩大 4 倍给了足够的容量来做非线性变换。

    为什么用 GELU 而不是 ReLU?
        GELU = Gaussian Error Linear Unit
        - 比 ReLU 更平滑 (可导，梯度不突然断掉)
        - GPT/BERT 都用 GELU
        - 近似公式: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
    """

    def __init__(self, d_model: int = 512, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = d_model * expansion_factor       # 通常是 4*d_model

        self.fc1 = nn.Linear(d_model, inner_dim)     # 放大
        self.fc2 = nn.Linear(inner_dim, d_model)     # 缩回
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        x = self.fc1(x)                               # 放大到 4*d_model
        x = F.gelu(x)                                 # ⚠️ 注意: 不是 relu! Transformer 用 gelu
        x = self.dropout(x)
        x = self.fc2(x)                               # 缩回 d_model
        return x


# ============================================================================
# 2. Transformer Block — 一个完整的 Transformer 层
# ============================================================================
class TransformerBlock(nn.Module):
    """
    标准 Pre-LN Transformer Block:
        x → LN → Attention → (+) → LN → FFN → (+)

    Pre-LN vs Post-LN:
        - Pre-LN (LayerNorm 在前): 更稳定，不需要 warmup，现代主流
        - Post-LN (LayerNorm 在后): 论文原版，但训练不稳定需要 warmup

    为什么需要 Residual Connection?
        假设没有残差: x → Attention(x) → FFN(Attention(x))
        梯度要从最后一层传到第一层 → 每过一层乘一次导数 → 可能梯度消失/爆炸
        有残差: x → x + Attention(x) → x + FFN(...)
        → 梯度有一根 "高速公路" 直接流回去 → 可以训练很深的网络
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Self-Attention 层
        self.attention = FlashMultiHeadAttention(d_model, n_heads, dropout)

        # Feed-Forward 层
        self.ffn = FeedForward(d_model, expansion_factor=4, dropout=dropout)

        # LayerNorm: 对每个 token 的 d_model 维度做归一化
        # nn.LayerNorm(normalized_shape): 对最后一维做归一化
        # 输入 (B, T, d_model) → 对 d_model 维做归一化 → 输出 (B, T, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pre-LN 风格:
            x = x + Attention(LN(x))    ← 先归一化，再 attention，再加回去
            x = x + FFN(LN(x))          ← 先归一化，再 FFN，再加回去

        Args:
            x:    (B, T, d_model)
            mask: (B, 1, T, T) or None — causal mask 或 padding mask

        Returns:
            (B, T, d_model)
        """
        # Sublayer 1: Self-Attention + Residual
        attn_out = self.attention(self.ln1(x), mask=mask)  # 先 LN 再 attention
        x = x + self.dropout(attn_out)                      # 残差连接

        # Sublayer 2: Feed-Forward + Residual
        ffn_out = self.ffn(self.ln2(x))                     # 先 LN 再 FFN
        x = x + self.dropout(ffn_out)                       # 残差连接

        return x


# ============================================================================
# 3. 验证: 测试 Block 在 GPU 上的表现
# ============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.mps_utils import get_device, print_device_info
    create_causal_mask = importlib.import_module("01_multi_head_attention").create_causal_mask

    print("=" * 60)
    print("Transformer Block 演示")
    print("=" * 60)

    device = get_device()
    print_device_info()

    # 超参数 (小型 Transformer 的经典配置)
    B, T, D = 2, 128, 256           # batch=2, seq=128, d_model=256
    n_heads = 8
    n_layers = 6

    # 创建输入
    x = torch.randn(B, T, D, device=device)
    mask = create_causal_mask(T, device=device)

    # --- 单层 Block ---
    print(f"\n1. 单层 TransformerBlock:")
    block = TransformerBlock(d_model=D, n_heads=n_heads).to(device)
    out = block(x, mask=mask)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape} (和输入一样)")
    print(f"   单层参数量: {sum(p.numel() for p in block.parameters()):,}")
    # 参数量计算: 4 * d_model^2 (attention QKV+out) + 2 * 4 * d_model^2 (FFN) + 2*d_model (2个LayerNorm)
    # ≈ 4 * 256^2 + 8 * 256^2 + 512 = 786,432
    print(f"   ✅ 单层 OK")

    # --- 多层堆叠 ---
    print(f"\n2. {n_layers} 层 TransformerBlocks 堆叠:")
    # nn.ModuleList: 像 list 一样存多个 module，但 PyTorch 能追踪它们的参数
    blocks = nn.ModuleList([
        TransformerBlock(d_model=D, n_heads=n_heads) for _ in range(n_layers)
    ]).to(device)
    #  为什么要用 ModuleList 而不是普通 list?
    #  普通 list 里的 module 不会被 parameters() 遍历到 → optimizer 找不到它们!

    out = x
    for i, block in enumerate(blocks):
        out = block(out, mask=mask)
        # 验证: 每层输出不应该发散
        assert not torch.isnan(out).any(), f"第 {i} 层输出了 NaN!"

    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   {n_layers} 层总参数量: {sum(p.numel() for p in blocks.parameters()):,}")
    print(f"   ✅ {n_layers} 层堆叠 OK, 无 NaN")

    # --- 架构示意图 ---
    print(f"\n3. Transformer Block 内部结构:")
    print(f"""
    ┌──────────────────────────────────┐
    │  INPUT x: (B={B}, T={T}, D={D})  │
    └──────────────┬───────────────────┘
                   │
        ┌──────────▼──────────┐
        │   LayerNorm #1      │  ← 归一化
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Multi-Head Attn    │  ← 每个 token 看所有 token
        │  ({n_heads} heads, d_head={D//n_heads})  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  (+) Residual       │  ← x = x + Attention(LN(x))
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   LayerNorm #2      │  ← 归一化
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  FFN (256→1024→256) │  ← 对每个 token 独立做变换
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  (+) Residual       │  ← x = x + FFN(LN(x))
        └──────────┬──────────┘
                   │
    ┌──────────────▼───────────────────┐
    │  OUTPUT: (B={B}, T={T}, D={D})   │
    └──────────────────────────────────┘
    """)

    print("=" * 60)
    print("✅ Transformer Block 完成！")
    print("核心: Attention + FFN + LayerNorm + Residual = 一个 Block")
    print("多层堆叠 → GPT!")
    print("下一步: 运行 04_mini_gpt.py 组装完整模型")
    print("=" * 60)
