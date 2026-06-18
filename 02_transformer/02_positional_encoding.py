# 运行: conda activate diffusion && python 02_transformer/02_positional_encoding.py
"""
Positional Encoding — 告诉模型词的位置
=========================================
Transformer 的 Self-Attention 是**排列不变**的:
    "猫 坐 在 垫子 上" 和 "上 垫子 在 坐 猫"
    → 如果不加位置信息，Attention 看不出区别!

Positional Encoding 解决了这个问题:
    给每个位置的 token 加一个独特的"位置签名"
    → 模型就知道 "第一个词" 和 "最后一个词" 的位置不同

两种方案:
    1. Sinusoidal (论文原版): 用 sin/cos 函数生成，不需要训练
    2. Learned (GPT 用的): 可训练的 Embedding，更灵活
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


# ============================================================================
# 1. Sinusoidal Positional Encoding — Transformer 论文原版
# ============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    用 sin 和 cos 生成位置编码，不需要训练，数学上生成。

    公式 (来自 "Attention Is All You Need"):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    直观理解:
        - 每个位置 pos 有一个 d_model 维的向量
        - 偶数维度用 sin，奇数维度用 cos
        - 不同频率的波 = 模型能学到不同粒度的位置关系
        - 低维度 (高频) → 捕捉相邻位置关系
        - 高维度 (低频) → 捕捉远距离位置关系
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000):
        """
        Args:
            d_model: 编码维度 (和 token embedding 一样，才能相加)
            max_len: 最大序列长度 (需要提前定好)
        """
        super().__init__()

        # 创建一个 (max_len, d_model) 的矩阵来存位置编码
        pe = torch.zeros(max_len, d_model)

        # position: 每个位置的索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # div_term: 分母 10000^(2i/d_model)，用于控制频率
        # 每个维度的频率不同: 低维度变化快(高频)，高维度变化慢(低频)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        #  为什么用 exp? 因为 10000^(2i/d) = exp(2i * -ln(10000)/d)

        # 偶数维度: sin
        pe[:, 0::2] = torch.sin(position * div_term)   # 所有行, 偶数列
        #  0::2 = 从第 0 列开始，每隔 2 列取一次 → [0, 2, 4, 6, ...]

        # 奇数维度: cos
        pe[:, 1::2] = torch.cos(position * div_term)   # 所有行, 奇数列
        #  1::2 = 从第 1 列开始，每隔 2 列取一次 → [1, 3, 5, 7, ...]

        # register_buffer: 不是训练参数，但需要存到 state_dict 里 (随模型保存/加载)
        # 不需要梯度 (因为不是训练的)
        self.register_buffer('pe', pe.unsqueeze(0))    # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) — token embeddings

        Returns:
            (batch_size, seq_len, d_model) — x + positional encoding
        """
        seq_len = x.size(1)
        # 取出对应长度的位置编码，加到输入上
        return x + self.pe[:, :seq_len, :]


# ============================================================================
# 2. Learned Positional Embedding — GPT 的用法
# ============================================================================
class LearnedPositionalEmbedding(nn.Module):
    """
    GPT 的做法: 把位置也当成一种 embedding，让模型自己学。

    和 Sinusoidal 的区别:
        - Sinusoidal: 固定的，不需要训练，可以外推到更长的序列
        - Learned: 可训练的，更灵活，但不能超过 max_len

    实现: 就和 token embedding 一样，只不过 position 是索引而不是词
    """

    def __init__(self, d_model: int = 512, max_len: int = 2048):
        super().__init__()
        # nn.Embedding(num_embeddings, embedding_dim)
        # = 一个查找表: 输入索引 i，返回第 i 行的向量
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) — token embeddings

        Returns:
            (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
        # 查表: 每个位置 → 对应的 embedding 向量
        pos_embeddings = self.pos_embed(positions)           # (seq_len, d_model)
        return x + pos_embeddings


# ============================================================================
# 3. 可视化: 看位置编码长什么样
# ============================================================================
def visualize_positional_encoding():
    """
    可视化 Sinusoidal PE，直观理解 "低维高频，高维低频"。
    """
    d_model = 128
    max_len = 100

    pe = SinusoidalPositionalEncoding(d_model, max_len)
    pe_matrix = pe.pe.squeeze(0).numpy()           # (max_len, d_model)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # 图 1: 热力图 — 全面貌
    im = axes[0].imshow(pe_matrix.T, aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Position (token 位置)')
    axes[0].set_ylabel('Dimension (维度)')
    axes[0].set_title('Sinusoidal Positional Encoding 热力图\n'
                      '红色=正值, 蓝色=负值, 低维度变化快(高频), 高维度变化慢(低频)')
    plt.colorbar(im, ax=axes[0])

    # 图 2: 几个位置的曲线 — 看不同位置的编码如何不同
    for pos in [0, 10, 20, 50]:
        axes[1].plot(pe_matrix[pos, :32], label=f'pos={pos}', alpha=0.7)
    axes[1].set_xlabel('Dimension (前 32 维)')
    axes[1].set_ylabel('Value')
    axes[1].set_title('不同位置的 Positional Encoding 向量 (前 32 维)')
    axes[1].legend()

    plt.tight_layout()
    save_path = "/tmp/positional_encoding.png"
    plt.savefig(save_path, dpi=100)
    print(f"  图片保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Positional Encoding 演示")
    print("=" * 60)

    # --- 测试 Sinusoidal PE ---
    print("\n1. Sinusoidal Positional Encoding:")
    sin_pe = SinusoidalPositionalEncoding(d_model=512, max_len=2048)
    x = torch.randn(2, 100, 512)                    # 100 个 token
    out = sin_pe(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   值范围: [{out.min():.3f}, {out.max():.3f}]")
    print(f"   训练参数: {sum(p.numel() for p in sin_pe.parameters())} (不需要训练!)")
    print(f"   ✅ Sinusoidal PE OK")

    # --- 测试 Learned PE ---
    print("\n2. Learned Positional Embedding:")
    learned_pe = LearnedPositionalEmbedding(d_model=512, max_len=2048)
    out2 = learned_pe(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out2.shape}")
    print(f"   训练参数: {sum(p.numel() for p in learned_pe.parameters()):,}")
    print(f"   (512 * 2048 = {512 * 2048:,} 个可训练参数)")
    print(f"   ✅ Learned PE OK")

    # --- 验证: 不同位置的编码是不同的 ---
    print("\n3. 验证: 不同位置的编码应该不同:")
    pos0 = sin_pe.pe[0, 0, :10]                     # 位置 0
    pos10 = sin_pe.pe[0, 10, :10]                   # 位置 10
    pos50 = sin_pe.pe[0, 50, :10]                   # 位置 50
    print(f"   pos=0  前 10 维: {pos0.numpy().round(3)}")
    print(f"   pos=10 前 10 维: {pos10.numpy().round(3)}")
    print(f"   pos=50 前 10 维: {pos50.numpy().round(3)}")
    print(f"   pos=0 == pos=10: {torch.allclose(pos0, pos10)}")  # False = 不同!

    # --- 可视化 (可选) ---
    print("\n4. 生成可视化图片...")
    try:
        visualize_positional_encoding()
    except Exception as e:
        print(f"   可视化跳过 (可能是无头环境): {e}")

    print("\n" + "=" * 60)
    print("✅ Positional Encoding 完成！")
    print("核心: PE(pos, 2i)=sin(...), PE(pos, 2i+1)=cos(...)")
    print("作用: 让 Transformer 知道 '第一个词' 和 '最后一个词' 位置不同")
    print("下一步: 运行 03_transformer_block.py")
    print("=" * 60)
