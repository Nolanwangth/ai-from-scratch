"""
Multi-Head Self-Attention — Transformer 的心脏
================================================
"If you understand attention, you understand Transformer."

Self-Attention 做什么？
    每个 token 和所有 token 比较相似度 → 加权平均 → 得到"上下文感知"的表示

直观理解 (用一句话做例子):
    "The cat sat on the mat because it was tired."
    Q: "it" 指代什么?
    A: Self-attention 让 "it" 去"看"前面所有词，发现 "cat" 的相似度最高
       → "it" 的表示会混入 "cat" 的信息

Multi-Head 是什么？
    多组 Q/K/V，每组关注不同的关系:
    一组关注"指代关系" (it ↔ cat)
    一组关注"语法结构" (was ↔ tired)
    一组关注"语义信息" (cat ↔ sat)

数学:
    Q = X @ W_Q           ← 查询 (我要找什么?)
    K = X @ W_K           ← 键   (我有什么信息?)
    V = X @ W_V           ← 值   (我的实际内容)
    Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Flash Attention: 高效版本的 attention，O(n^2) → 更少的 I/O 操作
    → PyTorch 2.0+ 内置 scaled_dot_product_attention (如果硬件支持会自动用 Flash Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. 单头 Attention — 理解核心机制
# ============================================================================
class SingleHeadAttention(nn.Module):
    """
    最简单的单头 Attention，帮助理解 Multi-Head。

    运算步骤:
        1. Q = X @ W_Q   (batch, seq_len, d_model) @ (d_model, d_head) → (batch, seq_len, d_head)
        2. K = X @ W_K   同上
        3. V = X @ W_V   同上
        4. scores = Q @ K^T / sqrt(d_head)  → 算每对 token 的相似度
        5. attn_weights = softmax(scores)    → 归一化成概率
        6. output = attn_weights @ V          → 加权平均
    """

    def __init__(self, d_model: int = 512, d_head: int = 64):
        """
        Args:
            d_model: 输入维度 (token embedding 的大小, 如 512)
            d_head:  每个 attention head 的维度 (如 64)
        """
        super().__init__()
        self.d_head = d_head

        # W_Q, W_K, W_V — 3 个线性变换，把输入投影到 "查询/键/值" 空间
        # nn.Linear 做了什么? output = input @ W^T + b
        # nn.Linear(in_features, out_features, bias=True)
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) — 输入序列

        Returns:
            output: (batch_size, seq_len, d_head) — 被 attention 增强后的表示
        """
        B, T, D = x.shape                      # Batch, Time(seq_len), Dimension

        # Step 1: 投影到 Q/K/V 空间
        Q = self.W_Q(x)                         # (B, T, d_head)
        K = self.W_K(x)                         # (B, T, d_head)
        V = self.W_V(x)                         # (B, T, d_head)

        # Step 2: 计算注意力分数 — Q @ K^T
        # Q @ K^T: (B, T, d_head) @ (B, d_head, T) = (B, T, T)
        # 结果是一个 T×T 矩阵: [i,j] 表示 "第 i 个 token 对第 j 个 token 的关注程度"
        attn_scores = Q @ K.transpose(-2, -1)    # (B, T, T)
        #  ^ transpose(-2, -1) = 交换最后两维 = K 的转置

        # Step 3: 缩放 — 除以 sqrt(d_head)
        # 为什么? 如果 d_head 很大，Q@K^T 的值会很大
        # → softmax 的输入很大 → 梯度很小 → 训练不动
        # 除以 sqrt(d_head) 让方差稳定在 1
        attn_scores = attn_scores / math.sqrt(self.d_head)

        # Step 4: Softmax — 把分数变成概率 (每行和为 1)
        # dim=-1 表示对最后一维做 softmax(即每个 query 对所有 key)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)

        # Step 5: 加权平均 — 用注意力权重加权 V
        output = attn_weights @ V                 # (B, T, T) @ (B, T, d_head) = (B, T, d_head)

        return output


# ============================================================================
# 2. Multi-Head Attention — 实际使用的版本
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention = 多组 Single-Head Attention 并行 + concat

    为什么需要 multi-head?
        不同的 head 关注不同的模式:
        - Head 1: 主语-谓语关系
        - Head 2: 代词指代关系
        - Head 3: 形容词修饰关系
        ...

    实现细节:
        不是创建 N 个独立的 SingleHeadAttention，而是用一个大的投影矩阵
        → Q = X @ W_Q_big  (形状: d_model → n_heads * d_head)
        → 然后 reshape 成 (B, T, n_heads, d_head)
        → 每个 head 独立算 attention
        → concat 回 (B, T, d_model)
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: 输入/输出维度 (必须是 n_heads 的整数倍!)
            n_heads: 几组 attention 并行
            dropout: attention weights 的 dropout 概率

            通常: d_model=512, n_heads=8, d_head = 512/8 = 64
        """
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除!"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads           # 每个 head 的维度

        # 合并投影: 4 * d_model (Q,K,V + output 各一个)
        # 为什么不分开 3 个? 合在一起快一点点 (一次大矩阵乘 vs 三次小矩阵乘)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        #  3 * d_model = d_model (for Q) + d_model (for K) + d_model (for V)

        # 输出投影: 把 multi-head concat 的结果投影回 d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len) or None
                  True=不让你看 (如 decoder 里不能看未来 token)

        Returns:
            (batch_size, seq_len, d_model)
        """
        B, T, D = x.shape

        # Step 1: 投影到 Q, K, V — 一次矩阵乘拿下
        qkv = self.qkv_proj(x)                      # (B, T, 3*D)
        # 把 3*D 切成 3 份，每份 D
        Q, K, V = qkv.chunk(3, dim=-1)              # 每个: (B, T, D)

        # Step 2: Reshape 成 multi-head 形状
        # (B, T, D) → (B, T, n_heads, d_head) → (B, n_heads, T, d_head)
        # 为什么要 transpose? 因为后面 @ 运算需要在 d_head 上做
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # 现在每个都是: (B, n_heads, T, d_head)

        # Step 3: Scaled Dot-Product Attention
        # Q @ K^T: (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) = (B, n_heads, T, T)
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # Step 4: 应用 mask (如果提供的话)
        # 场景: decoder 里的 causal mask — 不让 token 看到它后面的 token
        if mask is not None:
            # mask 里的 True → 设为 -inf (softmax(-inf) = 0，完全不关注)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Step 5: Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 6: 加权平均
        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) = (B, n_heads, T, d_head)
        out = attn_weights @ V

        # Step 7: 拼回原来的形状
        # (B, n_heads, T, d_head) → (B, T, n_heads, d_head) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        #                    ↑ contiguous(): 因为 transpose 不是连续的，view 需要连续内存

        # Step 8: 输出投影
        out = self.out_proj(out)

        return out


# ============================================================================
# 3. Flash Attention — PyTorch 2.0 内置高效版本
# ============================================================================
class FlashMultiHeadAttention(nn.Module):
    """
    使用 PyTorch 2.0+ 的 scaled_dot_product_attention — 自动选择最优实现
    如果你的硬件支持 (A100/H100 GPU 或新版 MPS)，PyTorch 会自动用 Flash Attention。
    Flash Attention = O(n^2) 的内存 → O(n)，能处理更长的序列。

    用法和上面的 MultiHeadAttention 一模一样，只是用
    F.scaled_dot_product_attention 代替了手写的 attention 步骤。
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ⭐ 核心: 这一行代替了手写的 scores/softmax/weights 三四行
        # PyTorch 自动决定用 Flash Attention, Memory-efficient attention 还是传统 attention
        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=(mask is not None),  # ⚠️ is_causal=True 时不要同时传 attn_mask!
        )
        #  返回: (B, n_heads, T, d_head)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out


# ============================================================================
# 4. Causal Mask — 让模型不能偷看未来
# ============================================================================
def create_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    创建因果遮罩 (causal mask / 上三角 mask)。
    用于 GPT 类模型 (decoder-only): token i 只能看到 token [0, 1, ..., i]，不能看到后面的。

    直观理解:
        预测 "The cat sat" 的第 3 个词时：
        - 可以看到: "The", "cat"
        - 不能看到: "sat" (以及之后的词，因为那是要预测的)

    返回的 mask 形状是 (1, 1, seq_len, seq_len)，可以直接广播到 batch。
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # torch.triu: 上三角为 1, 下三角为 0
    # diagonal=1: 第一个上对角线及以上为 1 (不包括对角线本身)
    #
    # 例如 seq_len=4:
    # [[0, 1, 1, 1],    ← token 0 只能看到自己
    #  [0, 0, 1, 1],    ← token 1 能看到 0 和 1
    #  [0, 0, 0, 1],    ← token 2 能看到 0,1,2
    #  [0, 0, 0, 0]]    ← token 3 能看到全部

    return mask.unsqueeze(0).unsqueeze(0)         # (seq_len,seq_len) → (1,1,seq_len,seq_len)


# ============================================================================
# 5. 演示 + 验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Head Attention 演示")
    print("=" * 60)

    # --- 单头 Attention ---
    print("\n1. 单头 Attention:")
    single = SingleHeadAttention(d_model=512, d_head=64)
    x = torch.randn(2, 10, 512)                    # (batch=2, seq_len=10, d_model=512)
    out = single(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   ✅ 单头 OK")

    # --- Multi-Head Attention ---
    print("\n2. Multi-Head Attention (8 heads):")
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    out = mha(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   d_head = 512/8 = 64")
    print(f"   参数量: {sum(p.numel() for p in mha.parameters()):,}")
    print(f"   ✅ Multi-Head OK")

    # --- Causal Mask ---
    print("\n3. Causal Mask (长度为 5 的序列):")
    seq_len = x.shape[1]  # 用实际序列长度，不是硬编码!
    mask = create_causal_mask(seq_len)
    print(f"   mask shape: {mask.shape}")
    print(f"   True = 不能看 (上三角), False = 可以看 (下三角):")
    print(f"   {mask.squeeze().numpy()}")
    print(f"   例如: token 2 只能看到 token [0,1,2], 不能看 [3,4]")

    # --- Multi-Head + Causal Mask (GPT 的标准输入) ---
    print("\n4. Multi-Head + Causal Mask (GPT 模式):")
    causual_out = mha(x, mask=mask)
    print(f"   输入: {x.shape}")
    print(f"   输出: {causual_out.shape}")
    print(f"   ✅ Mask 防止偷看未来")

    # --- Flash Attention ---
    print("\n5. Flash Attention (自动优化):")
    flash_mha = FlashMultiHeadAttention(d_model=512, n_heads=8)
    flash_out = flash_mha(x, mask=mask)
    print(f"   输入: {x.shape}")
    print(f"   输出: {flash_out.shape}")
    print(f"   ✅ Flash Attention OK")

    print("\n" + "=" * 60)
    print("Attention 机制完成！")
    print("下一步: 运行 02_positional_encoding.py")
    print("=" * 60)
