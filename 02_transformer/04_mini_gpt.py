# 运行: conda activate diffusion && python 02_transformer/04_mini_gpt.py
"""
Mini-GPT: 从零实现 Decoder-Only Transformer
=============================================
这是整个 Transformer 章节的最终形态 — 一个可以训练、可以生成文本的 GPT 模型。

GPT 架构 (从上到下):
    输入: token IDs → [Embedding] → [Position Encoding] → [N 个 Transformer Blocks] → [LM Head] → logits

GPT 和 BERT 的区别:
    GPT (Decoder-only): 只能看前面的 token (causal)，做 next-token prediction
    BERT (Encoder-only): 能看到全部 token (bidirectional)，做 fill-in-the-blank

GPT 的训练目标 — Next Token Prediction:
    输入:  "The cat sat on"
    目标:  预测 → "cat sat on the"
    本质:  给定前 N-1 个 token，预测第 N 个 token
    损失:  CrossEntropyLoss(logits, target_tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import importlib
TransformerBlock = importlib.import_module("03_transformer_block").TransformerBlock


# ============================================================================
# 1. GPT Config — 所有超参数集中管理
# ============================================================================
class GPTConfig:
    """
    (Best Practice) 用 config 对象管理所有超参数，而不是散落在代码各处。

    为什么不用 dict?
        1. class 有类型提示，IDE 能自动补全
        2. 属性访问 config.d_model 比 config['d_model'] 更安全 (IDE 会报 typo)
        3. 可以加验证逻辑 (比如 d_model 必须能被 n_heads 整除)
    """

    def __init__(
        self,
        vocab_size: int = 50257,        # 词表大小 (GPT-2 用 50257)
        block_size: int = 256,          # 最大序列长度 (context window)
        n_layer: int = 6,              # Transformer block 层数
        n_head: int = 8,               # Attention head 数
        d_model: int = 256,            # 隐藏维度 (token embedding 的大小)
        dropout: float = 0.1,          # Dropout 概率
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout

        # 验证配置合法性
        assert d_model % n_head == 0, \
            f"d_model ({d_model}) 必须能被 n_head ({n_head}) 整除!"

    @property
    def d_head(self) -> int:
        """每个 attention head 的维度"""
        return self.d_model // self.n_head


# ============================================================================
# 2. GPT Model — 完整实现
# ============================================================================
class MiniGPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) 的完整实现。

    架构图:
        ┌─────────────────────────┐
        │   Token Embedding       │  把 token ID → 稠密向量
        ├─────────────────────────┤
        │   Position Embedding    │  把位置 ID → 稠密向量 (GPT 用 learned, 不是 sinusoidal)
        ├─────────────────────────┤
        │   Dropout               │
        ├─────────────────────────┤
        │   Transformer Block #0  │
        │   Transformer Block #1  │
        │   ...                   │  N 层堆叠
        │   Transformer Block #N  │
        ├─────────────────────────┤
        │   LayerNorm             │  最后一层归一化
        ├─────────────────────────┤
        │   LM Head (Linear)      │  把 d_model 投影到 vocab_size
        └─────────────────────────┘
          ↓
        logits: (B, T, vocab_size) — 每个位置预测下一个 token 的概率分布

    参数量估算 (GPT-2 级别):
        Small:  d_model=768,  n_layer=12 → 124M 参数
        Medium: d_model=1024, n_layer=24 → 350M 参数
        Large:  d_model=1280, n_layer=36 → 774M 参数
        XL:     d_model=1600, n_layer=48 → 1.5B 参数
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Token Embedding ──
        # nn.Embedding(num_embeddings, embedding_dim)
        # 本质: 一个 weight 矩阵 (vocab_size, d_model)
        # 输入 token ID, 返回该 ID 对应的第几行
        # 训练时会学到: 语义相似的 token 的 embedding 向量会靠得很近
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # ── Position Embedding (GPT 用 Learned, 不是 Sinusoidal) ──
        # 为什么 GPT 用 Learned? 因为有足够的训练数据，可以学到更灵活的位置表示
        self.pos_embed = nn.Embedding(config.block_size, config.d_model)

        # ── Dropout (正则化，防止过拟合) ──
        self.drop = nn.Dropout(config.dropout)

        # ── Transformer Blocks ──
        # nn.ModuleList = 像 list 但要注册参数 (这样 optimizer 才能找到它们!)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_head, config.dropout)
            for _ in range(config.n_layer)
        ])

        # ── 最后的 LayerNorm ──
        # GPT-2 在 lm_head 之前加了一个 LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)

        # ── LM Head: 把 d_model 投影到 vocab_size ──
        # 输出 logits: 每个位置的词表概率分布
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ⭐ Weight Tying (权重绑定): LM Head 和 Token Embedding 共享权重
        # 原因: 如果两个 token 的 embedding 很接近，那它们的输出概率也应该接近
        # 能省 (vocab_size * d_model) 个参数!
        self.token_embed.weight = self.lm_head.weight

        # ── 初始化 ──
        self.apply(self._init_weights)               # 递归对所有子模块初始化
        print(f"MiniGPT 参数量: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module: nn.Module):
        """GPT-2 风格的权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx:     (B, T) — token ID 序列
            targets: (B, T) — 目标 token IDs (训练时提供，推理时 = None)

        Returns:
            logits: (B, T, vocab_size) — 每个位置的预测分布
            loss:   scalar — CrossEntropyLoss (只在训练时返回，推理时 = None)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"序列长度 ({T}) 超过 block_size ({self.config.block_size})"

        # ── Step 1: Token Embedding ──
        tok_emb = self.token_embed(idx)              # (B, T) → (B, T, d_model)

        # ── Step 2: Position Embedding ──
        # 生成位置索引 [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, device=idx.device)  # (T,)
        pos_emb = self.pos_embed(pos)                 # (T,) → (T, d_model)

        # ── Step 3: 元素相加 ──
        x = tok_emb + pos_emb                         # (B, T, d_model)
        x = self.drop(x)

        # ── Step 4: 创建 Causal Mask ──
        # 让 token i 不能看到 token i+1, i+2, ... (只看过去，不看未来)
        mask = torch.triu(
            torch.ones(T, T, device=idx.device), diagonal=1
        ).bool()                                      # (T, T), 上三角=True
        mask = mask.unsqueeze(0).unsqueeze(0)         # (1, 1, T, T)

        # ── Step 5: 经过 N 层 Transformer Blocks ──
        for block in self.blocks:
            x = block(x, mask=mask)

        # ── Step 6: 最后的 LayerNorm ──
        x = self.ln_f(x)

        # ── Step 7: LM Head → logits ──
        logits = self.lm_head(x)                      # (B, T, d_model) → (B, T, vocab_size)

        # ── Step 8: 计算 loss (训练时) ──
        loss = None
        if targets is not None:
            # CrossEntropyLoss 期望的输入形状:
            #   input:  (N, C) 其中 N=样本数, C=类别数
            #   target: (N,)   整数标签
            # 所以我们要把 (B, T, vocab_size) → (B*T, vocab_size)
            #          和 (B, T) → (B*T)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),     # (B*T, vocab_size)
                targets.view(-1),                     # (B*T)
                ignore_index=-1                       # -1 是 padding token, 不算 loss
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        文本生成 (自回归采样)。

        生成过程:
            while len < max_new_tokens:
                logits = model(current_sequence)      # 预测下一个 token 的分布
                logits = logits[:, -1, :] / temperature  # 只取最后一个位置的预测
                probs = softmax(logits)                # 变成概率
                next_token = sample(probs)             # 按概率采样
                current_sequence = [current_sequence, next_token]

        为什么取最后一个位置?
            输入 [A, B, C] → model → logits (3, vocab_size)
            logits[0] = 预测 B (给定 A 时)
            logits[1] = 预测 C (给定 A,B 时)
            logits[2] = 预测 ? (给定 A,B,C 时) ← 我们需要这个!

        Args:
            idx:              (1, T) — 起始序列 (如 ["Once", "upon"])
            max_new_tokens:   生成多少个新 token
            temperature:      温度 (0 = 贪心取最大, 1 = 正常, >1 = 更随机)

        Returns:
            (1, T + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 如果序列太长，只保留最后 block_size 个 token
            idx_cond = idx[:, -self.config.block_size:]

            # 前向传播
            logits, _ = self(idx_cond)
            #  返回 (B, T, vocab_size)

            # 只取最后一个位置的 logits
            logits = logits[:, -1, :]                 # (B, vocab_size)

            # Temperature scaling: 控制随机性
            # temperature → 0: 越确定性 (接近 greedy)
            # temperature → ∞: 越随机
            logits = logits / max(temperature, 1e-8)

            # 按概率采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 拼接到序列末尾
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    @torch.no_grad()
    def generate_top_k_top_p(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        改进版生成: top-k + top-p (nucleus) 采样。

        Top-k:  只从概率最高的 k 个 token 里采样 (砍掉低概率的 "噪音")
        Top-p:  只从累积概率 ≥ p 的最少 token 里采样 (动态调整)

        组合使用 → 生成质量显著提升，避免产生无意义的 token。
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k: 只保留最高的 k 个
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_values[:, -1:]] = float('-inf')

            # Top-p (nucleus): 按概率从高到低加，直到累积 > p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 把累积概率 > top_p 的砍掉
                sorted_logits[cumulative_probs > top_p] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ============================================================================
# 3. 快速测试
# ============================================================================
def test_forward_backward():
    """
    验证模型能正常 forward + backward，不会报错。
    这是 AI 工程师的日常操作: 写完模型先跑一遍 forward/backward 验证形状正确。
    """
    device = torch.device("cpu")                     # 测试用 CPU

    config = GPTConfig(
        vocab_size=1000,                              # 小词表，测试用
        block_size=128,
        n_layer=4,
        n_head=4,
        d_model=128,
    )

    model = MiniGPT(config).to(device)

    # 测试: forward pass
    B, T = 2, 64  # batch=2, seq=64
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx, targets)

    print(f"Forward: idx={idx.shape}, logits={logits.shape}, loss={loss.item():.4f}")

    # 测试: backward pass (验证梯度能正常流回去)
    loss.backward()

    # 检查: 所有参数都有梯度
    no_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
    assert len(no_grad_params) == 0, f"这些参数没有梯度: {no_grad_params}"
    print(f"Backward: 所有 {sum(1 for _ in model.parameters())} 个参数都有梯度 ✅")

    # 测试: 生成
    start_tokens = torch.randint(0, config.vocab_size, (1, 4))  # 4 个起始 token
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8)
    print(f"Generate: 输入 {start_tokens.shape} → 输出 {generated.shape} ✅")

    return model, config


if __name__ == "__main__":
    print("=" * 60)
    print("Mini-GPT 完整模型测试")
    print("=" * 60)

    model, config = test_forward_backward()

    print(f"\n模型概述:")
    print(f"  词表大小:     {config.vocab_size}")
    print(f"  最大序列长度:  {config.block_size}")
    print(f"  Transformer 层: {config.n_layer}")
    print(f"  Attention 头:  {config.n_head}")
    print(f"  隐藏维度:      {config.d_model}")
    print(f"  每头维度:      {config.d_head}")
    print(f"  总参数量:      {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("✅ Mini-GPT 模型完成！")
    print("核心架构: Embedding → N×TransformerBlock → LN → Linear → logits")
    print("训练目标: Next Token Prediction (预测下一个词)")
    print("下一步: 运行 05_train_on_shakespeare.py 训练它!")
    print("=" * 60)
