"""
Encoder-Decoder Transformer — 完整的序列到序列模型
=======================================================
你已经有 Decoder-only (GPT)，现在补 Encoder-Decoder。

三者的区别:
    Encoder-only (BERT):   双向看全部 token → 理解任务 (分类、实体识别)
    Decoder-only (GPT):    causal只看左边 → 生成任务 (续写、对话)
    Encoder-Decoder (T5):  encoder 双向理解输入 → decoder causal 生成输出
                           → 翻译、摘要、VLA 里 language→action

VLA 场景下:
    Encoder: "拿起桌上的红色杯子" → 双向理解完整指令
    Decoder: causal 生成动作序列 → 每步只能看之前生成的动作

文件覆盖:
    01_self_attention.py     ← Self-Attention + Causal Mask
    02_positional_encoding.py
    03_cross_attention.py    ← Q(decoder) attend K/V(encoder)
    04_encoder_decoder.py    ← 本文件: 完整 Encoder-Decoder
    05_transformer_block.py  ← 通用 Transformer Block
    06_mini_gpt.py           ← Decoder-only (GPT)
    07_train_on_shakespeare.py
    08_vit.py                ← Encoder-only (ViT)

架构:
    Input → [Encoder ×N] → encoder_output
                                ↓
    Target → [Decoder ×N: Self-Attn(causal) + Cross-Attn(encoder)] → output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Encoder Layer — 双向 Self-Attention
# ============================================================================
class EncoderLayer(nn.Module):
    """和你的 TransformerBlock 一样，但去掉 causal mask（双向看）"""

    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # Self-Attention: 双向! src_mask 是 padding mask (非 causal)
        a, _ = self.self_attn(x, x, x, key_padding_mask=src_mask)
        x = x + a
        x = self.ln1(x)
        x = x + self.ffn(x)
        return self.ln2(x)


# ============================================================================
# Decoder Layer — Causal Self-Attn + Cross-Attn to Encoder
# ============================================================================
class DecoderLayer(nn.Module):
    """
    两层 attention:
        1. Self-Attention (causal) —只看之前生成的 token
        2. Cross-Attention — Q 来自 decoder, K/V 来自 encoder 输出
    """

    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        # Self-Attention with causal mask
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)

        # Cross-Attention: decoder → encoder
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Self-Attention (causal — tgt_mask 是上三角)
        a, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + a
        x = self.ln1(x)

        # 2. Cross-Attention: Q=decoder, K/V=encoder output
        a, _ = self.cross_attn(x, enc_output, enc_output, key_padding_mask=src_mask)
        x = x + a
        x = self.ln2(x)

        # 3. FFN
        x = x + self.ffn(x)
        return self.ln3(x)


# ============================================================================
# 完整 Encoder-Decoder Transformer
# ============================================================================
class EncoderDecoder(nn.Module):
    """
    完整 Encoder-Decoder，和 T5/BART 一致。

    例: 翻译 "Hello world" → "你好世界"
        Encoder: 双向读 "Hello world" → encoder_output
        Decoder: causal 生成 ["<s>", "你", "好", "世", "界"]

    VLA 场景:
        Encoder: 双向读 instruction + visual → fused features
        Decoder: causal 生成 action trajectory
    """

    def __init__(self, src_vocab=1000, tgt_vocab=1000, d_model=256,
                 n_heads=8, n_enc=3, n_dec=3, max_len=128):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)  # learned position

        # Encoder stack
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(n_enc)])
        self.enc_norm = nn.LayerNorm(d_model)

        # Decoder stack
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(n_dec)])
        self.dec_norm = nn.LayerNorm(d_model)

        self.output_head = nn.Linear(d_model, tgt_vocab)

    def _create_causal_mask(self, sz, device):
        """上三角 causal mask: True=不能看"""
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def forward(self, src_ids, tgt_ids, src_pad_mask=None):
        """
        src_ids: (B, S) — 源序列 token IDs
        tgt_ids: (B, T) — 目标序列 token IDs (训练时 teacher forcing)
        """
        B, S = src_ids.shape
        _, T = tgt_ids.shape

        # 位置编码
        src_pos = torch.arange(S, device=src_ids.device).unsqueeze(0)
        tgt_pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0)

        # ── Encoder ──
        x = self.src_embed(src_ids) + self.pos_embed(src_pos)
        for layer in self.encoder:
            x = layer(x, src_pad_mask)
        enc_out = self.enc_norm(x)

        # ── Decoder ──
        y = self.tgt_embed(tgt_ids) + self.pos_embed(tgt_pos)
        causal_mask = self._create_causal_mask(T, y.device)

        for layer in self.decoder:
            y = layer(y, enc_out, src_pad_mask, causal_mask)

        y = self.dec_norm(y)
        return self.output_head(y)                       # (B, T, tgt_vocab)

    @torch.no_grad()
    def generate(self, src_ids, max_len=50, bos_id=0, eos_id=1):
        """自回归生成（翻译/摘要 推理）"""
        self.eval()
        B = src_ids.shape[0]
        src_pos = torch.arange(src_ids.shape[1], device=src_ids.device).unsqueeze(0)

        # Encoder 只跑一次!
        x = self.src_embed(src_ids) + self.pos_embed(src_pos)
        for layer in self.encoder:
            x = layer(x)
        enc_out = self.enc_norm(x)

        # Decoder 自回归生成
        generated = torch.full((B, 1), bos_id, device=src_ids.device, dtype=torch.long)

        for _ in range(max_len):
            t = generated.shape[1]
            tgt_pos = torch.arange(t, device=src_ids.device).unsqueeze(0)
            y = self.tgt_embed(generated) + self.pos_embed(tgt_pos)
            causal_mask = self._create_causal_mask(t, y.device)

            for layer in self.decoder:
                y = layer(y, enc_out, None, causal_mask)

            logits = self.output_head(self.dec_norm(y[:, -1:]))
            next_token = logits.argmax(-1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return generated


# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Encoder-Decoder Transformer 演示")
    print("=" * 60)

    model = EncoderDecoder(src_vocab=200, tgt_vocab=200, d_model=128, n_heads=4, n_enc=2, n_dec=2)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟翻译: 源=10 tokens, 目标=8 tokens
    src = torch.randint(0, 200, (2, 10))
    tgt = torch.randint(0, 200, (2, 8))

    # Forward
    logits = model(src, tgt)
    print(f"源: {src.shape}, 目标: {tgt.shape}")
    print(f"输出: {logits.shape} ← (B=2, T=8, vocab=200)")

    # Loss 验证 (teacher forcing)
    loss = F.cross_entropy(logits.view(-1, 200), tgt.view(-1))
    print(f"Loss: {loss:.4f}")

    # 生成
    gen = model.generate(src[:1], max_len=20)
    print(f"生成: {gen.shape} ← 自回归输出序列")

    print(f"\n架构验证:")
    print(f"  Encoder: 双向 Self-Attention ×{len(model.encoder)}")
    print(f"  Decoder: Causal Self-Attn + Cross-Attn ×{len(model.decoder)}")
    print(f"  ✅ Encoder-Decoder OK — 补全了你的 Transformer 体系!")
    print(f"  现在你有: Self-Attn + Cross-Attn + Encoder + Decoder — 全套齐了")
