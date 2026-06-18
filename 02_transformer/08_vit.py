# 运行: conda activate diffusion && python 02_transformer/08_vit.py
"""
ViT (Vision Transformer) — 把图像变成 token
===============================================
这就是 VLA 的 "V" — 视觉编码器。

    Image (3×224×224) → Patch Embed → +Pos Embed → N×Transformer Block → [CLS] token
                                                                             ↓
                                                                       visual features
                                                                       (→ Cross-Attn → Action)

和 GPT 的区别:
    GPT:   token ID → Embedding → causal Transformer → next token
    ViT:   image patch → Embedding → bidirectional Transformer → [CLS] feature

你已有的知识可以直接迁移:
    Patch Embed = 你 DiT 的 PatchEmbed (05_dit.py)
    Transformer Block = 你 02_transformer/03_transformer_block.py（去掉 causal mask）
    [CLS] token = 和 BERT 一样，在序列前面加一个可学习的 token，用它输出当全局特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
class ViT(nn.Module):
    """Vision Transformer — 工业标准视觉编码器（SigLIP/CLIP/DINO 都是这架构）"""

    def __init__(self, img_size=224, patch_size=16, channels=3,
                 d_model=384, n_heads=6, n_layers=12, output_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch Embedding (和你 DiT 里的 PatchEmbed 一样)
        self.patch_embed = nn.Conv2d(channels, d_model, patch_size, stride=patch_size)

        # [CLS] token — 可学习的全局表示
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))

        # Transformer Blocks — 双向！no causal mask!
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)    # 最终视觉特征

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        返回: vis_feat (B, output_dim)  ← VLA 的 Cross-Attention 输入
        """
        B = x.shape[0]

        # 1. Patch Embedding
        x = self.patch_embed(x)                         # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)                # (B, N, D)

        # 2. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (B, N+1, D)

        # 3. Add position embedding
        x = x + self.pos_embed

        # 4. Transformer (双向)
        for block in self.blocks:
            x = block(x)

        # 5. [CLS] token → visual feature
        x = self.norm(x[:, 0])                           # 只取 [CLS]
        return self.head(x)


# ============================================================================
# 极简版 TransformerBlock（双向，和你 02_transformer 一样只是去掉了 causal）
# ============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model=384, n_heads=6, expansion=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        #  ↑ 直接用 PyTorch 内置 MHA，可以和你手写的对比学习
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x):
        # Attention + Residual (双向，无 causal mask!)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # MLP + Residual
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# 完整 VLA: ViT + Cross-Attention → Action
# ============================================================================
class VLAPipeline(nn.Module):
    """
    极简完整 VLA pipeline:
        Image → ViT → visual_feat     (vision encoder)
        Text  → GPT → lang_feat        (language encoder — 你已经会了)
        visual_feat + lang_feat → Cross-Attn → Action Head → robot action
    """

    def __init__(self, action_dim=7, vis_dim=256, lang_dim=256):
        super().__init__()
        self.vision = ViT(img_size=224, output_dim=vis_dim)

        # 假装 language encoder 已经有了（你的 MiniGPT 就是）
        # 实际 VLA 里语言侧也是 Transformer

        # 融合 + 动作
        self.fusion = nn.Sequential(
            nn.Linear(vis_dim + lang_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, img, lang_feat):
        vis_feat = self.vision(img)                     # (B, vis_dim)
        fused = torch.cat([vis_feat, lang_feat], -1)    # concat 融合
        return self.fusion(fused)                        # (B, act_dim)


# ============================================================================
if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"设备: {device}")

    # 1. ViT 单独测试
    vit = ViT(img_size=224, patch_size=16, d_model=384, n_layers=6)
    img = torch.randn(2, 3, 224, 224)
    feat = vit(img)
    print(f"Image:  {img.shape}")
    print(f"Feature: {feat.shape}  ← 2张图 → 2个256d向量")
    print(f"Patch 数: {vit.n_patches} ({14}×{14}) + 1 [CLS] = {vit.n_patches + 1}")
    print(f"参数量: {sum(p.numel() for p in vit.parameters()):,}")
    print(f"✅ ViT OK")

    # 2. 完整 VLA pipeline
    vla = VLAPipeline()
    lang_feat = torch.randn(2, 256)                   # 假装GPT编码的指令
    action = vla(img, lang_feat)
    print(f"\nVLA Pipeline:")
    print(f"  Image + Lang → Action: {action.shape}   (例: 7DoF 力矩)")
    print(f"✅ VLA Pipeline OK — 这就是完整 VLA 架构!")
