# 运行: conda activate diffusion && python 03_diffusion/06_dit.py
"""
DiT (Diffusion Transformer) — 用 Transformer 替换 U-Net
=========================================================
这是现代扩散模型的标准架构。Stable Diffusion 3、Sora、Flux 都用 DiT。

U-Net vs DiT:
    U-Net: CNN backbone + 下采样/上采样 + skip connections
           → 好: inductive bias (卷积天然适合图像)
           → 坏: 不好 scale，做不大

    DiT:  ViT (Vision Transformer) backbone + AdaLN
           → 好: 可无限 scale (和 LLM 一样)，图像分 patch 当 token
           → 坏: 需要更多数据

核心思想：把图像切成 patch，当成 token 输入 Transformer。

架构（DiT Block）：
    x ──→ [AdaLN] ──→ [Multi-Head Self-Attention] ──→ (+) ──→ [AdaLN] ──→ [MLP] ──→ (+)
              ↑                                      ↑          ↑
              └── 时间 t + 类别 c ──────────────────────┘

    和你的 Transformer Block (02_transformer/03) 的区别：
        1. LayerNorm → AdaLN (Adaptive Layer Norm)
           → 不是简单的归一化，而是根据时间步 t 和类别 c 动态调整 scale/shift
        2. Attention 不做 causal（双向都看）
        3. 输出维度 = patch 数 × patch_size² × channels

AdaLN (Adaptive Layer Normalization):
    普通 LN:  y = (x - μ)/σ * γ + β           (γ, β 是固定的)
    AdaLN:   y = (x - μ)/σ * (1+γ(t,c)) + β(t,c)
                                      ↑
                    γ 和 β 由时间步 t 和条件 c 决定
                    → 网络知道 "现在 t=500 噪声很大，要大力去噪"

参考：Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
# 1. AdaLN — Adaptive Layer Normalization（DiT 的灵魂）
# ============================================================================
class AdaLN(nn.Module):
    """
    自适应层归一化：根据条件信息 (t, c) 动态生成 γ 和 β。

    输入 x: (B, N, D) — N 个 patch token
    条件 c: (B, D)   — 时间步编码 + 类别编码 拼起来
    输出:   (B, N, D) — 自适应归一化后的特征

    为什么比普通 LN 好？
        普通 LN: 对任何输入用同样的 γ, β
        AdaLN:   不同时间步/类别用不同的 γ, β
                  → 模型知道 "现在噪声很大" 或 "这是猫的图片"
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        #  ^ elementwise_affine=False: 不用自带的 γ,β，我们自己从条件生成
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, d_model * 6),       # 6 = 3 个 (scale, shift, gate) × 2 层
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple:
        """
        Returns:
            x_normed: 归一化并调制后的 x
            3 组 (scale, shift, gate): 分别给 attention、MLP 使用
        """
        # 生成 6 个调制参数
        modulation = self.adaLN_modulation(c).chunk(6, dim=-1)
        # modulation = [γ1, β1, α1, γ2, β2, α2]
        #   γ = scale (缩放)
        #   β = shift (平移)
        #   α = gate  (门控 — 控制这个分支走多少)

        shift, scale, gate = modulation[0], modulation[1], modulation[2]

        # AdaLN: (x - μ)/σ * (1 + scale) + shift
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        #    ↑ unsqueeze(1): (B, D) → (B, 1, D) 方便广播到 N 个 token

        return x, gate.unsqueeze(1), modulation[3], modulation[4], modulation[5]


# ============================================================================
# 2. DiT Block — Transformer Block with AdaLN
# ============================================================================
class DiTBlock(nn.Module):
    """
    一个 DiT Block = Attention + MLP，都用 AdaLN 调制。

    和你的 TransformerBlock (02_transformer/03) 对比：
        相同: Self-Attention → Residual → MLP → Residual
        不同: LayerNorm → AdaLN, 不加 causal mask (双向)
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, cond_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # AdaLN for attention branch
        self.adaLN_attn = AdaLN(d_model, cond_dim)

        # Self-Attention (双向，不加 causal mask!)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # AdaLN for MLP branch
        self.adaLN_mlp = AdaLN(d_model, cond_dim)

        # MLP (和你的 Transformer FFN 一样)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — patch tokens
            c: (B, cond_dim) — 条件 (时间步 + 可选类别)

        Returns:
            (B, N, D)
        """
        # ── Attention Branch ──
        x_norm, gate_attn, _, _, _ = self.adaLN_attn(x, c)

        B, N, D = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)                          # 每个: (B, N, n_heads, d_head)

        # 转成 (B, n_heads, N, d_head) 做 multi-head attention
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        # Scaled Dot-Product Attention (双向! no causal mask)
        attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.out_proj(attn_out)

        # Gate: 控制 attention 分支走多少
        x = x + gate_attn * attn_out                     # 残差 + 门控!

        # ── MLP Branch ──
        x_norm2, gate_mlp, _, _, _ = self.adaLN_mlp(x, c)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp * mlp_out                       # 残差 + 门控!

        return x


# ============================================================================
# 3. Patch Embedding — 把图像切成 patch（ViT 的思路）
# ============================================================================
class PatchEmbed(nn.Module):
    """
    图像 → patch token 序列。

    例如：32×32 的图，patch_size=4 → 切成 (32/4)×(32/4) = 8×8 = 64 个 patch
    每个 patch = 4×4×3 = 48 维 → 投影到 d_model

    这就是 LLM 里 "图像 token" 的来源！
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2   # 总共多少个 patch

        # 一个 stride=patch_size 的卷积 = patch 切分 + 投影一步完成
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, n_patches, d_model)
        """
        x = self.proj(x)                                 # (B, D, H/p, W/p)
        x = x.flatten(2)                                 # (B, D, N)
        x = x.transpose(1, 2)                            # (B, N, D)
        return x


# ============================================================================
# 4. DiT Model — 完整 Diffusion Transformer
# ============================================================================
class DiT(nn.Module):
    """
    Diffusion Transformer — 用 Transformer 做去噪。

    架构：
        Image → Patch Embed → + Position Embed → N × DiTBlocks → Final Layer → Unpatch → Noise Prediction
                         ↑
                    Time Embed + Class Embed → cond vector
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        d_model: int = 384,                              # Transformer 隐藏维度
        n_heads: int = 6,
        n_layers: int = 12,                              # DiT Block 层数
        num_classes: int = 10,                           # 类别数 (0 = unconditional)
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size) ** 2
        self.d_model = d_model

        # ── Patch Embedding ──
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model)

        # ── Position Embedding (learned, 和 GPT 一样) ──
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        # ── 时间步 Embedding ──
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # ── 类别 Embedding (可选) ──
        self.num_classes = num_classes
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, d_model)
        self.cond_dim = d_model  # 条件维度 = time_embed (可能 + class_embed)

        # ── DiT Blocks ──
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, d_model) for _ in range(n_layers)
        ])

        # ── Final Layer (AdaLN → Linear → unpatch) ──
        self.final_adaLN = AdaLN(d_model, d_model)
        self.final_linear = nn.Linear(d_model, patch_size * patch_size * in_channels)

        self._init_weights()

    def _init_weights(self):
        # DiT 论文的初始化策略
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — 噪声图像 x_t
            t: (B,) — 时间步
            y: (B,) — 类别标签 (可选，unconditional generation 时不传)

        Returns:
            (B, C, H, W) — 预测的噪声
        """
        B, C, H, W = x.shape

        # ── 1. Patch Embedding ──
        x = self.patch_embed(x)                          # (B, N, D)

        # ── 2. 位置编码 ──
        x = x + self.pos_embed

        # ── 3. 条件向量: 时间步 + (可选) 类别 ──
        c = self.time_embed(t)                           # (B, D)
        if y is not None and self.num_classes > 0:
            c = c + self.class_embed(y)                  # 加类别信息

        # ── 4. DiT Blocks ──
        for block in self.blocks:
            x = block(x, c)

        # ── 5. Final Layer ──
        x, gate, _, _, _ = self.final_adaLN(x, c)
        x = self.final_linear(gate * x)                  # gate (B,1,D) × x (B,N,D) → Linear → (B,N,P*P*C)

        # ── 6. Unpatchify → 恢复图像形状 ──
        x = self._unpatchify(x, H, W)

        return x

    def _unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        把 patch token 序列拼回图像。

        (B, N, P*P*C) → (B, C, H, W)
        其中 N = (H/P) * (W/P), P = patch_size
        """
        B = x.shape[0]
        P = self.patch_size
        C = self.in_channels
        h_patches = H // P
        w_patches = W // P

        x = x.reshape(B, h_patches, w_patches, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)                 # (B, C, h_patches, P, w_patches, P)
        x = x.reshape(B, C, H, W)
        return x


# ============================================================================
# 5. Sinusoidal Embedding — 和 Transformer PE 一样
# ============================================================================
class SinusoidalEmbedding(nn.Module):
    """时间步的正弦嵌入"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


# ============================================================================
# 6. 测试 + 对比
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DiT (Diffusion Transformer) 演示")
    print("=" * 60)

    device = get_device()
    print(f"设备: {device}")

    # 创建 DiT
    dit = DiT(
        img_size=32,
        patch_size=4,                                    # 32/4=8, 8×8=64 patches
        in_channels=3,
        d_model=384,
        n_heads=6,
        n_layers=12,
    ).to(device)

    n_params = sum(p.numel() for p in dit.parameters())
    print(f"\nDiT 参数量: {n_params:,}")
    print(f"Patch 数: {dit.n_patches} (8×8 grid)")
    print(f"d_model: {dit.d_model}")
    print(f"层数: {len(dit.blocks)}")

    # Forward test
    B = 4
    x_t = torch.randn(B, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    y = torch.randint(0, 10, (B,), device=device)       # 10 个类别 (CIFAR-10)

    noise_pred = dit(x_t, t, y)
    print(f"\n输入 x_t: {x_t.shape}")
    print(f"时间步 t: {t}")
    print(f"类别 y:   {y}")
    print(f"输出噪声: {noise_pred.shape}")
    assert noise_pred.shape == x_t.shape, "形状不对!"
    print("✅ 形状验证通过")

    # 对比: DiT vs U-Net
    print(f"\n{'='*40}")
    print(f"DiT vs U-Net 对比")
    print(f"{'='*40}")
    print(f"""
    ┌──────────────┬─────────────────┬──────────────────┐
    │    特性       │     U-Net       │      DiT         │
    ├──────────────┼─────────────────┼──────────────────┤
    │  Backbone    │ CNN (卷积)      │ Transformer      │
    │  图像处理     │ 下采样+上采样    │ Patch Embedding  │
    │  条件注入     │ 时间嵌入拼到特征  │ AdaLN 动态调制   │
    │  Attention   │ 只在瓶颈层有     │ 每一层都有       │
    │  Scalability │ 受限 (CNN 难 scale) │ 无限 (和 LLM 一样) │
    │  适用场景     │ 中小模型, 少数据  │ 大模型, 多数据   │
    │  代表模型     │ SD 1.5/2.0      │ SD3, Sora, Flux  │
    └──────────────┴─────────────────┴──────────────────┘
    """)

    print("\n" + "=" * 60)
    print("✅ DiT 完成！")
    print("核心: 图像 patch → token → Transformer + AdaLN → unpatch")
    print("这就是 Stable Diffusion 3 / Sora / Flux 的架构")
    print("下一步: 运行 06_flow_matching.py")
    print("=" * 60)
