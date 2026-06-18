# 运行: conda activate diffusion && python 03_diffusion/02_unet_denoiser.py
"""
U-Net Denoiser — 噪声预测网络
================================
Diffusion 模型的训练目标: 给定 x_t (噪声图像) 和 t (时间步)，预测噪声 ε。

U-Net 架构:
    Encoder (下采样):   逐步压缩空间分辨率，提取抽象特征
    Bottleneck:         最深层的语义信息
    Decoder (上采样):   逐步恢复空间分辨率
    Skip Connections:   把 Encoder 的特征直接连到 Decoder 同层

为什么 U-Net?
    去噪任务需要:
    - 全局理解 (这个区域是天空还是草地?) → 感受野要大
    - 局部细节 (边缘清晰, 纹理正确) → 需要高分辨率特征
    U-Net 通过 Encoder-Decoder + Skip Connection 完美满足这两个需求。

时间步注入: 网络需要知道当前在哪个 t (因为不同 t 的噪声量不同)
    方法: 把 t 编码成向量，加到每一层的特征里
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. Sinusoidal Time Embedding — 把时间步 t 变成向量
# ============================================================================
class TimeEmbedding(nn.Module):
    """
    把时间步 t (整数, 如 500) 编码成一个向量。

    第 1 步: 用 sinusoidal 编码 (和 Transformer 的 Positional Encoding 一样)
    第 2 步: 过一个小的 MLP

    为什么需要这个?
        U-Net 需要知道当前噪声水平: t=10 (轻噪声) 和 t=900 (重噪声) 的处理策略完全不同
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # 小 MLP: sin/cos → 128 → 256 → 512 (最终嵌入维度 = MLP 输出的维度)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),          # 128 → 512
            nn.SiLU(),                                # ⚠️ 注意: Diffusion 用 SiLU (也叫 Swish)
            nn.Linear(d_model * 4, d_model * 4),      # 512 → 512
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) — 整数时间步 [0, T)

        Returns:
            (B, d_model*4) — 时间步嵌入向量
        """
        # Sinusoidal encoding (和 Transformer PE 一样的公式)
        half_dim = self.d_model // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half_dim - 1))
        )                                            # (half_dim,)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, 1) * (1, half_dim) = (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, d_model)

        # 通过 MLP
        return self.mlp(emb)                         # (B, d_model*4)


# ============================================================================
# 2. Residual Block — U-Net 的基本单元
# ============================================================================
class ResBlock(nn.Module):
    """
    残差块: Conv → GroupNorm → SiLU → Conv → GroupNorm → +时间嵌入 → +残差

    GroupNorm 代替 BatchNorm:
        BatchNorm 对小 batch_size 不稳定 (Diffusion 训练 batch 通常较小)
        GroupNorm 不依赖 batch，更稳定
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = 512):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # 时间嵌入投影: 把时间向量映射成两个 scale+shift
        self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)

        # 如果输入和输出通道数不同，用 1x1 conv 对齐
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W)
            t_emb: (B, time_emb_dim) — 时间嵌入

        Returns:
            (B, out_ch, H, W)
        """
        # 第一层: GroupNorm → SiLU → Conv
        h = self.conv1(F.silu(self.norm1(x)))        # SiLU(x) = x * sigmoid(x)

        # 注入时间信息: scale + shift
        # 时间嵌入 → scale (out_ch), shift (out_ch)
        t_out = self.time_proj(F.silu(t_emb))        # (B, out_ch*2)
        scale, shift = t_out.chunk(2, dim=1)         # 切成两半
        h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        #  ^ unsqueeze 两次 → (B, C, 1, 1) 方便广播

        # 第二层: GroupNorm → SiLU → Conv
        h = self.conv2(F.silu(self.norm2(h)))

        # 残差连接
        return h + self.shortcut(x)


# ============================================================================
# 3. Attention Block — 让 U-Net 也有全局感受野
# ============================================================================
class AttentionBlock(nn.Module):
    """
    在 U-Net 的瓶颈层加 Self-Attention，让网络能处理全局关系。

    为什么在 Diffusion U-Net 里加 Attention?
        U-Net 的卷积层只能看到局部 (3x3 卷积核)
        → 加 Self-Attention 让网络能看到 "远处的像素"
        → 显著提升生成质量 (尤其是大图像)
    """

    def __init__(self, channels: int):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)                            # (B, 3C, H, W)
        Q, K, V = qkv.chunk(3, dim=1)                # 各 (B, C, H, W)

        # Flatten 空间维度 → (B, C, H*W)
        Q = Q.view(B, C, -1)
        K = K.view(B, C, -1)
        V = V.view(B, C, -1)

        # Scaled Dot-Product Attention (和 Transformer 一样的!)
        scale = C ** -0.5
        attn = torch.softmax(Q.transpose(-2, -1) @ K * scale, dim=-1)  # (B, H*W, H*W)
        out = V @ attn                                # (B, C, H*W)
        out = out.view(B, C, H, W)                    # 恢复空间形状

        return x + self.out_proj(out)                 # 残差连接


# ============================================================================
# 4. U-Net — 组装所有组件
# ============================================================================
class SimpleUNet(nn.Module):
    """
    简化版 U-Net，专门用于 Diffusion 去噪。

    架构:
        Encoder:
            x → ResBlock → ResBlock → Downsample
            → ResBlock → ResBlock → Downsample
            → ResBlock + Attention → ResBlock + Attention → Downsample
        Bottleneck:
            → ResBlock + Attention → ResBlock + Attention
        Decoder:
            → Upsample → ResBlock + Attention → ResBlock + Attention
            → Upsample → ResBlock → ResBlock
            → Upsample → ResBlock → ResBlock → Conv → 输出

    下采样倍数: 8x (3 次 Downsample)
    """

    def __init__(
        self,
        in_channels: int = 3,           # 输入通道 (RGB = 3, 灰度 = 1)
        base_channels: int = 64,        # 基础通道数 (第一层的通道数)
        channel_mult: list = [1, 2, 4], # 每层通道倍数
        time_emb_dim: int = 512,        # 时间嵌入维度
    ):
        super().__init__()

        # ── 时间嵌入 ──
        self.time_emb = TimeEmbedding(d_model=128)

        # ── 输入卷积 (图像 → 基础通道) ──
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ── Encoder (下采样) ──
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels

        enc_chs = []                                    # 存下每一层的通道数 (给 skip connection 用)
        for mult in channel_mult:
            ch_out = base_channels * mult
            # 两个 ResBlock
            self.enc_blocks.append(nn.ModuleList([
                ResBlock(ch, ch_out, time_emb_dim),
                ResBlock(ch_out, ch_out, time_emb_dim),
            ]))
            enc_chs.append(ch_out)

            # Downsample (除了最后一层)
            if mult != channel_mult[-1]:
                self.downsamples.append(nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

            ch = ch_out

        # ── Bottleneck (最深, 注意力) ──
        self.bottleneck = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch),
            ResBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch),
        ])

        # ── Decoder (上采样) ──
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            ch_out = base_channels * mult
            skip_ch = enc_chs[-(i + 1)]                # 对应的 encoder 层通道数

            # 两个 ResBlock (输入 = 当前通道 + skip 通道)
            self.dec_blocks.append(nn.ModuleList([
                ResBlock(ch + skip_ch, ch_out, time_emb_dim),
                ResBlock(ch_out, ch_out, time_emb_dim),
            ]))

            # Upsample (除了最后一层)
            if i < len(channel_mult) - 1:
                self.upsamples.append(nn.ConvTranspose2d(ch_out, ch_out, kernel_size=4, stride=2, padding=1))
            else:
                self.upsamples.append(nn.Identity())

            ch = ch_out

        # ── 输出卷积 (通道 → 原图通道) ──
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — 噪声图像 x_t
            t: (B,) — 时间步

        Returns:
            (B, C, H, W) — 预测的噪声 ε_θ(x_t, t)
        """
        # 时间嵌入
        t_emb = self.time_emb(t)                     # (B, time_emb_dim)

        # 输入卷积
        h = self.input_conv(x)

        # ── Encoder ──
        skips = []                                    # 存特征给 skip connections
        for blocks, downsample in zip(self.enc_blocks, self.downsamples):
            for block in blocks:
                h = block(h, t_emb)                   # ResBlock
            skips.append(h)                           # 存下来
            h = downsample(h)                         # 下采样 (stride=2 → H,W 减半)

        # ── Bottleneck ──
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)                          # AttentionBlock

        # ── Decoder ──
        for i, (blocks, upsample) in enumerate(zip(self.dec_blocks, self.upsamples)):
            # Skip connection: concat encoder 特征
            skip = skips[-(i + 1)]                    # 取对应的 encoder 层
            h = torch.cat([h, skip], dim=1)           # 通道维度拼接

            for block in blocks:
                h = block(h, t_emb)                   # ResBlock
            h = upsample(h)                           # 上采样 (stride=2 → H,W 翻倍)

        # ── 输出 ──
        return self.output_conv(h)


# ============================================================================
# 5. 测试
# ============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.mps_utils import get_device, print_device_info

    print("=" * 60)
    print("U-Net Denoiser 演示")
    print("=" * 60)

    device = get_device()
    print_device_info()

    # 测试模型
    model = SimpleUNet(
        in_channels=3,
        base_channels=64,
        channel_mult=[1, 2, 4],                       # 通道数: 64 → 128 → 256
    ).to(device)

    # 使用 torch.compile 加速 (PyTorch 2.0+, MPS 可能不支持)
    # model = torch.compile(model)  # 留给你自己试

    B, C, H, W = 4, 3, 32, 32
    x_t = torch.randn(B, C, H, W, device=device)      # 噪声图像
    t = torch.randint(0, 1000, (B,), device=device)    # 随机时间步

    # Forward pass
    predicted_noise = model(x_t, t)

    print(f"输入 x_t:  {x_t.shape}")
    print(f"时间步 t:  {t}")
    print(f"输出噪声:  {predicted_noise.shape}")
    print(f"参数量:    {sum(p.numel() for p in model.parameters()):,}")

    # 验证: 输出形状和输入一样 (因为是预测噪声 ε)
    assert predicted_noise.shape == x_t.shape
    print(f"✅ 形状验证通过")

    # 验证: 不同 t 的输出应该不同
    out_t10 = model(x_t[:1], torch.tensor([10], device=device))
    out_t900 = model(x_t[:1], torch.tensor([900], device=device))
    print(f"t=10 vs t=900 输出相同? {torch.allclose(out_t10, out_t900, atol=0.1)} (应该 False)")

    print("\n" + "=" * 60)
    print("✅ U-Net Denoiser 完成！")
    print("架构: Encoder → Bottleneck(+Attention) → Decoder + Skip Connections")
    print("作用: 输入 (x_t, t) → 输出 预测的噪声 ε")
    print("下一步: 运行 03_ddpm_trainer.py")
    print("=" * 60)
