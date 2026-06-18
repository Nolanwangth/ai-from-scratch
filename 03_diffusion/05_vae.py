"""
VAE (Variational Autoencoder) — 从像素到潜空间的桥梁
========================================================
这是你理解 Latent Diffusion 和 World Model 的前提。

    图像 (3×256×256) → Encoder → z (16维潜向量) → Decoder → 重建图像

为什么需要 VAE？
    1. 压缩: 3×256×256 = 196,608  →  16 维  (压缩 12,288 倍!)
    2. 连续: 潜空间是平滑的，相近的 z 生成相似的图像
    3. 可采样: 在潜空间随机采样 → 生成新图像 (generative!)

VAE 的数学:
    Encoder:  x → q(z|x) = N(μ(x), σ²(x))    输出分布参数
    Decoder:  z → p(x|z)                       从潜变量重建
    Loss:     L = ||x - x̂||² + β·KL(N(μ,σ²) || N(0,1))
               └─重建误差─┘   └──正则项: 让潜空间接近标准正态──┘

这和 Diffusion 的关系:
    Latent Diffusion = VAE 把图像压缩到潜空间 + Diffusion 在潜空间里运行
    → Stable Diffusion 就是这样: VAE(8×压缩) + UNet in latent + VAE Decoder

这和 World Model 的关系:
    VAE 编码观测 → 在潜空间里预测未来状态 → VAE 解码预测的观测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
class VAE(nn.Module):
    """最简 VAE，专为学习设计。"""

    def __init__(self, latent_dim: int = 16, img_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: image → μ, log(σ²)  ——两个头独立输出
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),    # 28→14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),              # 14→7
            nn.ReLU(),
            nn.Flatten(),                              # → 64*7*7 = 3136
        )
        self.fc_mu = nn.Linear(3136, latent_dim)        # 均值 μ
        self.fc_logvar = nn.Linear(3136, latent_dim)    # log(σ²) ——不用 σ 本身，取 log 保证正值

        # Decoder: z → image
        self.decoder_input = nn.Linear(latent_dim, 3136)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 14→28
            nn.Sigmoid(),                              # [0,1] 像素
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        ⭐ 核心技巧: Reparameterization Trick
        直接从 N(μ, σ²) 采样会切断梯度流。
        → 改成: z = μ + σ·ε,  ε ~ N(0,1)
        → 梯度能穿过 μ 和 σ 传回去!
        """
        std = torch.exp(0.5 * logvar)               # σ = e^{0.5·log(σ²)}
        eps = torch.randn_like(std)                 # ε ~ N(0,1)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device) -> torch.Tensor:
        """从潜空间随机采样生成新图像"""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(x_recon, x, mu, logvar, beta: float = 1.0) -> torch.Tensor:
    """
    VAE Loss = 重建误差 + KL 散度
    KL(N(μ,σ²)||N(0,1)) = -0.5·Σ(1 + log(σ²) - μ² - σ²)
    这个β控制了潜空间的 "规整度" (β-VAE)
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


# ============================================================================
if __name__ == "__main__":
    device = get_device()
    print(f"设备: {device}")

    vae = VAE(latent_dim=16).to(device)
    print(f"VAE 参数量: {sum(p.numel() for p in vae.parameters()):,}")

    # 测试
    x = torch.randn(8, 1, 28, 28, device=device)
    x_recon, mu, logvar = vae(x)
    print(f"输入:  {x.shape}")
    print(f"潜变量: {mu.shape}  ← {x.shape[2]*x.shape[3]} 像素 → {vae.latent_dim} 维")
    print(f"重建:  {x_recon.shape}")
    print(f"压缩比: {x.shape[2]*x.shape[3]/vae.latent_dim:.0f}x")

    loss = vae_loss(x_recon, x, mu, logvar)
    print(f"Loss: {loss.item():.1f}")
    print(f"✅ VAE OK — 这就是 Latent Diffusion 和 World Model 的基石")
