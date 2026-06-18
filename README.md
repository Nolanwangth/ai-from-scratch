# 🧠 AI From Scratch — 深度学习完整学习项目

> **目标**：Python 新手 → 掌握 Transformer + Diffusion + RL + Robotics + Harness 全部核心

## 环境

```bash
conda activate diffusion          # 主力环境 (torch 2.12 + diffusers + transformers)
conda activate rl-mujoco          # RL 文件 (gym + gymnasium)
```

## 项目结构 (7853 行, 29 个 .py)

```
01_foundations/         Python 基础 + PyTorch 核心
02_transformer/         Self-Attn → Cross-Attn → EncDec → GPT → ViT (全套)
03_diffusion/           DDPM → VAE → DiT → Flow Matching (3种范式)
04_rl/                  REINFORCE → PPO → RLHF (从 CartPole 到 LLM 对齐)
05_robotics/            Diffusion Policy + World Model (VLA 动作 + 世界预测)
06_harness/             Model Registry → Prompt → Agent Loop (Harness 工程)
```

## 学习路径

### 01_foundations/ — Python + PyTorch (5 files, 1359 lines)
```bash
python 01_python_basics.py     # ⭐ 完整 Python 基础 (split/set/切片/pathlib/常见报错)
python 01_python_crash.py      # 速成版 (类/函数/JSON/异常/文件IO)
python 02_tensors.py           # Tensor = NumPy + MPS GPU
python 03_autograd.py          # backward() 自动求导
python 04_training_loop.py     # 所有 AI 训练的共同骨架
```

### 02_transformer/ — 完整 Transformer 体系 (8 files, ~2500 lines)
```bash
# 核心 Attention:
python 01_multi_head_attention.py  # Self-Attention + Causal Mask
python 02_positional_encoding.py   # sin/cos 位置编码
python 06_cross_attention.py       # ⭐ Cross-Attention (VLA 多模态核心)

# 模型架构:
python 03_transformer_block.py     # Attention + FFN + LN + Residual
python 04_mini_gpt.py              # Decoder-only (GPT)
python 07_encoder_decoder.py       # ⭐ Encoder-Decoder (T5/BART 同款)
python 08_vit.py                   # ⭐ ViT (Encoder-only, 视觉编码器)

# 训练:
python 05_train_on_shakespeare.py  # 训练生成莎翁文本
```

### 03_diffusion/ — 扩散模型 (7 files, ~2700 lines)
```bash
python 01_forward_diffusion.py     # 正向加噪 + noise schedule
python 02_unet_denoiser.py         # U-Net 去噪网络
python 03_ddpm_trainer.py          # DDPM 训练 + 采样 + DDIM
python 04_train_on_cifar10.py      # CIFAR-10 真实训练
python 05_vae.py                   # ⭐ VAE 潜空间 (Latent Diffusion 基础)
python 06_dit.py                   # ⭐ DiT — Transformer 替代 U-Net
python 07_flow_matching.py         # ⭐ Flow Matching — 直线 ODE 替代 DDPM SDE
```

### 04_rl/ — 强化学习 (3 files, ~1100 lines)
```bash
python 01_policy_gradient.py       # REINFORCE + CartPole 从零
python 02_ppo.py                   # PPO + GAE + Clipped Objective
# ↑ 用: conda activate rl-mujoco          (需要 gymnasium)
python 03_rlhf_pipeline.py         # SFT→Reward Model→PPO 桥接
```

### 05_robotics/ — VLA 动作 + 世界模型 (2 files, ~350 lines)
```bash
python 01_diffusion_policy.py      # Diffusion 生成机器人动作轨迹
python 02_world_model.py           # 预测未来观测 (World Model 核心)
```

### 06_harness/ — Harness 工程 (3 files, ~1100 lines)
```bash
python 01_model_registry.py        # 注册/创建/加载模型
python 02_prompt_template.py       # Template → Call → Retry → Stream
python 03_simple_agent_loop.py     # Think → Decide → Act → Observe
```

## 关键概念映射

| 概念 | 文件 | 一句话 |
|------|------|--------|
| Self-Attention | 02_01 | 每个 token 看所有 token (Q·K^T/√d → softmax → weighted V) |
| Causal Mask | 02_01 | 上三角=True → 不能偷看未来 |
| Cross-Attention | 02_06 | Q(视觉) attend K/V(语言) → 多模态融合 |
| Encoder-Decoder | 02_07 | Encoder 双向理解 → Decoder causal 生成 |
| ViT | 02_08 | 图像→Patch→Transformer→[CLS] 特征 |
| VAE | 03_05 | 像素→潜空间 (Latent Diffusion 的基石) |
| DDPM | 03_01-04 | x_t = √ᾱ·x₀ + √(1-ᾱ)·ε → predict ε → step denoise |
| DiT | 03_06 | 用 Transformer (+ AdaLN) 替代 U-Net |
| Flow Matching | 03_07 | 直线路径: x_t=(1-t)x₀+t·ε → predict v=ε-x₀ → ODE求解 |
| PPO | 04_02 | Clipped Objective: min(ratio·A, clip(ratio)·A) |
| RLHF | 04_03 | SFT → Reward Model → PPO → Aligned LLM |
| Diffusion Policy | 05_01 | 用扩散去噪动作轨迹 (替代直接回归) |
| World Model | 05_02 | Encode→Predict next→Decode (在想象中规划) |
| Agent Loop | 06_03 | Think→Decide→Act→Observe (Claude Code 就是这样) |
| MPS | utils/ | Apple GPU 加速 (等同于 CUDA on Mac) |

## 项目哲学

- **每个 .py 文件可独立运行** — 不依赖其他文件 (除了少数有编号的)
- **中文注释解释概念，英文注释解释代码**
- **每个参数都有注释**
- **最小依赖** — torch + numpy + matplotlib
- **有兄弟项目** — [`~/Desktop/agi/minillm/`] ← 完整 pretrain→SFT→GRPO→generate
