# 运行: conda activate diffusion && python 01_foundations/02_tensors.py
"""
Tensor = NumPy + GPU + Autograd
=================================
PyTorch 的 Tensor 就是能跑在 GPU 上的 NumPy ndarray。
如果你会 NumPy，Tensor 只多学 3 个东西:
    1. .to("mps")   → 把数据搬到 GPU
    2. .requires_grad → 追踪梯度 (自动求导)
    3. .backward()   → 反向传播

核心心智模型:
    NumPy:  np.array  → 在 CPU 上做数学，不能求导，不能 GPU
    PyTorch: torch.Tensor → 在 CPU/GPU 上做数学，能自动求导，能 GPU 加速
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 把 utils 加到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device, print_device_info


# ============================================================================
# ※ NumPy 最低限度入门 — 先看懂 NumPy 再学 Tensor (差不多的东西)
# ============================================================================
# 关键心法: PyTorch Tensor 和 NumPy 的 API 几乎一模一样
#           区别只在 Tensor 能跑 GPU + 能自动求导
#
# NumPy 只需要记住 5 个操作:
#   np.array([1,2,3])    → 创建一个数组
#   np.zeros((2,3))      → 全零矩阵
#   np.random.randn(2,3) → 随机矩阵
#   arr.shape            → 看形状
#   arr @ arr.T          → 矩阵乘法
print("\n" + "=" * 60)
print("※ NumPy 对照 — 下面的每个 Tensor 操作都有 NumPy 版")
print("=" * 60)

# np.array — 创建数组 (和 torch.tensor 一模一样)
np_arr = np.array([1.0, 2.0, 3.0])
print(f"np.array([1,2,3]) → {np_arr}")

# np.zeros — 全零矩阵 (和 torch.zeros 一模一样)
np_zeros = np.zeros((2, 3))
print(f"np.zeros((2,3))  的形状: {np_zeros.shape}")

# np.random.randn — 随机矩阵 (和 torch.randn 一模一样)
np_rand = np.random.randn(2, 3)
print(f"np.random.randn(2,3) 的形状: {np_rand.shape}")

print("""
📌 全部对照表 (你会 NumPy = 已经会了 80% 的 Tensor):
   np.array()          ↔ torch.tensor()
   np.zeros()          ↔ torch.zeros()
   np.random.randn()   ↔ torch.randn()
   np.arange()         ↔ torch.arange()
   np.linspace()       ↔ torch.linspace()
   arr.shape           ↔ tensor.shape
   arr.reshape()       ↔ tensor.view()
   arr.T / .transpose  ↔ tensor.T / .transpose()
   np.expand_dims()    ↔ tensor.unsqueeze()
   np.squeeze()        ↔ tensor.squeeze()
   a @ b               ↔ a @ b    (矩阵乘写法一摸一样)
   区别: NumPy 只能 CPU, Tensor 可以 GPU + 自动求导
""")


# ============================================================================
# 1. 创建 Tensor — 和 NumPy 一模一样的 API
# ============================================================================
print("=" * 60)
print("1. 创建 Tensor")
print("=" * 60)
device = get_device()
print_device_info()

# 直接创建
x = torch.tensor([1.0, 2.0, 3.0])              # 从 list 创建
# 对照: np.array([1.0, 2.0, 3.0])
print(f"从 list 创建: {x}")

# 全零 / 全一 (初始化模型权重时经常用)
zeros = torch.zeros(2, 3)                       # shape (2, 3)
ones = torch.ones(2, 3)
# 对照: np.zeros((2, 3)), np.ones((2, 3))
print(f"zeros shape: {zeros.shape}")

# 随机 (初始化神经网络权重的方式)
rand = torch.randn(2, 3)                         # 标准正态分布 N(0,1)
#  ↑ randn: "random normal"，PyTorch 里最常用的随机函数
# 对照: np.random.randn(2, 3)
print(f"randn(2,3):\n{rand}")

# 递增序列 (生成 token IDs 时常用)
seq = torch.arange(0, 10)                        # [0, 1, 2, ..., 9]
# 对照: np.arange(0, 10)
print(f"arange: {seq}")

# 等间距 (noise schedule 时常用)
lin = torch.linspace(0, 1, steps=5)             # [0, 0.25, 0.5, 0.75, 1.0]
# 对照: np.linspace(0, 1, 5)
print(f"linspace: {lin}")

# eye (单位矩阵，attention mask 时用)
eye = torch.eye(4)
# 对照: np.eye(4)
print(f"eye(4) 对角线为1:\n{eye}")


# ============================================================================
# 2. Tensor 属性 — shape, dtype, device (AI 三要素)
# ============================================================================
print("\n" + "=" * 60)
print("2. Tensor 属性")
print("=" * 60)

t = torch.randn(2, 8, 256)                       # (batch, seq_len, d_model)
print(f"shape:  {t.shape}       ← (batch, seq_len, d_model) 是 AI 代码的标准注释法")
print(f"dtype:  {t.dtype}       ← float32 是默认，推理时可能用 float16")
print(f"device: {t.device}      ← CPU 上，需要 .to('mps') 搬到 GPU")
print(f"ndim:   {t.ndim}        ← 几维张量，这里 = 3")
print(f"numel:  {t.numel()}     ← 总元素数 = 2*8*256")

# shape 的命名惯例（AI 代码里你会反复看到这些字母）:
# B or N  = batch_size   (批量大小)
# T or L  = sequence_length (序列长度，如 token 数)
# C       = channels     (通道数，RGB=3)
# H, W    = height, width (图像高宽)
# D or d_model = hidden dimension (隐藏维度)


# ============================================================================
# 3. 关键操作 — reshape, transpose, squeeze, unsqueeze
# ============================================================================
print("\n" + "=" * 60)
print("3. 变形操作 (AI 代码里最常见的 bug 来源)")
print("=" * 60)

x = torch.randn(2, 16, 256)                      # (B, T, D)

# view/reshape: 改变形状（不改变数据）
# 场景: Transformer 把 (B,T,D) 拆成 (B,T,heads,d_head)
# NumPy 对照: np.reshape(x, (B, T, n_heads, d_head))
B, T, D = x.shape
n_heads = 8
d_head = D // n_heads                            # // = 整除
x_multihead = x.view(B, T, n_heads, d_head)      # (2, 16, 8, 32)
print(f"view 变形: {x.shape} → {x_multihead.shape}")

# transpose/permute: 转置/重排维度
# 场景: attention 计算 Q @ K^T 时，需要把 (B,T,D) 变成 (B,D,T)
# NumPy 对照: np.transpose(x_multihead, (0, 2, 1, 3))
x_T = x_multihead.transpose(1, 2)                # 交换维度 1 和 2
print(f"transpose 1↔2: {x_multihead.shape} → {x_T.shape}")
#    含义: 原来是 (B=2, T=16, heads=8, d_head=32)
#          transpose 后是 (B=2, heads=8, T=16, d_head=32)

# unsqueeze: 增加维度
# 场景: 单张图 (C,H,W) 要变成 batch (1,C,H,W) 才能输入模型
# NumPy 对照: np.expand_dims(img, axis=0)
img = torch.randn(3, 32, 32)                     # 单张 RGB 图
batch_img = img.unsqueeze(0)                      # 在第 0 维加一维 → (1, 3, 32, 32)
print(f"unsqueeze: {img.shape} → {batch_img.shape}")

# squeeze: 删除大小为 1 的维度
# NumPy 对照: np.squeeze(batch_img, axis=0)
back = batch_img.squeeze(0)
print(f"squeeze:   {batch_img.shape} → {back.shape}")


# ============================================================================
# 4. 运算 — 全是 element-wise + broadcast
# ============================================================================
print("\n" + "=" * 60)
print("4. 运算")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 逐元素运算 (element-wise) — 每个位置的数各自算
# NumPy 对照: a + b, a * b (和 torch 写法一摸一样)
print(f"a+b:  {a + b}")
print(f"a*b:  {a * b}")                         # ⚠️ 这是逐元素乘，不是矩阵乘！
print(f"a**2: {a ** 2}")

# 矩阵乘法 — @ (Python 3.5+ 的矩阵乘运算符)
# 场景: Q @ K^T 就是矩阵乘，是整个 Transformer 的核心运算
# NumPy 对照: q @ k.T (和 torch 写法一摸一样!)
q = torch.randn(4, 64)                          # (T, d_head)
k = torch.randn(4, 64)                          # (T, d_head)
attn_scores = q @ k.T                           # (4, 64) @ (64, 4) = (4, 4)
print(f"Q @ K^T shape: {attn_scores.shape}")    # 4 个 token 互相看 → 4x4 矩阵

# torch.matmul 和 @ 等价 (但 @ 更简洁)
same = torch.matmul(q, k.T)
print(f"matmul == @: {torch.allclose(attn_scores, same)}")

# Broadcasting — NumPy/PyTorch 最吊的特性
# 规则: 从最后一维开始对齐，维度=1 的会自动扩展
# 场景: 加 bias 向量到每个 token (不用写 for 循环!)
# NumPy 对照: 和 numpy 的 broadcasting 规则完全一样
x = torch.randn(2, 4, 256)                      # (B, T, D)
bias = torch.randn(256)                          # (D,) → 自动 broadcast 到 (1, 1, 256) → (2, 4, 256)
y = x + bias                                     # PyTorch 自动帮你复制，不用 for 循环
print(f"Broadcast: {x.shape} + {bias.shape} = {y.shape}")


# ============================================================================
# 5. GPU/MPS: 把数据搬上/搬下 GPU
# ============================================================================
print("\n" + "=" * 60)
print("5. MPS (Apple GPU) 加速")
print("=" * 60)

# 创建 tensor 在指定设备上
x_cpu = torch.randn(1000, 1000)
x_mps = torch.randn(1000, 1000, device=device)   # 直接在 MPS 上创建
print(f"CPU tensor device: {x_cpu.device}")
print(f"MPS tensor device: {x_mps.device}")

# .to() 是最常用的设备转换方式
x_mps_from_cpu = x_cpu.to(device)                # CPU → MPS
x_cpu_from_mps = x_mps.cpu()                     # MPS → CPU (用于保存/打印/numpy转换)

# 速度对比: 矩阵乘法
import time

# MPS
if torch.backends.mps.is_available():
    torch.mps.synchronize()
    start = time.time()
    _ = x_mps @ x_mps.T
    torch.mps.synchronize()
    mps_time = time.time() - start
    print(f"⚡ MPS 矩阵乘 1000x1000: {mps_time:.4f}s")

# CPU
start = time.time()
_ = x_cpu @ x_cpu.T
cpu_time = time.time() - start
print(f"🐢 CPU 矩阵乘 1000x1000: {cpu_time:.4f}s")

if torch.backends.mps.is_available():
    print(f"🚀 加速比: {cpu_time / mps_time:.1f}x")


# ============================================================================
# 6. 索引和切片 — 和 NumPy / Python list 一样
# ============================================================================
print("\n" + "=" * 60)
print("6. 索引和切片")
print("=" * 60)

x = torch.arange(24).reshape(2, 3, 4)           # (2, 3, 4) 的 tensor
print(f"原始 shape: {x.shape}")

# 取单个元素
print(f"x[0,0,0]: {x[0,0,0]}")                 # 第 0 个 batch、第 0 行、第 0 列

# 切片: start:end:step (和 Python list / NumPy 完全一样)
print(f"x[0]:     shape={x[0].shape}")         # 第 0 个 batch → (3,4)
print(f"x[0, :2]: shape={x[0, :2].shape}")    # 第 0 个 batch 的前 2 行 → (2,4)
print(f"x[..., -1]: shape={x[..., -1].shape}") # ... = 前面全部维度, -1 = 最后一维最后一个

# bool mask (AI 里用于 mask padding tokens)
# NumPy 对照: scores[scores > 0.3] (写法一摸一样)
scores = torch.tensor([0.5, 0.1, 0.9, 0.02])
mask = scores > 0.3                              # tensor([True, False, True, False])
filtered = scores[mask]                           # 只保留 True 的
print(f"Mask: {mask}, filtered: {filtered}")


# ============================================================================
# 7. Tensor ↔ NumPy 互转
# ============================================================================
print("\n" + "=" * 60)
print("7. Tensor ↔ NumPy")
print("=" * 60)

# Tensor → NumPy (注意: 必须是 CPU tensor!)
x = torch.randn(5)
x_np = x.numpy()                                  # tensor → numpy (共享内存!)
print(f"Tensor → NumPy: {type(x_np)}")

# NumPy → Tensor
arr = np.array([1.0, 2.0, 3.0])
x2 = torch.from_numpy(arr)                        # numpy → tensor (共享内存!)
print(f"NumPy → Tensor: {type(x2)}")


print("\n" + "=" * 60)
print("✅ Tensor 基础完成！")
print("核心: Tensor = NumPy 的 API + .to('mps') + .backward()")
print("下一步: 运行 03_autograd.py 学习自动求导")
print("=" * 60)
