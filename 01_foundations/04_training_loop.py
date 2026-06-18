# 运行: conda activate diffusion && python 01_foundations/04_training_loop.py
"""
The Universal Training Loop — 所有 AI 训练都长这样
=====================================================
不管是训练 Transformer、Diffusion、CNN、RL agent，
训练循环的骨架都是这 7 行:

    for batch in dataloader:                # 1. 取数据
        optimizer.zero_grad()               # 2. 清梯度
        output = model(batch)               # 3. 前向传播
        loss = loss_fn(output, target)      # 4. 算 loss
        loss.backward()                     # 5. 反向传播
        optimizer.step()                    # 6. 更新参数
        log(loss)                           # 7. 记录

本文件会用一个完整的线性回归例子教会你这个模式。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
# 1. 数据: 造一点假数据来演示
# ============================================================================
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 造 100 个 (x, y) 点, 真实关系是 y = 3x + 2 + noise
torch.manual_seed(42)                               # 固定随机种子, 保证可复现
device = get_device()

N = 100
x_data = torch.randn(N, 1)                           # (100, 1) — 输入
y_true_weights = 3.0                                  # 真实 w
y_true_bias = 2.0                                     # 真实 b
y_data = y_true_weights * x_data + y_true_bias + torch.randn(N, 1) * 0.5  # y = 3x + 2 + noise

print(f"数据: {N} 个点, 真实关系 y = 3x + 2 + noise")
print(f"x shape: {x_data.shape}, y shape: {y_data.shape}")

# 移到 GPU
x_data = x_data.to(device)
y_data = y_data.to(device)


# ============================================================================
# 2. 模型: 最简单的线性层
# ============================================================================
print("\n" + "=" * 60)
print("2. 定义模型")
print("=" * 60)


class LinearRegression(nn.Module):
    """
    最简单的模型: y = wx + b
    nn.Module 是所有 PyTorch 模型的基类。
    你只需要实现 __init__ 和 forward。
    """

    def __init__(self):
        super().__init__()                            # 必须调用父类 __init__
        # nn.Linear(in_features=1, out_features=1) = 1个输入 → 1个输出
        # 内部自动创建了 weight 和 bias 两个参数
        self.linear = nn.Linear(1, 1)
        # 等价于:
        # self.w = nn.Parameter(torch.randn(1, 1))  ← nn.Parameter = "这个 tensor 需要梯度"
        # self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 输入 → 输出
        PyTorch 自动追踪这里的每一步运算，以便 backward() 时求梯度。
        你只需要写正向逻辑，PyTorch 负责反向。
        """
        return self.linear(x)


model = LinearRegression().to(device)
print(f"模型:\n{model}")
print(f"初始参数: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")

# 查看模型有多少参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params}, 可训练: {trainable_params}")


# ============================================================================
# 3. Loss + Optimizer: 定义 "什么是好" 和 "怎么变好"
# ============================================================================
print("\n" + "=" * 60)
print("3. Loss 函数 和 Optimizer")
print("=" * 60)

# Loss 函数: 衡量 "预测值和真实值差多远"
# MSELoss = mean((y_pred - y_true)^2)
loss_fn = nn.MSELoss()

# Optimizer: 决定 "参数怎么更新"
# SGD = w := w - lr * gradient
# Adam = 自适应学习率版本 (最常用)
optimizer = optim.SGD(model.parameters(), lr=0.01)   # 学习率 0.01

# Adam 是默认首选 (大多数情况用这个):
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Loss: MSELoss (均方误差)")
print(f"Optimizer: SGD lr=0.01")
print(f"\n参数更新公式: w = w - lr * d(loss)/dw")
print(f"               b = b - lr * d(loss)/db")


# ============================================================================
# 4. THE TRAINING LOOP — 记住这 7 行
# ============================================================================
print("\n" + "=" * 60)
print("4. 训练循环 (THE LOOP)")
print("=" * 60)

EPOCHS = 200
log_every = 40

for epoch in range(EPOCHS):
    # ┌────────────────────────────────────────────────────────┐
    # │  Step 1: 前向传播 — 模型给出预测                         │
    # └────────────────────────────────────────────────────────┘
    y_pred = model(x_data)                           # forward pass

    # ┌────────────────────────────────────────────────────────┐
    # │  Step 2: 计算 Loss — 衡量预测多差                        │
    # └────────────────────────────────────────────────────────┘
    loss = loss_fn(y_pred, y_data)

    # ┌────────────────────────────────────────────────────────┐
    # │  Step 3: 清空梯度 — 千万别忘! gradient 是累加的            │
    # └────────────────────────────────────────────────────────┘
    optimizer.zero_grad()

    # ┌────────────────────────────────────────────────────────┐
    # │  Step 4: 反向传播 — PyTorch 自动算所有参数的梯度            │
    # └────────────────────────────────────────────────────────┘
    loss.backward()

    # ┌────────────────────────────────────────────────────────┐
    # │  Step 5: 更新参数 — optimizer 用梯度更新 weight 和 bias    │
    # └────────────────────────────────────────────────────────┘
    optimizer.step()

    # ┌────────────────────────────────────────────────────────┐
    # │  Step 6: 记录日志 — 看训练有没有收敛                       │
    # └────────────────────────────────────────────────────────┘
    if epoch % log_every == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, "
              f"w={w:.4f} (真实=3.0), b={b:.4f} (真实=2.0)")


# ============================================================================
# 5. 结果验证
# ============================================================================
print("\n" + "=" * 60)
print("5. 训练结果")
print("=" * 60)

w_final = model.linear.weight.item()
b_final = model.linear.bias.item()
print(f"真实值:    w=3.0, b=2.0")
print(f"学到的值:  w={w_final:.4f}, b={b_final:.4f}")
print(f"w 误差:    {abs(w_final - 3.0):.4f}")
print(f"b 误差:    {abs(b_final - 2.0):.4f}")


# ============================================================================
# 6. 进阶: Eval 模式 + no_grad (推理的标准模式)
# ============================================================================
print("\n" + "=" * 60)
print("6. Eval 模式 (推理)")
print("=" * 60)

# model.eval() 的作用:
# - 关闭 Dropout (推理时不需要随机丢掉神经元)
# - 关闭 BatchNorm 的统计更新 (用训练时存的均值/方差)
model.eval()

# torch.no_grad() 的作用:
# - 不追踪梯度 (省内存, 加速)
with torch.no_grad():
    # 预测一些新数据
    x_test = torch.tensor([[1.0], [2.0], [10.0]], device=device)
    y_test = model(x_test)
    for i in range(len(x_test)):
        print(f"  x={x_test[i].item():.1f} → 预测 y={y_test[i].item():.3f} "
              f"(真实 y={3*x_test[i].item() + 2:.1f})")


# ============================================================================
# 7. 保存和加载模型
# ============================================================================
print("\n" + "=" * 60)
print("7. 保存 / 加载模型")
print("=" * 60)

# 保存 (state_dict 包含所有参数)
save_path = "/tmp/linear_model.pt"
torch.save(model.state_dict(), save_path)     # state_dict = 所有参数的字典
print(f"模型已保存到: {save_path}")

# 加载: 创建相同结构的模型 → load_state_dict → eval
new_model = LinearRegression().to(device)
new_model.load_state_dict(torch.load(save_path, map_location=device))
new_model.eval()
print(f"加载成功! w={new_model.linear.weight.item():.4f}, b={new_model.linear.bias.item():.4f}")


# ============================================================================
# 8. 训练循环模板 — 直接复制使用
# ============================================================================
print("\n" + "=" * 60)
print("8. 训练循环模板 (复制即用)")
print("=" * 60)

print("""
# === 训练循环模板 ===
model = YourModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # or CrossEntropyLoss, L1Loss, etc.

for epoch in range(num_epochs):
    model.train()                    # 训练模式
    for batch in dataloader:         # 遍历数据
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()        # ← 清空梯度 (必须!)
        y_pred = model(x)            # ← 前向
        loss = loss_fn(y_pred, y)    # ← 算 loss
        loss.backward()              # ← 反向
        optimizer.step()             # ← 更新

    # 每隔一段时间验证
    if epoch % 10 == 0:
        model.eval()                 # 切换到 eval 模式
        with torch.no_grad():        # 不追踪梯度
            val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "model.pt")
""")

print("=" * 60)
print("✅ 训练循环完成！")
print("核心公式: for batch → zero_grad → forward → loss → backward → optimizer.step")
print("这就是所有 AI 训练的共同骨架！")
print("=" * 60)
