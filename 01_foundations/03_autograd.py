"""
Autograd — PyTorch 的自动微分引擎
===================================
Autograd 是 PyTorch 最重要的特性，没有之一。
它自动追踪所有运算，然后帮你算出梯度。
你只需要: loss.backward() — 梯度就自动算好了。

核心心智模型:
    正向: x → model(x) → loss          (你定义计算图)
    反向: loss.backward()               (PyTorch 自动算每个参数的梯度)
    更新: optimizer.step()              (用梯度更新参数)

"能自动求导" = "能训练任何神经网络" — 这才是 PyTorch 吊打 NumPy 的原因
"""

import torch
from typing import List


# ============================================================================
# 1. requires_grad: 告诉 PyTorch "我需要这个值的梯度"
# ============================================================================
print("=" * 60)
print("1. requires_grad — 开启梯度追踪")
print("=" * 60)

# 普通 tensor: 不能求导
x_normal = torch.tensor([1.0, 2.0, 3.0])
print(f"普通 tensor requires_grad: {x_normal.requires_grad}")  # False

# require_grad=True: PyTorch 会记录所有运算，以便反向求导
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"追踪梯度 tensor requires_grad: {x.requires_grad}")    # True

# grad_fn: 每个 tensor 都记得自己是"怎么来的"
y = x * 2                        # y 的 grad_fn 记录着 "这是 x 乘以 2 得到的"
z = y.sum()                      # z 的 grad_fn 记录着 "这是 y 求和得到的"
print(f"x 的 grad_fn: {x.grad_fn}")    # None (叶子节点，没有来源)
print(f"y 的 grad_fn: {y.grad_fn}")    # MulBackward0
print(f"z 的 grad_fn: {z.grad_fn}")    # SumBackward0


# ============================================================================
# 2. backward(): 自动计算梯度
# ============================================================================
print("\n" + "=" * 60)
print("2. backward() — 一键求出所有梯度")
print("=" * 60)

# 简单例子: y = x^2, 求 dy/dx
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2                       # y = x^2
y.backward()                     # 反向传播 → 计算 dy/dx
print(f"x = 3, y = x^2 = 9")
print(f"dy/dx = 2x = 6, PyTorch 算出的 x.grad = {x.grad.item()}")

# 复杂例子: 链式求导 z = (w1*x + b1)^2 + (w2*x + b2)^3
# 如果手算会很痛苦，但 backward() 一行搞定
x = torch.tensor([2.0], requires_grad=True)
w1 = torch.tensor([3.0], requires_grad=True)
b1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([0.5], requires_grad=True)
b2 = torch.tensor([0.0], requires_grad=True)

# 正向: 定义计算图
h1 = w1 * x + b1                # = 3*2 + 1 = 7
h2 = w2 * x + b2                # = 0.5*2 + 0 = 1
z = h1**2 + h2**3               # = 49 + 1 = 50

# 反向: 一行求出所有梯度
z.backward()

print(f"\nz = (w1*x + b1)^2 + (w2*x + b2)^3 = {z.item()}")
print(f"∂z/∂x  = {x.grad.item()}")          # 手算: 2*7*3 + 3*1*0.5 = 42 + 1.5 = 43.5
print(f"∂z/∂w1 = {w1.grad.item()}")          # 手算: 2*7*2 = 28
print(f"∂z/∂b1 = {b1.grad.item()}")          # 手算: 2*7*1 = 14
print(f"∂z/∂w2 = {w2.grad.item()}")          # 手算: 3*1*2 = 6
print(f"∂z/∂b2 = {b2.grad.item()}")          # 手算: 3*1*1 = 3


# ============================================================================
# 3. 梯度累积: backward() 多次调用会累加梯度
# ============================================================================
print("\n" + "=" * 60)
print("3. 梯度累积 — 最容易踩的坑")
print("=" * 60)

x = torch.tensor([2.0], requires_grad=True)

# 第一次 backward
y1 = x ** 2
y1.backward()
print(f"第一次 backward 后 x.grad: {x.grad.item()}")  # dy1/dx = 2*2 = 4

# 第二次 backward (梯度会自动累加! 不会清零!)
y2 = x ** 3
y2.backward()
print(f"第二次 backward 后 x.grad: {x.grad.item()}")  # 4 + dy2/dx = 4 + 3*4 = 16

# 清零梯度 (训练循环里必须做!)
x.grad.zero_()
print(f"zero_() 后 x.grad: {x.grad.item()}")          # 0

# ⚠️ 这就是为什么训练循环里有 optimizer.zero_grad()!
print("\n💡 为什么训练循环里必须有 optimizer.zero_grad()?")
print("  因为 backward() 是累加的，不清零会越攒越大!")


# ============================================================================
# 4. torch.no_grad(): 临时关闭梯度追踪 (推理/eval 时用)
# ============================================================================
print("\n" + "=" * 60)
print("4. torch.no_grad() — 推理时省内存")
print("=" * 60)

x = torch.tensor([3.0], requires_grad=True)

with torch.no_grad():
    y = x ** 2                   # 在 no_grad 上下文里，不追踪梯度
    print(f"no_grad 里的 requires_grad: {y.requires_grad}")  # False!

# 出了上下文恢复正常
z = x ** 3
print(f"出来后的 requires_grad: {z.requires_grad}")  # True

# 使用场景:
# 1. 推理/eval: 不训练，不需要梯度 (省内存)
# 2. 生成采样: Diffusion 的去噪步骤不需要梯度
# 3. 计算指标: accuracy, BLEU 等不需要梯度


# ============================================================================
# 5. detach(): 切断计算图
# ============================================================================
print("\n" + "=" * 60)
print("5. detach() — 切断梯度传播")
print("=" * 60)

# 场景: GAN 训练时，生成器的输出送到判别器，但更新判别器时不更新生成器
x = torch.tensor([2.0], requires_grad=True)

y = x ** 2                       # y 依赖 x
y_detached = y.detach()          # y_detached 的值和 y 一样，但不再依赖 x
print(f"y requires_grad: {y.requires_grad}")
print(f"y_detached requires_grad: {y_detached.requires_grad}")


# ============================================================================
# 6. 实战: 手写梯度 vs autograd 验证
# ============================================================================
print("\n" + "=" * 60)
print("6. 实战验证: 手算 vs Autograd")
print("=" * 60)


def manual_grad_example():
    """
    一个简单的线性回归: y_pred = wx + b, loss = (y_pred - y_true)^2
    手算梯度 vs PyTorch 自动求导，验证结果一致。
    """
    # 数据
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])    # 真实值: y = 2x

    # 参数 (需要梯度)
    w = torch.tensor([0.5], requires_grad=True)      # 初始猜 0.5
    b = torch.tensor([0.0], requires_grad=True)      # 初始猜 0.0

    # --- 正向传播 (forward) ---
    y_pred = w * x_data + b                           # [0.5, 1.0, 1.5, 2.0]
    loss = ((y_pred - y_true) ** 2).mean()             # Mean Squared Error

    # --- 反向传播 (backward) — PyTorch 自动算梯度 ---
    loss.backward()

    # --- 手算梯度做对比 ---
    # loss = mean((wx + b - y_true)^2)
    # ∂loss/∂w = mean(2 * (wx + b - y_true) * x)
    # ∂loss/∂b = mean(2 * (wx + b - y_true) * 1)
    residuals = y_pred - y_true                        # (wx+b - y)
    manual_dw = (2 * residuals * x_data).mean()        # 链式求导
    manual_db = (2 * residuals).mean()

    print(f"y_pred: {y_pred}")
    print(f"loss:   {loss.item():.4f}")
    print(f"∂loss/∂w — PyTorch: {w.grad.item():.4f}, 手算: {manual_dw.item():.4f}")
    print(f"∂loss/∂b — PyTorch: {b.grad.item():.4f}, 手算: {manual_db.item():.4f}")
    print(f"匹配: {torch.allclose(w.grad, manual_dw) and torch.allclose(b.grad, manual_db)}")


manual_grad_example()

print("\n" + "=" * 60)
print("✅ Autograd 基础完成！")
print("核心: loss.backward() 自动算出所有参数的梯度")
print("规则: 每次 backward 前必须 zero_grad()")
print("下一步: 运行 04_training_loop.py 学习训练循环")
print("=" * 60)
