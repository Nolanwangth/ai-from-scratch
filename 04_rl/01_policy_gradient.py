"""
Policy Gradient (REINFORCE) — RL 的起点
=========================================
如果你只学一个 RL 算法，学这个。

Policy Gradient 的核心洞见：
    "好的 action 增加概率，坏的 action 减少概率"
    用梯度上升直接优化策略，不需要知道环境模型。

为什么对 LLM 重要？
    RLHF (Reinforcement Learning from Human Feedback) 的核心就是 Policy Gradient。
    只不过把 "CartPole 的 reward" 换成了 "人类偏好模型的分数"。

本文件包含：
    1. 从零实现的 CartPole 环境（不依赖 gym）
    2. Policy Network（用你熟悉的 nn.Module）
    3. REINFORCE 算法
    4. 可视化训练曲线
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
# 1. CartPole 环境 — 从零实现（50 行）
# ============================================================================
class CartPoleEnv:
    """
    推车摆杆问题：
        小车上有一根杆，你要通过左右推车来保持杆子不倒。

    状态 (4 维): [小车位置, 小车速度, 杆角度, 杆角速度]
    动作 (2 个):  0=左推, 1=右推
    奖励:        每活一步 +1（目标就是活越久越好）
    终止:        杆倾斜 >12° 或 小车出界

    为什么用这个作为 RL 的入门环境？
        - 简单：4 维状态，2 个动作，够浅的网络就能学
        - 直观：你能看着杆子不倒，理解 "policy 变好了"
        - 快速：几十秒训练就能看到效果
    """

    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5          # 半杆长
        self.force_mag = 10.0
        self.tau = 0.02            # 时间步长 (模拟 dt)

        # 终止条件
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * np.pi / 180  # 12 度

        self.reset()

    def reset(self):
        """重置环境，返回初始状态"""
        self.state = np.random.uniform(low=-0.05, high=0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int):
        """
        执行一步，返回 (下一状态, 奖励, 是否终止)

        物理模型 (简化的 CartPole 动力学)：
            用力推小车 → 小车加速 → 杆跟着动 → 检查杆是否倒了
        """
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # 动力学方程 (拉格朗日力学推导的简化版本)
        total_mass = self.mass_cart + self.mass_pole
        temp = (force + self.mass_pole * self.length * theta_dot**2 * sintheta) / total_mass

        theta_acc = (self.gravity * sintheta - costheta * temp) / \
                    (self.length * (4.0/3.0 - self.mass_pole * costheta**2 / total_mass))
        x_acc = temp - self.mass_pole * self.length * theta_acc * costheta / total_mass

        # Euler 积分
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        # 终止判断
        done = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not done else 0.0       # 每活一步 +1
        return self.state.copy(), reward, done


# ============================================================================
# 2. Policy Network — 输出动作的概率分布
# ============================================================================
class PolicyNetwork(nn.Module):
    """
    策略网络：状态 → 动作概率。

    输入:  state (4 维) — 小车位置、速度、杆角度、角速度
    输出:  action probabilities (2 维) — [P(左推), P(右推)]

    为什么叫 "Policy" 而不是 "Model"？
        RL 里 model 通常指环境模型（预测环境如何变化）。
        Policy 专指 "给定状态，该做什么动作"。
        对应 LLM 里：state = prompt, action = 下一个 token。
    """

    def __init__(self, state_dim=4, hidden_dim=128, n_actions=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        # 输出不加 softmax — 让 CrossEntropyLoss 自己处理

        # 小初始化让初始策略接近均匀随机
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)                       # logits (未归一化)

    def get_action(self, state, deterministic=False):
        """
        给定状态，采样一个动作。

        deterministic=True: 取概率最大的动作 (评估时用)
        deterministic=False: 按概率随机采样 (训练时用 — 探索!)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            return probs.argmax(dim=-1)
        else:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def get_action_and_log_prob(self, state, action):
        """
        同时返回采样动作的对数概率。
        这是 Policy Gradient 训练的核心 — 你需要 log π(a|s) 来算梯度!
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled_action = dist.sample()
        log_prob = dist.log_prob(sampled_action)
        return sampled_action, log_prob


# ============================================================================
# 3. REINFORCE — 最基础的 Policy Gradient
# ============================================================================
def compute_returns(rewards, gamma=0.99):
    """
    计算折扣累积回报 G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...

    为什么需要这个？
        RL 的目标是最大化累积回报，不是单步奖励。
        比如：下围棋时，弃子（当前负奖励）可能是为了最终赢棋（巨大正回报）。

    参数：
        rewards: list[float] — 每一步的即时奖励
        gamma:   折扣因子 — 0=只看眼前, 1=眼光长远

    实现：从后往前递推（O(n)），避免重复求和
        G_T = r_T
        G_{T-1} = r_{T-1} + γ * G_T
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G                    # 从后往前累加
        returns.insert(0, G)                 # 插到最前面（因为我们是从后往前遍历）
    return returns


def train_reinforce(env, policy, optimizer, n_episodes=500, gamma=0.99):
    """
    REINFORCE 算法。

    算法（每一步 episode）：
        1. 用当前策略采样一整条轨迹 (s₀, a₀, r₀, s₁, a₁, r₁, ..., s_T)
        2. 计算每一步的回报 G_t（从后往前）
        3. 对每一步，计算 loss = -log π(a_t|s_t) * G_t
           ↑ 负号因为要做梯度上升（PyTorch 默认是做梯度下降）
           ↑ G_t 大的 action  → 梯度推它一把，增加概率
           ↑ G_t 小的 action  → 梯度拉它回来，减少概率
        4. 梯度下降（实际是上升）更新参数

    这被称为 "Monte Carlo Policy Gradient" 因为：
        - Monte Carlo: 用完整轨迹采样，不走 bootstrap（不用 value 估计）
        - Policy Gradient: 直接优化策略参数
    """
    episode_rewards = []                    # 记录每个 episode 的总奖励

    for episode in range(n_episodes):
        state = env.reset()
        log_probs = []                      # 存每一步的 log π(a_t|s_t)
        rewards = []                        # 存每一步的 r_t

        # ── 采样一条轨迹 ──
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, log_prob = policy.get_action_and_log_prob(state_tensor, None)
            # action 是随机采样的（和 policy 内部采样不同），直接用 policy 的输出
            # 更简单的做法：
            logits = policy(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # ── 计算回报 ──
        returns = compute_returns(rewards, gamma)
        returns_tensor = torch.tensor(returns)
        # 标准化回报（稳定训练的小技巧，非必须但强烈建议）
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # ── Policy Gradient Loss ──
        # 核心公式: L = - Σ log π(a_t|s_t) * G_t
        # 这个公式的数学基础：Policy Gradient Theorem
        #   ∇J(θ) = E[ Σ ∇log π(a|s) * G_t ]
        # 我们用采样来近似期望。
        policy_loss = []
        for log_prob, G in zip(log_probs, returns_tensor):
            policy_loss.append(-log_prob * G)  # 负号 = 梯度上升
        loss = torch.cat(policy_loss).sum()    # 所有时间步加起来

        # ── 更新 ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        # 每 50 个 episode 打印一次
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode+1:4d}/{n_episodes} | "
                  f"Avg Reward (最近 50): {avg_reward:6.1f} | "
                  f"Episode Length: {len(rewards):3d}")

    return episode_rewards


# ============================================================================
# 4. 主函数
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("REINFORCE Policy Gradient — CartPole")
    print("=" * 60)

    device = get_device()
    print(f"设备: {device}")

    # 创建环境 & 策略
    env = CartPoleEnv()
    policy = PolicyNetwork().to(device)

    # Adam 优化器（学习率比 SGD 高一截因为这里 reward 信号稀疏）
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    print(f"\n训练 500 个 episodes...")
    print(f"目标: 让杆子保持不倒（每个 episode 活越久越好，满分 ~200 步）\n")

    rewards = train_reinforce(env, policy, optimizer, n_episodes=500)

    # ── 最终评估 ──
    print(f"\n最终评估（不探索，只选最优动作）:")
    eval_rewards = []
    for _ in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device).to(device)
            action = policy.get_action(state_tensor, deterministic=True)
            state, reward, done = env.step(action.item())
            total_reward += reward
        eval_rewards.append(total_reward)
    print(f"  10 局平均步数: {np.mean(eval_rewards):.1f} (满分 ~200)")
    print(f"  最高: {max(eval_rewards):.0f}, 最低: {min(eval_rewards):.0f}")

    print("\n" + "=" * 60)
    print("✅ REINFORCE 完成！")
    print("核心公式: loss = -Σ log π(a|s) * G_t")
    print("直觉: 好动作（G_t 大）→ 推概率 ↑；坏动作（G_t 小）→ 拉概率 ↓")
    print()
    print("RL 和 LLM 训练的类比:")
    print("  CartPole:   state=小车状态  action=左/右推   reward=不倒")
    print("  RLHF:      state=prompt    action=下一个token  reward=人类偏好")
    print("  底层优化都是 Policy Gradient!")
    print()
    print("下一步: 运行 02_ppo.py — 学习 PPO（你 minillm 里 GRPO 的爸爸）")
    print("=" * 60)
