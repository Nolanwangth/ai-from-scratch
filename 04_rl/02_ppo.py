# 运行: conda activate diffusion && python 04_rl/02_ppo.py
"""
PPO (Proximal Policy Optimization) — RL 的工业标准
=====================================================
PPO 是 OpenAI 2017 年提出的算法，现在是 RL 的默认选择。
GPT-4、Claude 的 RLHF 训练都用 PPO（或其变体 GRPO）。

为什么 REINFORCE 不够好？
    - 用完一条轨迹就丢了，数据利用率极低
    - 梯度方差大（G_t 可能从 -100 到 +100），训练不稳定
    - 没有 "trust region" — 一步更新太大可能导致策略崩溃

PPO 的 3 个核心创新：
    1. Clipped Surrogate Objective — 限制每次更新的幅度
       L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
       如果新旧策略差异太大 → 截断 → 防止 "一步迈太大摔死"

    2. GAE (Generalized Advantage Estimation) — 平衡偏差和方差
       A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
       其中 δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD error)

    3. Multiple Epochs — 同一条数据复用多次
       收集一批轨迹 → 在这批数据上更新好几轮
       → 数据利用率远超 REINFORCE

PPO vs GRPO (你 minillm 里用的)：
    PPO:  需要 Critic (Value Network) 估计 V(s)，用 GAE 算 advantage
    GRPO: 不需要 Critic，对同一个 prompt 采样 N 个回答，组内归一化算 advantage
    → GRPO 更简单（少一个网络），但需要更多采样（每个 prompt 生成 N 次）

用法：
    conda activate rl-mujoco
    python 05_rl/02_ppo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device


# ============================================================================
# 1. Actor-Critic 网络 — 同时输出 策略 和 价值
# ============================================================================
class ActorCritic(nn.Module):
    """
    Actor-Critic 架构：
        Actor (策略头):   状态 → 每个动作的概率
        Critic (价值头):  状态 → V(s) 标量（当前状态有多好）

    共享 backbone（前几层），两个 head 独立输出。
    共享 = 更快、更少参数、特征表示更通用。

    这对应 LLM RLHF 里：
        Actor   = 正在训练的 LLM（输出 token 概率）
        Critic  = 价值网络（判断 "目前这个回答有多好"）
    """

    def __init__(self, state_dim=4, hidden_dim=128, n_actions=2):
        super().__init__()

        # 共享 backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),                              # Tanh 比 ReLU 在 RL 里更稳（有界输出）
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: 输出每个动作的 logit
        self.actor = nn.Linear(hidden_dim, n_actions)

        # Critic head: 输出状态价值 V(s) — 一个标量
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """正交初始化 — RL 训练更稳定"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """返回 (action logits, state value)"""
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, state, deterministic=False):
        """
        采样动作。返回 (action, log_prob, value)。

        这是 PPO 收集轨迹时调的 — 一步拿到所有需要的东西。
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value, dist.entropy()


# ============================================================================
# 2. GAE — 计算优势函数
# ============================================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    GAE (Generalized Advantage Estimation)

    TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
              "实际拿到的奖励 + 对未来的估计 - 之前对现在的估计"
              > 0 = 实际比预期好（惊喜）
              < 0 = 实际比预期差（失望）

    Advantage: A_t = δ_t + γλ * δ_{t+1} + (γλ)² * δ_{t+2} + ...
               = "这个动作到底有多好，考虑了多步未来的影响"

    λ 的作用 (0 ≤ λ ≤ 1)：
        λ = 0: A_t = δ_t  (只看一步，低方差但高偏差)
        λ = 1: A_t = TD(∞) = Monte Carlo return (无偏差但高方差)
        λ = 0.95: 平衡（几乎所有 RL 实现都选这个值）

    返回：
        advantages: 每个时间步的优势值 (用于更新 Actor)
        returns:    每个时间步的回报     (用于更新 Critic)
    """
    advantages = []
    gae = 0.0

    # 从后往前算（和 compute_returns 一样的递推思路）
    for i in reversed(range(len(rewards))):
        if dones[i]:
            # 这一步是终止步 → 没有 V(s_{t+1})
            next_value = 0.0
        else:
            next_value = values[i + 1] if i + 1 < len(values) else 0.0

        delta = rewards[i] + gamma * next_value - values[i]   # TD error
        gae = delta + gamma * lam * gae  # 关键递推！
        #     ↑ 当前 TD error + γλ × 上一步已经算好的 "累积优势"
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages)
    returns = advantages + torch.tensor(values[:-1])
    # ↑ advantage = return - baseline, 所以 return = advantage + V(s)

    # 标准化优势（稳定训练的关键技巧）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ============================================================================
# 3. PPO 更新 — Clipped Objective
# ============================================================================
def ppo_update(ac, optimizer, states, actions, old_log_probs, advantages, returns,
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, n_epochs=10):
    """
    PPO 的核心更新逻辑。

    每次更新循环：
        1. 拿到当前策略对老动作的 log_prob 和 value
        2. 算 ratio = exp(new_log_prob - old_log_prob)
        3. Clipped objective:
           L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        4. 加上 Value loss 和 Entropy bonus
        5. SGD 更新

    为什么 clipped？
        设 ε = 0.2，A_t > 0（这个动作好）：
            old: π_old(a|s) = 0.3, new: π_new(a|s) = 0.5
            → ratio = 0.5/0.3 = 1.67, clip 到 1.2
            → 不让优势动作的概率增加超过 20%
            → 防止过拟合一条轨迹

        设 A_t < 0（这个动作差）：
            old: π_old(a|s) = 0.3, new: π_new(a|s) = 0.1
            → ratio = 0.1/0.3 = 0.33, clip 到 0.8
            → 不让劣势动作的概率减少超过 20%
            → 防止策略剧变
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0

    for _ in range(n_epochs):
        # 当前策略对这批老数据的评估
        logits, values = ac(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # ── 1. Policy Loss (Clipped Surrogate) ──
        ratio = torch.exp(new_log_probs - old_log_probs)
        #  为什么用 exp？因为 log(a/b) = log(a) - log(b)，取 exp 回到比值

        surr1 = ratio * advantages                    # 正常的目标
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages  # 截断版本
        policy_loss = -torch.min(surr1, surr2).mean() # 取 min 做保守估计
        #  为什么取 min？因为我们想做梯度上升（最大化优势），
        #  取 min 确保 "该推的时候不过推，该拉的时候不过拉"

        # ── 2. Value Loss (MSE) ──
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        #  这就是让 Critic 学会准确预测 V(s)

        # ── 3. Entropy Bonus ──
        #  鼓励策略保持一定的随机性（探索），不要过早收敛到一个确定性策略
        #  在 LLM RLHF 里 → 相当于控制 "模型的回答不要太千篇一律"

        # ── 总 Loss ──
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（防止单步梯度过大）
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=0.5)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / n_epochs, total_value_loss / n_epochs


# ============================================================================
# 4. 轨迹收集 — Rollout
# ============================================================================
def collect_trajectories(env, ac, n_steps, device):
    """
    用当前策略采集 n_steps 步数据。
    返回所有需要的东西作为 tensors。

    PPO 是 on-policy 算法：每次数据只能用当前策略采集的。
    所以必须：采集 → 更新 → 丢弃 → 再采集 → 再更新 → ...
    """
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    state, _ = env.reset()
    episode_reward = 0.0
    episode_rewards = []

    for _ in range(n_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, value, _ = ac.get_action(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
        values.append(value.item())

        episode_reward += reward
        state = next_state

        if done:
            # 记录完成的 episode 奖励
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()

    # 最后还要算一下最后一个状态的 value（给 GAE 用的 V(s_{T+1})）
    with torch.no_grad():
        final_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        _, final_value = ac(final_state)
        values.append(final_value.item())

    # 转成 tensor
    return (
        torch.FloatTensor(np.array(states)).to(device),
        torch.LongTensor(actions).to(device),
        rewards,
        dones,
        torch.FloatTensor(log_probs).to(device),
        values,                                     # 多最后一项 (V(s_{T+1}))
        episode_rewards,
    )


# ============================================================================
# 5. 主训练循环
# ============================================================================
def train_ppo(env_name="CartPole-v1", total_steps=50000, n_steps_per_update=512):
    """
    PPO 完整训练。

    参数：
        total_steps:        总共和环境交互多少步
        n_steps_per_update: 每次更新前收集多少步数据
                            (大 = 数据多但慢, 小 = 快但不稳)

    CartPole 在 ~20,000 步就能学到完美策略（活满 500 步）。
    """
    device = get_device()
    print(f"设备: {device}")

    # ── 环境 ──
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"环境: {env_name} | 状态维度: {state_dim} | 动作数: {n_actions}")
    print(f"目标: CartPole 立满 500 步 (= 满分)")

    # ── 模型 ──
    ac = ActorCritic(state_dim, hidden_dim=128, n_actions=n_actions).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=3e-4)

    # ── 统计 ──
    all_episode_rewards = []
    recent_rewards = deque(maxlen=100)             # 最近 100 个 episode 的平均
    step_count = 0
    update_count = 0

    print(f"\n训练 {total_steps} 步...")
    print(f"每次更新前收集 {n_steps_per_update} 步数据\n")

    while step_count < total_steps:
        # ── Step 1: 收集轨迹 ──
        states, actions, rewards, dones, old_log_probs, values, ep_rewards = \
            collect_trajectories(env, ac, n_steps_per_update, device)

        step_count += len(rewards)
        all_episode_rewards.extend(ep_rewards)
        recent_rewards.extend(ep_rewards)

        # ── Step 2: 计算 GAE ──
        advantages, returns = compute_gae(rewards, values, dones)

        # ── Step 3: PPO 更新（同一批数据更新多轮） ──
        p_loss, v_loss = ppo_update(
            ac, optimizer,
            states, actions, old_log_probs,
            advantages.to(device), returns.to(device),
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            n_epochs=4,                              # 同一批数据更新 4 轮
        )

        update_count += 1

        # 打印
        if update_count % 10 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"  Step {step_count:6d}/{total_steps} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"P_Loss: {p_loss:.4f} | V_Loss: {v_loss:.4f} | "
                  f"最近 Episodes: {len(recent_rewards)}")

    env.close()

    # ── 最终评估 ──
    print(f"\n最终评估:")
    eval_env = gym.make(env_name, render_mode=None)
    eval_rewards = []
    for _ in range(20):
        state, _ = eval_env.reset()
        total_r = 0
        done = False
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _, _ = ac.get_action(s, deterministic=True)
            state, reward, terminated, truncated, _ = eval_env.step(action.item())
            total_r += reward
            done = terminated or truncated
        eval_rewards.append(total_r)
    eval_env.close()

    avg_eval = np.mean(eval_rewards)
    print(f"  20 局平均: {avg_eval:.1f} 步 (满分 500)")
    print(f"  最高: {max(eval_rewards)}, 最低: {min(eval_rewards)}")
    solved = avg_eval >= 450
    print(f"  {'✅ 已解决!' if solved else '⚠️ 还需训练'}")
    print("  (CartPole 解决标准: 连续 100 局平均 >= 475, 或单局 500 满分)")

    return ac, all_episode_rewards


# ============================================================================
# 6. Run
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PPO — Proximal Policy Optimization")
    print("=" * 60)

    ac, rewards = train_ppo(
        env_name="CartPole-v1",
        total_steps=30000,                           # 3 万步约 1-2 分钟
        n_steps_per_update=512,
    )

    print("\n" + "=" * 60)
    print("✅ PPO 完成！")
    print()
    print("你学到的 PPO 核心:")
    print("  1. Clipped Surrogate: min(ratio*A, clip(ratio)*A)")
    print("     → 限制每次更新幅度，防止策略崩溃")
    print("  2. GAE: A_t = δ_t + γλ·δ_{t+1} + ...")
    print("     → 平衡偏差和方差，稳定的优势估计")
    print("  3. Multiple Epochs: 同一批数据循环更新")
    print("     → 数据利用率远超 REINFORCE")
    print()
    print("PPO → LLM RLHF 的映射:")
    print("  CartPole state → LLM prompt (对话历史)")
    print("  CartPole action → LLM 下一个 token")
    print("  CartPole reward → 人类偏好模型 (Reward Model) 打分")
    print("  Actor-Critic  → Actor=正在训练的LLM, Critic=价值网络")
    print("  Clip ε=0.2    → 限制 LLM 每次更新不要太激进")
    print("  KL penalty     → 额外约束：新 LLM 不要偏离 SFT 模型太远")
    print()
    print(f"你的 minillm 项目的 grpo.py 就是 PPO 的简化版!")
    print(f"  → /Users/nolan/Desktop/agi/minillm/grpo.py")
    print(f"  → GRPO 去掉了 Critic，用组内归一化代替 GAE")
    print()
    print("下一步: 运行 03_rlhf_pipeline.py — 看完整 LLM 对齐流程")
    print("=" * 60)
