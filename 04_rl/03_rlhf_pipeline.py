"""
RLHF Pipeline — 从 Policy Gradient 到对齐 LLM
================================================
这是连接你学的 RL 和你做的 LLM 工作的关键文件。

你不需要跑这个文件（它没有实际训练），而是读懂它，
然后回来看懂你 minillm 项目里的 grpo.py。

RLHF 的三个步骤：
    Step 1: SFT (Supervised Fine-Tuning)
            → 用人工标注的高质量 Q&A 微调 LLM
            → 让 LLM 学会 "对话格式"

    Step 2: Reward Model 训练
            → 让人类比较两个回答 (A 更好还是 B 更好？)
            → 训练一个模型预测人类的偏好
            → 输出: 一个能打分的奖励函数 r(s, a)

    Step 3: PPO 微调
            → 用 Reward Model 当奖励源
            → Policy Gradient 更新 LLM
            → 加 KL penalty 防止偏离太远
            → 输出: 对齐后的 LLM（更安全、更有用）

对应关系（CartPole → LLM）：
    ┌──────────────┬─────────────────────┬──────────────────────┐
    │   CartPole   │       LLM           │    你的 minillm 项目   │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ state        │ prompt (对话历史)    │ [USER] + 问题文本      │
    │ action       │ 下一个 token         │ token_id              │
    │ policy π     │ LLM (输出概率)       │ MiniLLM.forward()     │
    │ reward       │ Reward Model 打分    │ reward_model.py       │
    │ V(s) (Critic)│ 价值网络             │ GRPO 不需要 (组内归一化)│
    │ episode      │ 完整回答序列         │ model.generate()      │
    └──────────────┴─────────────────────┴──────────────────────┘

GRPO 的简化（为什么 DeepSeek-R1 选它）：
    PPO 需要 4 个模型: Actor, Reference, Critic, Reward
    GRPO 只需要 2 个:   Actor, Reference (+ Reward 函数)
    → 省一半显存，训练更简单
    → 代价: 每个 prompt 要生成 N 个回答（组内比较）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. 模拟 RLHF 的 3 个步骤（概念代码，不训练）
# ============================================================================

def step_1_sft():
    """
    SFT — 监督微调（你 minillm 的 sft.py）

    目标: 让 LLM 学会 "按格式回答"
    数据: [(prompt, answer), (prompt, answer), ...]
    损失: CrossEntropyLoss(LLM(prompt), answer)

    这和你 01_foundations/04_training_loop.py 里学的训练循环
    一模一样，只不过数据从 (x, y) 变成了 (prompt, answer)。
    """
    print("""
    Step 1: SFT (Supervised Fine-Tuning)
    ─────────────────────────────────────
    代码位置: /Users/nolan/Desktop/agi/minillm/sft.py

    输入: 预训练好的 MiniLLM (只会续写，不懂对话)
    数据: [USER] 问题 [ASST] 高质量回答 [END]
          (20 条 CoT 数据在 cot_data.py)

    训练:
        loss = CrossEntropy(logits, target_tokens)
        optimizer.step()

    输出: 会按格式对话的 MiniLLM
          但 "回答质量" 还没优化 — 这是 PPO 的活
    """)


def step_2_reward_model():
    """
    Reward Model 训练（你 minillm 的 reward_model.py）

    目标: 训练一个 "打分模型"，判断回答的好坏
    数据: [(prompt, 好回答, 差回答), ...]
          人类标注者说 "A 比 B 好"
    损失: Pairwise Ranking Loss
          loss = -log(σ(r(chosen) - r(rejected)))

    为什么需要这个？
        人类不能实时给每个 token 打分。
        训练一个 Reward Model 来模拟人类的判断。
        → PPO 训练时，每个生成的 token 都能得到 "奖励信号"

    对应关系:
        CartPole: env 自动给 reward（杆倒了 = -1）
        RLHF:     Reward Model 给 reward（回答差 = 低分）
    """
    print("""
    Step 2: Reward Model 训练
    ──────────────────────────
    代码位置: /Users/nolan/Desktop/agi/minillm/reward_model.py

    架构: MiniLLM backbone + value_head (Linear(1))
          输入 [USER]问题[ASST]回答 → 输出一个标量分数

    训练数据格式 (Pairwise):
        prompt: "什么是 Transformer?"
        chosen:  "Transformer 是一种基于自注意力的..." (好回答)
        rejected: "Transformer 是变形金刚..." (差回答)

    损失: Pairwise Ranking Loss
        loss = -log(σ(score_chosen - score_rejected))
        直觉: 强行让好回答的分数比差回答高
    """)


def step_3_ppo_rlhf():
    """
    PPO 微调 LLM（你 minillm 的 grpo.py 就是这步的简化版）

    PPO 更新 LLM 的目标函数（RLHF 版本）:
        objective = E[R(s,a) - β·KL(π_new || π_ref)]
        ↑                    ↑     ↑        ↑
        最大化奖励            减去  对 KL 散度的惩罚
                                    (不让新模型偏离参考模型太远)

    为什么需要 KL penalty？
        如果只最大化奖励，LLM 会 "钻空子" (reward hacking):
        - 生成 "好好好好好好好..." 这种无意义但 reward 不低的话
        - 生成特别长的回答来凑格式分
        KL penalty 确保新模型的行为不离 SFT 模型太远

    实际操作（每一步）:
        1. 从 prompt 集合取一个 prompt
        2. 当前 LLM (π_new) 生成一个回答
        3. Reward Model 给回答打分 → r
        4. 参考 LLM (π_ref = SFT 模型) 也生成 log_probs
        5. 用 PPO clipped objective 更新 π_new
        6. 同时加 KL penalty: -β * KL(π_new || π_ref)

    对应你 02_ppo.py 的 CartPole 训练:
        CartPole:                      RLHF:
        env.step(action) → reward      RewardModel(prompt+answer) → score
        action log_prob                 token log_prob (LLM 输出)
        old_log_prob                    ref log_prob (SFT 模型的输出)
        advantage (GAE 算的)            advantage (reward + KL)
        clip_eps=0.2                   clip_eps=0.2 (一样的!)
    """
    print("""
    Step 3: PPO 微调 LLM
    ─────────────────────
    代码位置: /Users/nolan/Desktop/agi/minillm/grpo.py  (GRPO 版本)

    PPO 目标函数:
        max E[ R(prompt, answer) - β * KL(π_new || π_ref) ]
        奖励最大化 ↑                  ↑ 不要太偏离 SFT

    伪代码:
        for each prompt:
            # 1. 生成回答
            answer = llm.generate(prompt)     # 按 π_new 采样

            # 2. 打分
            reward = reward_model(prompt, answer)

            # 3. 算 KL penalty
            kl = log(π_new(answer)) - log(π_ref(answer))

            # 4. PPO 更新 (和你 CartPole 的一样!)
            advantage = reward - β * kl
            ratio = exp(new_log_prob - old_log_prob)
            loss = -min(ratio*A, clip(ratio, 0.8, 1.2)*A)
            loss.backward()

    你的 minillm 用 GRPO 代替 PPO:
        区别1: 不需要 Critic (省一半显存)
        区别2: 每个 prompt 采样 N=4 个回答，组内归一化算 advantage
        区别3: 用 rule-based reward (简单正则) 而不是 Reward Model

    GRPO 的 advantage 计算:
        scores = [r1, r2, r3, r4]                    # 4 个回答的分数
        advantages = (scores - mean(scores)) / std(scores)  # 组内归一化
        → 最好的回答有正的 advantage → 增加概率
        → 最差的有负的 advantage → 减少概率
    """)


# ============================================================================
# 2. 概念代码: RLHF 的 PPO 更新（不训练，纯教学）
# ============================================================================
def rlhf_update_example():
    """
    这是一个简化的 RLHF PPO 更新步骤的 Python 代码示例。
    不真的跑，只用来理解流程。

    和 02_ppo.py 对比着看 — 你会发现核心更新逻辑一模一样。
    """
    print("=" * 60)
    print("RLHF PPO 更新 — 概念代码")
    print("=" * 60)
    print("""
# ═══ 伪代码: RLHF 的 PPO 更新步骤 ═══

def rlhf_training_step(llm, ref_model, reward_model, prompt, optimizer):
    \"\"\"
    llm:        正在训练的模型 (π_new, the Actor)
    ref_model:  SFT 冻结的模型 (π_ref, 用来算 KL)
    reward_model: 打分模型
    prompt:     "什么是 Transformer?"
    \"\"\"

    # 1. 生成回答
    answer_tokens = llm.generate(prompt)

    # 2. Reward Model 打分
    reward = reward_model(prompt + answer_tokens)
    #   reward: 标量，比如 3.5 分

    # 3. 算 token 级别的对数概率
    new_log_probs = llm.log_probs(prompt + answer_tokens)
    old_log_probs = ref_model.log_probs(prompt + answer_tokens)
    #   new_log_probs: [log P(token1), log P(token2), ...]
    #   每个 token 有一个 log probability

    # 4. 算 KL 惩罚
    kl_penalty = new_log_probs - old_log_probs
    #   KL > 0 = 新模型在这个 token 上更 "确定" (偏离参考)

    # 5. 算 advantage
    #   最后一个 token 拿到 reward，前面的用 GAE 传回去
    advantages = compute_gae(
        rewards=[0, 0, ..., 0, reward],  # 只有最后一个 token 有奖励
        kl_penalties=kl_penalty,
        gamma=1.0,                       # 对话任务是 finite horizon
        beta=0.04,                       # KL 惩罚系数
    )

    # 6. PPO Clipped Objective (和 CartPole 一模一样!)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
    loss = -torch.min(surr1, surr2).mean()

    # 7. 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""")

    print("""
核心发现:
    上面的 PPO update 和你 CartPole 的 ppo_update() 函数
    唯一区别就是 reward 的来源不同:
        CartPole: reward = env.step(action)
        RLHF:     reward = reward_model(prompt + answer)

    其他的 — ratio, clip, advantage GAE — 完全一样!
""")


# ============================================================================
# 3. 对照表: 你的学习路径
# ============================================================================
def learning_path():
    print("=" * 60)
    print("你的完整学习路径")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────────┐
│ 理论                              代码                      │
├─────────────────────────────────────────────────────────────┤
│                                                                 │
│ Policy Gradient Theorem         01_policy_gradient.py          │
│   ∇J = E[∇log π·G]                 → REINFORCE on CartPole    │
│                                                                 │
│ PPO + GAE + Clipping             02_ppo.py                     │
│   L_clip = min(r·A, clip(r)·A)     → PPO on CartPole           │
│                                                                 │
│ RLHF Pipeline                    03_rlhf_pipeline.py (本文件)   │
│   SFT → Reward Model → PPO                                  │
│                                                                 │
│ GRPO (DeepSeek-R1)               minillm/grpo.py               │
│   group-relative advantage         → MiniLLM + GRPO             │
│                                                                 │
│ Full Stack                       minillm/ 完整项目              │
│   pretrain→SFT→GRPO→generate       → 端到端 LLM 训练            │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# 4. 下一步建议
# ============================================================================
def next_steps():
    print("=" * 60)
    print("🚀 对着 minillm 学")
    print("=" * 60)
    print("""
现在你已经理解了 Policy Gradient → PPO → RLHF 的完整链路。
回来看你的 minillm 项目，每个文件你都能看懂 "为什么这样写":

    train.py
        → 标准的 next-token prediction
        → 和你 02_transformer/05_train_on_shakespeare.py 一样

    sft.py
        → 把 next-token prediction 用在对话数据上
        → 和你 01_foundations/04_training_loop.py 一样

    reward_model.py
        → LLM backbone + Regression head
        → Pairwise ranking loss
        → 这是 RLHF Step 2 的完整实现

    grpo.py
        → PPO 的简化版（无 Critic）
        → 组内归一化代替 GAE
        → Clipped surrogate objective = 和 02_ppo.py 一样!

    每个文件你都能对号入座了!
""")

    print("=" * 60)
    print("📖 推荐阅读顺序（论文）")
    print("=" * 60)
    print("""
    1. Attention Is All You Need (Vaswani et al., 2017)
       → 你 02_transformer/ 实现的就是这篇

    2. DDPM (Ho et al., 2020)
       → 你 03_diffusion/ 实现的就是这篇

    3. PPO (Schulman et al., 2017)
       → 你 02_ppo.py 实现的就是这篇的核心

    4. InstructGPT / RLHF (Ouyang et al., 2022)
       → RLHF 的原始论文，解释了为什么 PPO 能对齐 LLM

    5. DeepSeek-R1 (DeepSeek, 2025)
       → GRPO 的出处，你 minillm/grpo.py 用的算法
    """)


if __name__ == "__main__":
    step_1_sft()
    step_2_reward_model()
    step_3_ppo_rlhf()
    rlhf_update_example()
    learning_path()
    next_steps()

    print("\n" + "=" * 60)
    print("✅ RLHF Pipeline 概念完成！")
    print()
    print("你现在掌握了:")
    print("  ✅ Python 基础 (完整版)")
    print("  ✅ Tensor + Autograd + 训练循环")
    print("  ✅ Transformer 从零实现 (GPT)")
    print("  ✅ Diffusion 从零实现 (DDPM)")
    print("  ✅ REINFORCE → PPO → GRPO → RLHF")
    print("  ✅ Agent Harness 工程")
    print()
    print("🎉 恭喜！你已经从 Python 新手成长为 AI 工程师！")
    print("=" * 60)
