# 运行: conda activate diffusion && python 02_transformer/05_train_on_shakespeare.py
"""
训练 Mini-GPT 生成莎士比亚风格文本
====================================
这个脚本会:
    1. 下载一个小型文本数据集 (Tiny Shakespeare)
    2. 训练一个 Mini-GPT
    3. 生成文本让你看效果

完整展示了 AI 工程的标准流程:
    数据准备 → 模型构建 → 训练循环 → 评估 → 保存 → 推理
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
from tqdm import tqdm
import urllib.request

sys.path.insert(0, str(Path(__file__).parent))
import importlib
_minigpt = importlib.import_module("04_mini_gpt")
GPTConfig = _minigpt.GPTConfig
MiniGPT = _minigpt.MiniGPT

# 把 utils 加到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device, print_device_info


# ============================================================================
# 1. 数据准备 — TensorDataset (最简单的方式)
# ============================================================================
def load_shakespeare() -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    下载 Tiny Shakespeare 数据集 (~1MB)，一个字符级任务。

    数据集: 莎士比亚全部作品拼接成的长文本
    任务: 给定前 N 个字符，预测下一个字符 (next-character prediction)

    为什么用字符级而不是 token 级?
        字符级: 只需要 65 个类别 (字母+标点)，不需要 tokenizer
        token 级: 需要 BPE tokenizer (如 tiktoken)，增加复杂度
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = Path("/tmp/shakespeare.txt")

    if not data_path.exists():
        print(f"下载 Shakespeare 数据集...")
        urllib.request.urlretrieve(url, data_path)
        print(f"下载完成: {data_path}")

    # 读文本
    text = data_path.read_text(encoding="utf-8")
    print(f"数据集大小: {len(text):,} 字符")

    # 创建字符→ID 映射 (最简单的手工 tokenizer)
    chars = sorted(list(set(text)))                 # 所有不同的字符
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size} 个不同字符")

    # 映射表: char ↔ index
    stoi = {ch: i for i, ch in enumerate(chars)}   # string to index
    itos = {i: ch for i, ch in enumerate(chars)}   # index to string

    # encode: 把整个文本变成数字序列
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"编码后: {data.shape} (每个字符一个 int)")

    # 切分训练/验证集 (90%/10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    tokenizer_info = {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}
    return train_data, val_data, tokenizer_info


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    """
    从长文本中随机切出一个 batch。

    数据准备策略: 随机起点 + 固定长度
        GPT 的训练不需要 "标签" — 输入和目标是同一个序列!
        输入:  data[0], data[1], ..., data[N-1]
        目标:  data[1], data[2], ..., data[N]
        即: 给定前 N 个，预测后 N 个 (shifted by 1)

    为什么不需要人工标注?
        GPT 用自监督学习: 文本本身就是标签
        输入 "The cat sat" → 目标预测 "he cat sat on"
    """
    # 随机选择 batch_size 个起点
    ix = torch.randint(0, len(data) - block_size, (batch_size,))

    # 取出对应的序列
    x = torch.stack([data[i:i + block_size] for i in ix])       # (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T), 右移 1
    #  ↑ x 和 y 的区别: y 是 x 每个位置的后一个字符

    return x.to(device), y.to(device)


# ============================================================================
# 2. 训练函数 — 生产级代码
# ============================================================================
@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    """
    在训练集和验证集上评估 loss，判断过拟合与否。
    用 torch.no_grad() 包起来 — 不追踪梯度，省内存加速。
    """
    model.eval()
    losses = {"train": 0.0, "val": 0.0}

    for split, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters

    model.train()
    return losses


def train():
    """主训练函数"""
    # ── 配置 ──
    device = get_device()
    print_device_info()

    # 超参数
    batch_size = 32           # 每批训练样本数
    block_size = 128          # 序列长度 (context window)
    max_iters = 2000          # 训练步数
    eval_interval = 200       # 每隔多少步评估一次
    eval_iters = 50           # 评估时跑多少个 batch
    learning_rate = 3e-4      # 学习率
    n_layer = 4              # Transformer 层数
    n_head = 4               # Attention 头数
    d_model = 128            # 隐藏维度

    # ── 数据 ──
    train_data, val_data, tokenizer = load_shakespeare()
    vocab_size = tokenizer["vocab_size"]

    # ── 模型 ──
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        d_model=d_model,
        dropout=0.1,
    )
    model = MiniGPT(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ──
    # AdamW = Adam + Weight Decay (权重衰减，防过拟合)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # ── 学习率调度 (Cosine Annealing) ──
    # 学习率从高到低，像一个余弦曲线
    # 好处: 前期大步快走，后期小步精调
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

    # ── 训练循环 ──
    print(f"\n开始训练 {max_iters} 步...")
    train_losses = []
    val_losses = []

    pbar = tqdm(range(max_iters), desc="训练中")
    for step in pbar:
        # 如果序列太长 GPU 内存不够，每步截断 — 这里用 full sequence 就行

        # 1. 取数据
        x, y = get_batch(train_data, batch_size, block_size, device)

        # 2. 清空梯度
        optimizer.zero_grad()

        # 3. 前向传播
        _, loss = model(x, y)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # 6. 更新学习率
        scheduler.step()

        # 7. 评估
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
            train_losses.append((step, losses["train"]))
            val_losses.append((step, losses["val"]))
            pbar.set_postfix({
                "train_loss": f"{losses['train']:.3f}",
                "val_loss": f"{losses['val']:.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

    # ── 保存模型 ──
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    save_path = checkpoint_dir / "mini_gpt_shakespeare.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer": tokenizer,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, save_path)
    print(f"\n模型已保存: {save_path}")

    return model, tokenizer, save_path


# ============================================================================
# 3. 生成 — 看模型学会了什么
# ============================================================================
def generate_samples(model, tokenizer, device, n_samples=3, max_tokens=200):
    """生成文本样本，展示训练效果"""
    model.eval()
    itos = tokenizer["itos"]

    # 从训练数据中取一些起始序列，或者用换行符开头
    start_char = "\n"
    start_id = tokenizer["stoi"][start_char]

    print("\n" + "=" * 60)
    print("生成的 Shakespear 风格文本:")
    print("=" * 60)

    for i in range(n_samples):
        # 用单个字符开始 (实际的 GPT 推理会这样)
        idx = torch.tensor([[start_id]], device=device)

        # 生成 (使用 top-k + top-p 提高质量)
        generated = model.generate_top_k_top_p(
            idx,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
        )

        # Decode: 数字 → 字符
        text = "".join([itos[id] for id in generated[0].tolist()])

        print(f"\n--- 样本 {i + 1} ---")
        print(text[:500])  # 只打印前 500 个字符
        print("...")


# ============================================================================
# 4. Run!
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Mini-GPT Shakespeare 训练")
    print("=" * 60)

    # 训练
    model, tokenizer, save_path = train()

    device = get_device()
    # 生成样本
    generate_samples(model, tokenizer, device, n_samples=3, max_tokens=300)

    print("\n" + "=" * 60)
    print("✅ Transformer 全流程完成！")
    print(f"模型保存在: {save_path}")
    print("你已经从零实现了 Attention → Positional Encoding → Block → GPT → 训练 → 生成")
    print("下一步: 学习 Diffusion! 运行 03_diffusion/")
    print("=" * 60)
