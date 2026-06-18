# 运行: conda activate diffusion && python 02_transformer/09_mixture_of_experts.py
"""
Mixture of Experts (MoE) — 让 Transformer 用更少的算力拥有更多参数
================================================================
经典 MoE = 多个小网络 (Expert) + 一个路由器 (Router)

核心思想:
    普通 FFN: 每个 token 都经过同一个 FFN
    MoE FFN:  每个 token 被 Router 分配到 top-2 个 Expert，然后加权求和

    "如果一个问题太复杂，找一群专家投票，而不是让一个人硬想"

架构图:
                  输出 (B, T, d_model)
                         ↑
            ┌────────────┴────────────┐
            │   加权求和: Σ w_i · E_i │
            └────────────┬────────────┘
            ┌────────────┼────────────┐
           w_0 ↑ E_0    w_1 ↑ E_1   ...  w_7 ↑ E_7
            └────────────┼────────────┘
                         ↑
                  [Router]
                   ↑ 输入 x

为什么需要 MoE?
    1. 参数多但计算量不大: 8 个 Expert = 8× 参数，但每个 token 只激活 2 个 = 2× 计算
    2. 自动分工: 不同 Expert 学会处理不同类型的 token (元音/辅音/标点)
    3. 可扩展: Mixtral 8×7B 用这个架构，GPT-4 也是 MoE

Router 是什么?
    就是一层 Linear(d_model → n_experts, bias=False) + softmax + top-k
    通过训练，自动学会把不同的 token 分给不同的 Expert

训练时怎么知道该给哪个 Expert?
    Router 的梯度通过 gating weights 回传 → 选中的 Expert 收到梯度 → 没选中的没有
    + 额外的 aux_loss 防止所有 token 都选同一个 Expert

参考:
    Mixtral 8×7B: 8 experts, top-2 routing
    Switch Transformer: top-1 routing (更激进)
    DeepSeek MoE: fine-grained experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from pathlib import Path
import math

# import 03_transformer_block 里的 FeedForward 作为 Expert 的基础结构
sys.path.insert(0, str(Path(__file__).parent))
import importlib
_tblock = importlib.import_module("03_transformer_block")
FeedForward = _tblock.FeedForward

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mps_utils import get_device, print_device_info


# ============================================================================
# 1. Router — 给每个 token 分配 Expert
# ============================================================================

class Router(nn.Module):
    """
    Router: 决定每个 token 去哪些 Expert。

    数学上:
        scores_i = W_router · x_i           # Linear: d_model → n_experts
        probs_i  = softmax(scores_i)         # 归一化成概率
        top-k: 只保留概率最高的 k 个, 其余清零

    为什么没有 bias?
        Router 只关心 token 和 Expert 的 "相似度" (dot product)。
        bias 会引入不依赖输入的偏移，可能让某些 Expert 永远被选或永远不被选。

    为什么用 softmax 而不是 sigmoid?
        Softmax 产生竞争: 一个 Expert 概率高了，其他的自然低了。
        这是 MoE 的核心机制 — Expert 之间互相竞争 token。

    输入: (B, T, d_model)
    输出: gate = (B, T, n_experts)  — 每个 token 对每个 Expert 的权重 (非选中的为 0)
          indices = (B, T, top_k)    — 被选中的 Expert 的 ID
          probs  = (B, T, n_experts) — softmax 后的完整概率 (用于 aux loss)
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # 核心参数: 一个简单的 Linear 层
        # 训练中，每一行 W[i] 就是 Expert i 的 "偏好方向"
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        """
        # 给每个 token 打分: (B, T, d_model) → (B, T, n_experts)
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)                     # (B, T, n_experts)

        # 选 top-k 个得分最高的 Expert
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # 构造 gate: 只有被选中的 Expert 才有非零权重
        gate = torch.zeros_like(probs)
        gate.scatter_(dim=-1, index=top_k_indices, src=top_k_probs)

        return gate, top_k_indices, probs


# ============================================================================
# 2. Sparse MoE Layer — 多个 Expert + 一个 Router
# ============================================================================

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts。

    流程:
        输入 x → [Router] → 每个 token 分到 top-k 个 Expert
                          → [dispatch] → Expert_i(x) → [combine] → 输出

    核心特性:
        1. Sparse: 每个 token 只激活 top-k 个 Expert
        2. Conditional: 不同的 token 走不同的 Expert
        3. Learnable: Router 通过梯度学习分配

    参数量 vs 计算量 (假设 8 Expert, top-2):
        参数: 8 Expert × 2·d_model·ff_dim + d_model × 8(router) ≈ 8× 普通 FFN
        计算: 2 Expert × 2·d_model·ff_dim + d_model × 8(router) ≈ 2× 普通 FFN
        结论: 8× 的参数，但只有 2× 的计算量!

    这就是 MoE 的核心优势:
        "用更多参数来记忆知识，但每次推理只用其中一小部分"
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: 学分配
        self.router = Router(d_model, n_experts, top_k)

        # Expert: n_experts 个完全一样的 FFN 结构
        # 每个 Expert 都是一个 FeedForward(同 03_transformer_block.py)
        # 虽然结构一样，但训练后参数会分化 — 它们学到不同的"知识"!
        self.experts = nn.ModuleList([
            FeedForward(d_model, expansion_factor, dropout)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        输入: (B, T, d_model)
        返回: (output, aux_loss)
              output:  (B, T, d_model) — MoE 的输出
              aux_loss: scalar           — 负载均衡辅助损失
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)           # (B*T, D) — 把 batch 和序列拼平
        N = B * T                        # 总共 N 个 token

        # 1. Router: 给每个 token 分配 Expert
        gate, indices, probs = self.router(x_flat)

        # 2. 构造输出
        output = torch.zeros(N, D, device=x.device)

        # 3. Dispatch: 每个 token 去它选中的 Expert
        # 这里用 for 循环，教学清晰。
        # 生产环境会用 batched dispatch (scatter/gather) 来加速。
        for expert_idx, expert in enumerate(self.experts):
            # 哪些 token 选中了这个 Expert?
            # indices: (N, top_k), 每个位置存了 Expert 的 ID
            mask = (indices == expert_idx)                     # (N, top_k)
            if not mask.any():
                continue  # 这个 Expert 被冷落了

            # 找到具体的 token 和它在 top_k 中的位置
            token_ids, rank_ids = torch.where(mask)            # (M,), (M,)

            # 获取对应的 routing weight
            weights = gate[token_ids, rank_ids].unsqueeze(-1)  # (M, 1) — 路由权重

            # Expert 前向
            expert_out = expert(x_flat[token_ids])             # (M, D) — Expert 的输出

            # 加权求和
            output[token_ids] += weights * expert_out

        # 4. 负载均衡辅助损失
        aux_loss = self._compute_aux_loss(indices, probs)

        return output.view(B, T, D), aux_loss

    def _compute_aux_loss(self, indices: torch.Tensor, probs: torch.Tensor):
        """
        负载均衡辅助损失 (Switch Transformer 版)

        问题: 如果不用 aux loss，Router 可能会把所有 token 都给同一个 Expert
             → 其他 Expert 永远拿不到梯度 → 废了

        公式:
            L_aux = n_experts × Σ_i (f_i × P_i)

            f_i = 被分到 Expert i 的 token 比例
            P_i = 所有 token 给 Expert i 的平均路由概率

        直觉:
            - 如果 Expert i 收到了很多 token (f_i 大)，它的概率均值 P_i 也应该大
            - f_i × P_i 在均匀分配时最小
            - n_experts 乘回去让 loss 值约等于 1（方便调 aux_loss_coef）
        """
        N = indices.shape[0]    # 总共多少 token
        k = indices.shape[1]    # top-k 的值

        # f_i: 每个 Expert 收到了多少 token
        f_i = torch.zeros(self.n_experts, device=indices.device)
        for i in range(self.n_experts):
            f_i[i] = (indices == i).sum().float()
        f_i = f_i / (N * k)     # 归一化: 比例

        # P_i: 所有 token 给 Expert i 的平均概率
        P_i = probs.mean(dim=0)  # (n_experts,)

        # L_aux = n_experts × Σ(f_i × P_i)
        aux_loss = self.n_experts * (f_i * P_i).sum()

        return aux_loss


# ============================================================================
# 3. MoE Transformer Block — 把 FFN 换成 MoE
# ============================================================================

class MoETransformerBlock(nn.Module):
    """
    MoE Transformer Block。

    和 TransformerBlock (03_transformer_block.py) 的唯一区别:
        普通:  x → LN → Attention → (+) → LN → FeedForward → (+)
        MoE:   x → LN → Attention → (+) → LN → SparseMoE → (+) + aux_loss

    MoE 是 FeedForward 的"插拔替换"，不影响 Attention 和残差结构。
    """

    def __init__(self, d_model: int, n_heads: int, n_experts: int,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()

        # Attention 部分 — 和普通 TransformerBlock 一样
        sys.path.insert(0, str(Path(__file__).parent))
        attention_module = importlib.import_module("01_multi_head_attention")
        FlashMultiHeadAttention = attention_module.FlashMultiHeadAttention
        self.attention = FlashMultiHeadAttention(d_model, n_heads, dropout)

        # MoE 部分 — 替换 FeedForward
        self.moe = SparseMoE(d_model, n_experts, top_k, dropout=dropout)

        # LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        输入: (B, T, d_model)
        返回: (output, aux_loss)
        """
        # Attention sublayer (和普通 Block 一样)
        attn_out = self.attention(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)

        # MoE sublayer (替换 FeedForward)
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)

        return x, aux_loss


# ============================================================================
# 4. MoE MiniGPT — 用 MoE Block 堆叠的语言模型
# ============================================================================

class MoEConfig:
    """
    MoE MiniGPT 的配置。

    和 GPTConfig 比多了 MoE 专属参数:
        n_experts:   Expert 数量
        top_k:       每个 token 激活几个 Expert
        aux_loss_coef: 负载均衡损失的权重
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        d_model: int = 256,
        n_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.dropout = dropout

        assert d_model % n_head == 0, f"d_model ({d_model}) 必须能被 n_head ({n_head}) 整除!"

    @property
    def d_head(self):
        return self.d_model // self.n_head


class MoEMiniGPT(nn.Module):
    """
    MoE 版的 MiniGPT。

    和 MiniGPT 的区别:
        1. TransformerBlock → MoETransformerBlock
        2. forward 返回 (logits, task_loss, aux_loss) 而不是 (logits, loss)
        3. 总 loss = task_loss + aux_loss_coef × aux_loss
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Token + Position Embedding (和 MiniGPT 一样)
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # MoE Transformer Blocks
        self.blocks = nn.ModuleList([
            MoETransformerBlock(config.d_model, config.n_head,
                                config.n_experts, config.top_k, config.dropout)
            for _ in range(config.n_layer)
        ])

        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: 输入 embedding 和输出 projection 共享权重
        self.token_embed.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx: (B, T) — token IDs
        targets: (B, T) — 目标 token IDs (训练时)

        返回: (logits, task_loss, aux_loss)
        """
        B, T = idx.shape
        device = idx.device

        # Embedding
        tok_emb = self.token_embed(idx)                          # (B, T, d_model)
        pos_emb = self.pos_embed(torch.arange(T, device=device))  # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)                     # (1, 1, T, T)

        # MoE Blocks — 和 MiniGPT 一样，只是多了 aux_loss
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)                                  # (B, T, vocab)

        # Loss
        task_loss = None
        if targets is not None:
            B, T, V = logits.shape
            task_loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, task_loss, total_aux_loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0):
        """自回归生成 (和 MiniGPT.generate 一样，但忽略 aux_loss)"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        self.train()
        return idx

    @torch.no_grad()
    def generate_top_k_top_p(self, idx, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.95):
        """Top-k + Top-p 采样 (同 MiniGPT)"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        self.train()
        return idx


# ============================================================================
# 5. 路由分析 — 看 Expert 学会了什么
# ============================================================================

def analyze_routing(model: MoEMiniGPT, data: torch.Tensor, stoi: dict, itos: dict, device: torch.device):
    """
    分析 Router 学到了什么。

    跑一个完整的前向，收集每个 token 被分配给了哪些 Expert。
    然后按 token 类型统计，看看 Expert 是否"专精"于某些类型的 token。
    """
    model.eval()

    # 把中文 token 类型映射到英文类别 (for the table)
    def token_type(ch: str) -> str:
        if ch.isalpha():
            if ch.lower() in 'aeiou':
                return '元音'
            else:
                return '辅音'
        elif ch.isspace():
            return '空格'
        elif ch.isdigit():
            return '数字'
        else:
            return '标点'

    # 收集路由信息
    expert_token_counts = {}   # expert_id → {token_type: count}
    expert_token_probs = {}    # expert_id → {token: total_prob}

    # 取一个 batch 的数据
    B = 8
    ix = torch.randint(0, len(data) - model.config.block_size, (B,))
    x = torch.stack([data[i:i + model.config.block_size] for i in ix]).to(device)

    # 前向: 只取中间层的 router 信息
    # 先做 embedding
    tok_emb = model.token_embed(x)
    pos_emb = model.pos_embed(torch.arange(x.shape[1], device=device))
    h = model.drop(tok_emb + pos_emb)

    # 过第一层 block 的 MoE 看路由
    first_block = model.blocks[0]
    h = first_block.ln1(h)
    attn_out = first_block.attention(h)
    h = h + first_block.dropout(attn_out)
    h = first_block.ln2(h)

    # 获取第一个 block 的 router 输出
    gate, indices, probs = first_block.moe.router(h.view(-1, h.shape[-1]))

    # 准备分析数据结构
    expert_token_counts = {i: {} for i in range(model.config.n_experts)}

    # 遍历每个 token
    tokens = []
    for b in range(min(B, 4)):  # 只看前 4 个 batch
        for t in range(x.shape[1]):
            token_id = x[b, t].item()
            ch = itos[token_id]
            ttype = token_type(ch)

            flat_idx = b * x.shape[1] + t
            # 这个 token 被分到了哪些 Expert?
            expert_ids = indices[flat_idx].tolist()
            expert_weights = gate[flat_idx][expert_ids].tolist()

            tokens.append((ch, ttype, expert_ids, expert_weights))

            for eid, w in zip(expert_ids, expert_weights):
                if ttype not in expert_token_counts[eid]:
                    expert_token_counts[eid][ttype] = {"count": 0, "total_weight": 0.0}
                expert_token_counts[eid][ttype]["count"] += 1
                expert_token_counts[eid][ttype]["total_weight"] += w

    # ── 打印分析结果 ──
    print(f"\n{'='*60}")
    print(f"路由分析 — Expert 分工情况 (采样 {len(tokens)} 个 token)")
    print(f"{'='*60}")

    # 统计每个类型总共出现了多少次
    type_totals = {}
    for ch, ttype, *_ in tokens:
        type_totals[ttype] = type_totals.get(ttype, 0) + 1

    # 打印每个 Expert 的 token 类型分布
    print(f"\n{'类别':<8}", end="")
    for eid in range(model.config.n_experts):
        print(f"{'Expert '+str(eid):<18}", end="")
    print(f"{'总出现':<8}")

    print(f"{'-'*8}", end="")
    for _ in range(model.config.n_experts):
        print(f"{'-'*18}", end="")
    print(f"{'-'*8}")

    for ttype in ['元音', '辅音', '空格', '标点']:
        if ttype not in type_totals:
            continue
        print(f"{ttype:<8}", end="")
        total = type_totals[ttype]
        for eid in range(model.config.n_experts):
            data = expert_token_counts[eid].get(ttype, {"count": 0})
            pct = data["count"] / max(total, 1) * 100
            print(f"{pct:>5.1f}% ({data['count']:<3})    ", end="")
        print(f"{total:<8}")

    # 打印一些具体 token 的路由
    print(f"\n具体 token 路由示例:")
    for ch, ttype, eids, weights in tokens[:20]:
        route_str = ", ".join([f"Expert {eid} (w={w:.2f})" for eid, w in zip(eids, weights)])
        print(f"  '{ch}' ({ttype}) → {route_str}")

    model.train()


# ============================================================================
# 6. 训练 + 演示
# ============================================================================

def load_shakespeare() -> tuple[torch.Tensor, torch.Tensor, dict]:
    """加载 Tiny Shakespeare 数据集 (和 05_train_on_shakespeare.py 一样)"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = Path("/tmp/shakespeare.txt")

    if not data_path.exists():
        import urllib.request
        print(f"下载 Shakespeare 数据集...")
        urllib.request.urlretrieve(url, data_path)

    text = data_path.read_text(encoding="utf-8")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    tokenizer_info = {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}
    return train_data, val_data, tokenizer_info


def get_batch(data, batch_size, block_size, device):
    """随机取一个 batch"""
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    device = get_device()
    print_device_info()

    # ── 配置 ──
    batch_size = 16
    block_size = 64
    max_iters = 500
    eval_interval = 100

    # MoE 参数
    n_experts = 4
    top_k = 2
    aux_loss_coef = 0.01

    # ── 数据 ──
    train_data, val_data, tokenizer = load_shakespeare()
    vocab_size = tokenizer["vocab_size"]
    stoi = tokenizer["stoi"]
    itos = tokenizer["itos"]

    # ── 模型 ──
    config = MoEConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        d_model=128,
        n_experts=n_experts,
        top_k=top_k,
        aux_loss_coef=aux_loss_coef,
        dropout=0.1,
    )
    model = MoEMiniGPT(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    active_params_per_token = (  # 每个 token 实际参与计算的参数
        2 * config.d_model * config.d_model * 4  # 2 experts × d_model → 4d → d_model
        + config.d_model * config.n_experts       # router
    ) * config.n_layer

    print(f"\nMoE MiniGPT 配置:")
    print(f"  Vocab:   {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  层数:    {config.n_layer}")
    print(f"  Experts: {config.n_experts} (top-{config.top_k})")
    print(f"  总参数:  {total_params:,}")
    print(f"  每 token 激活参数: ~{active_params_per_token:,}")
    print(f"  (参数/激活比: {total_params / active_params_per_token:.1f}x)")
    print(f"      4 个 Expert = 4× 参数，但每个 token 只用 2 个 = 2× 计算")

    # ── 对比: 同等大小的普通 Transformer ──
    _gpt_cfg = importlib.import_module("04_mini_gpt").GPTConfig
    _MiniGPT = importlib.import_module("04_mini_gpt").MiniGPT

    standard_config = _gpt_cfg(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=config.n_layer, n_head=config.n_head,
        d_model=config.d_model, dropout=0.1,
    )
    standard_model = _MiniGPT(standard_config)
    standard_params = sum(p.numel() for p in standard_model.parameters())
    print(f"\n  同等大小标准 Transformer: {standard_params:,} 参数")
    print(f"  MoE/Standard 参数比: {total_params / standard_params:.1f}x")

    # ── 优化器 ──
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

    # ── 训练 ──
    print(f"\n开始训练 {max_iters} 步...")
    print(f"{'步数':<8} {'Task Loss':<12} {'Aux Loss':<12} {'评估':<30}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*30}")

    for step in range(max_iters):
        x, y = get_batch(train_data, batch_size, block_size, device)
        optimizer.zero_grad()

        logits, task_loss, aux_loss = model(x, y)
        total_loss = task_loss + aux_loss_coef * aux_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            model.eval()
            with torch.no_grad():
                x_eval, y_eval = get_batch(train_data, batch_size, block_size, device)
                _, eval_task, eval_aux = model(x_eval, y_eval)
            model.train()
            print(f"{step:<8} {task_loss.item():.4f}      {aux_loss.item():.4f}      "
                  f"task={eval_task.item():.3f} aux={eval_aux.item():.3f}")

    # ── 路由分析 ──
    analyze_routing(model, train_data, stoi, itos, device)

    # ── 生成 ──
    print(f"\n{'='*60}")
    print(f"生成文本 (MoE 模型)")
    print(f"{'='*60}")

    start_char = "\n"
    start_id = stoi[start_char]
    idx = torch.tensor([[start_id]], device=device)
    generated = model.generate_top_k_top_p(idx, max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.9)
    generated_text = "".join([itos[i] for i in generated[0].tolist()])
    print(generated_text)

    print(f"\n{'='*60}")
    print(f"MoE 总结")
    print(f"{'='*60}")
    print(f"""
    MoE = Mixture of Experts

    核心创新:
        用 Router 自动学习"什么 token 该去哪" — 不需要人工规则

    为什么需要 aux_loss?
        没有 aux_loss → Router 把所有 token 都发给同一个 Expert
        → 其他 Expert 白训了 → 浪费参数

    Router 到底学了什么?
        Linear 层的每一行 W[i] 代表 Expert i 的"偏好方向"
        某个 token 的 embedding 和 W[i] 的 dot product 越大
        → 它越可能被分到 Expert i

    和真实系统的对应:
        Mixtral 8×7B: 8 Expert, top-2, d_model=4096
        DeepSeek-V2:  160 Expert (fine-grained), top-6
        GPT-4:        传言为 16 Expert, top-2
        本 demo:       {config.n_experts} Expert, top-{config.top_k}, d_model={config.d_model}

    每个 Expert "专精"什么?
        训练 500 步后还不明显（需要更多训练）。
        但你能看到 Router 给不同的字符分配了不同的权重分布 —
        这是"专业化"的萌芽。
    """)


if __name__ == "__main__":
    main()
