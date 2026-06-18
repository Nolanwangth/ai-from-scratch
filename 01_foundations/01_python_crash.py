"""
Python 速成 — AI 工程师必须掌握的 20%
=========================================
"20% 的 Python 覆盖 90% 的 AI 工程场景"

本文件目标：
  读完 + 跑完 = 能看懂所有 AI 项目的 Python 代码
  每个知识点都配有 AI 场景的实例
"""

# ============================================================================
# 1. 基础类型 — 就这 4 种
# ============================================================================
print("=" * 60)
print("1. 基础类型")
print("=" * 60)

name: str = "GPT-4"          # str   字符串 → 模型名、文件路径、提示词
layers: int = 12             # int   整数   → 层数、batch_size、epoch
lr: float = 3e-4             # float 浮点   → 学习率、loss、概率
training: bool = True        # bool  布尔   → 是否是训练模式

print(f"模型: {name}, 层数: {layers}, 学习率: {lr}, 训练中: {training}")
#     ↑ f-string: 在 AI 代码里到处都是，花括号里放变量名


# ============================================================================
# 2. 数据结构 — list + dict = 90%
# ============================================================================
print("\n" + "=" * 60)
print("2. 数据结构")
print("=" * 60)

# --- list: 有序集合 ---
# 场景: 一个 batch 的训练数据、模型的所有层名
layers_list = ["embedding", "attention.0", "attention.1", "ffn", "output"]
print(f"模型层: {layers_list}")
layers_list.append("norm")                          # 追加
print(f"第 2 层: {layers_list[1]}")                 # 索引从 0 开始，不是 1!
print(f"最后 2 层: {layers_list[-2:]}")             # 负数 = 从后往前

# --- dict: 键值对 ---
# 场景: 模型配置、API 请求、训练结果 — AI 工程里 80% 的数据结构是 dict
config = {
    "model_name": "mini-gpt",
    "n_layers": 6,
    "d_model": 256,
    "n_heads": 8,
    "dropout": 0.1,
}
print(f"模型配置: {config}")
print(f"隐藏维度: {config['d_model']}")             # 方括号取值
print(f"词表大小: {config.get('vocab_size', 50257)}")  # .get() 安全取值，有默认值

# --- dict 嵌套 (API 返回的数据全这样) ---
api_response = {
    "id": "msg_abc123",
    "model": "claude-4",
    "usage": {
        "input_tokens": 150,
        "output_tokens": 80,
    },
    "content": [{"type": "text", "text": "Hello!"}],
}
# .get().get().get() 是 AI 工程师的基本功
output_tokens = api_response["usage"]["output_tokens"]
print(f"输出 token 数: {output_tokens}")


# ============================================================================
# 3. 流程控制 — if + for + while (只需要这 3 个)
# ============================================================================
print("\n" + "=" * 60)
print("3. 流程控制")
print("=" * 60)

# --- if/elif/else: 条件判断 ---
# 场景: 根据模型名决定用哪个 API
model_family = "claude"
if model_family == "claude":
    base_url = "https://api.anthropic.com"
elif model_family == "gpt":
    base_url = "https://api.openai.com"
else:
    base_url = "http://localhost:11434"  # Ollama 本地模型
print(f"API 地址: {base_url}")

# --- for: 遍历 ---
# 场景 1: 遍历训练数据
train_dataset = ["样本1", "样本2", "样本3"]
for i, sample in enumerate(train_dataset):  # enumerate 同时给索引和值
    print(f"  批次 {i}: {sample}")

# 场景 2: 列表推导式 — AI 代码里遍地都是
# 格式: [对 item 做什么 for item in 列表 if 条件]
token_ids = [1, 2, 3, 0, 4, 0, 5]
non_pad = [tid for tid in token_ids if tid != 0]  # 去掉 padding token
print(f"去掉 padding: {non_pad}")

squares = [x**2 for x in range(10)]               # 平方数
print(f"平方: {squares}")

# 场景 3: 字典推导式
# 格式: {key: value for item in 列表}
idx_to_name = {i: name for i, name in enumerate(layers_list[:3])}
print(f"索引→层名: {idx_to_name}")


# ============================================================================
# 4. 函数 — def + 类型注解
# ============================================================================
print("\n" + "=" * 60)
print("4. 函数")
print("=" * 60)


def count_tokens(text: str, tokenizer=None) -> int:
    """
    估算文本的 token 数 (一个粗糙的实现)。
    在 AI 工程里，每个函数都应该有:
    1. 类型注解 (: str, -> int)
    2. docstring (三引号里的说明)
    3. 单一职责 (只做一件事)

    Args:
        text: 输入文本
        tokenizer: tokenizer 实例 (可选)

    Returns:
        估算的 token 数量
    """
    # 简单规则: 英文约 4 字符 = 1 token，中文约 1.5 字符 = 1 token
    english_chars = sum(1 for c in text if c.isascii())
    chinese_chars = len(text) - english_chars
    return int(english_chars / 4 + chinese_chars / 1.5)


# 函数调用很简单
tokens = count_tokens("Hello, AI world! 你好世界")
print(f"估算 token 数: {tokens}")

# 函数也是对象 — 可以当参数传给别的函数（callback 模式）
def log_call(func, *args):
    """wrapper 模式: 记录函数调用"""
    print(f"  调用 {func.__name__} 参数={args}")
    return func(*args)

log_call(count_tokens, "test")


# ============================================================================
# 5. 异常处理 — try/except (API 调用必须)
# ============================================================================
print("\n" + "=" * 60)
print("5. 异常处理")
print("=" * 60)


def call_llm_api(prompt: str, max_retries: int = 3) -> dict:
    """
    模拟 LLM API 调用，展示标准的异常处理模式。
    真实场景: API 超时、限流、网络错误 — try/except 是必须的。

    Returns:
        {"content": str, "tokens": int} 或 {"error": str}
    """
    import random

    for attempt in range(max_retries):
        try:
            # 模拟 API 调用 — 真代码里这里是 requests.post()
            if random.random() < 0.5:  # 50% 概率失败(模拟不稳定网络)
                raise ConnectionError("网络超时")

            # 成功返回
            return {"content": f"回复: {prompt[:20]}...", "tokens": len(prompt)}

        except ConnectionError as e:
            print(f"  第 {attempt + 1} 次尝试失败: {e}, 重试中...")
            if attempt == max_retries - 1:
                return {"error": str(e)}  # 所有重试都失败了

    return {"error": "unknown"}


result = call_llm_api("什么是 Transformer?")
print(f"API 结果: {result}")


# ============================================================================
# 6. 文件 I/O — 读配置、写日志、存 checkpoint
# ============================================================================
print("\n" + "=" * 60)
print("6. 文件 I/O")
print("=" * 60)

import json
from pathlib import Path

# --- JSON: AI 工程的核心格式 ---
# 模型配置存 JSON，训练结果存 JSON，API 请求也 JSON
model_config = {
    "name": "mini-gpt",
    "d_model": 256,
    "n_layers": 6,
    "vocab_size": 50257,
}

# 写 JSON — with 语句自动关闭文件(不会忘)
config_path = Path("/tmp/model_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(model_config, f, indent=2, ensure_ascii=False)  # indent=2 好看

# 读 JSON
with open(config_path, "r", encoding="utf-8") as f:
    loaded = json.load(f)  # json.load 读文件, json.loads 读字符串
print(f"读回的配置: {loaded['name']}, d_model={loaded['d_model']}")

# 清理
config_path.unlink()


# ============================================================================
# 7. class — 你只需要知道一种用法
# ============================================================================
print("\n" + "=" * 60)
print("7. class")
print("=" * 60)


class ModelConfig:
    """
    PyTorch 里所有的模型都继承 nn.Module，配置用 dataclass 或普通 class。

    你不需要精通 OOP，只需要懂这 3 个概念:
    - __init__: 初始化 (构造函数)
    - self: 自己的属性 (self.xxx = yyy)
    - 继承: class Child(Parent)
    """

    def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 8):
        """__init__ 是初始化方法，self 指向实例本身"""
        self.d_model = d_model      # self.xxx = 实例属性
        self.n_layers = n_layers
        self.n_heads = n_heads
        # 验证: 如果配置不合法，早点报错
        assert d_model % n_heads == 0, f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除!"

    def get_total_params(self) -> int:
        """方法 (method): 和函数一样，但第一个参数永远是 self"""
        # 粗糙估算: 12 * d_model^2 * n_layers
        return 12 * self.d_model**2 * self.n_layers

    def __repr__(self) -> str:
        """__repr__: 打印这个对象时显示什么"""
        return f"ModelConfig(d_model={self.d_model}, layers={self.n_layers})"


# 使用
cfg = ModelConfig(d_model=512, n_layers=12)
print(f"配置: {cfg}")
print(f"估算参数量: {cfg.get_total_params():,}")

# 这行会报错 (d_model=257 不能被 n_heads=8 整除)
# cfg_bad = ModelConfig(d_model=257, n_heads=8)  # AssertionError


# ============================================================================
# 8. 你永远不会用到但必须认识的语法
# ============================================================================
print("\n" + "=" * 60)
print("8. 特殊语法速查")
print("=" * 60)

# *args: 接收任意数量的位置参数，打包成 tuple
def debug_print(*args):
    for i, arg in enumerate(args):
        print(f"  参数 {i}: {arg}")
debug_print("model", 256, True)

# **kwargs: 接收任意数量的关键字参数，打包成 dict
def build_model(**kwargs):
    print(f"  构建参数: {kwargs}")
    return kwargs
build_model(d_model=256, n_layers=6, use_flash_attn=True)

# lambda: 一行匿名函数 (在 sort/filter/map 里用)
nums = [3, 1, 4, 1, 5, 9]
sorted_nums = sorted(nums, key=lambda x: -x)  # 降序排列
print(f"  降序: {sorted_nums}")

# zip: 把多个列表"拉链"在一起
names = ["layer_0", "layer_1", "layer_2"]
params = [100, 200, 300]
for name, param in zip(names, params):
    print(f"  {name}: {param} 参数")


print("\n" + "=" * 60)
print("✅ Python 速成完成！你已经掌握了 AI 工程所需的 20% Python。")
print("下一步: 运行 02_tensors.py")
print("=" * 60)
