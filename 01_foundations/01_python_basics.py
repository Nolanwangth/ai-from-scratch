"""
Python 基础完整版 — AI 工程师必知必会
======================================
覆盖你读/写 AI 代码时 95% 会遇到的 Python 语法。
每个知识点都有 AI 场景中的真实用例。

运行方式: python 01_foundations/01_python_basics.py
预计时间: 20 分钟读完 + 抄一遍
"""

# ============================================================================
# 1. 字符串 — AI 里 90% 的数据预处理就是玩字符串
# ============================================================================
print("=" * 50)
print("1. 字符串操作")
print("=" * 50)

text = "  Hello, World!  "

# 去空格 (处理原始文本必用)
print(f"strip:  '{text.strip()}'")       # 去两边空格
print(f"lstrip: '{text.lstrip()}'")      # 只去左边
print(f"rstrip: '{text.rstrip()}'")      # 只去右边

# 切分/合并 (tokenizer 的底层就是这俩)
sentence = "the cat sat on the mat"
words = sentence.split()                 # 按空格切 → list
print(f"split: {words}")
print(f"join:  {' '.join(words)}")       # 拼回去

# 查找/替换
print(f"startswith 'the': {sentence.startswith('the')}")
print(f"endswith 'mat':   {sentence.endswith('mat')}")
print(f"'cat' 在哪个位置:  {sentence.find('cat')}")    # -1 表示没找到
print(f"替换: {sentence.replace('cat', 'dog')}")

# 大小写
print(f"upper: {sentence.upper()}")
print(f"title: {sentence.title()}")      # 每个单词首字母大写

# 判断字符类型 (数据清洗必用)
print(f"isdigit('123'): {'123'.isdigit()}")
print(f"isalpha('abc'): {'abc'.isalpha()}")
print(f"isalnum('a1'):  {'a1'.isalnum()}")


# ============================================================================
# 2. list 的所有常用操作（不只是 append）
# ============================================================================
print("\n" + "=" * 50)
print("2. list 操作全集")
print("=" * 50)

tokens = ["<s>", "Hello", "world", "</s>", "padding"]

# 增删改
tokens.append("extra")                   # 末尾加
tokens.insert(0, "<BOS>")              # 任意位置插入
last = tokens.pop()                      # 弹出最后一个 (默认)
first = tokens.pop(0)                    # 弹出指定位置
tokens.remove("padding")                 # 按值删除 (只删第一个匹配的)
print(f"操作后: {tokens}")

# 合并
a = [1, 2]; b = [3, 4]
a.extend(b)                              # a 原地扩展 (不改 a 的 id)
print(f"extend: {a}")                   # [1,2,3,4]
# extend vs + : extend 原地改, + 创建新 list

# 排序 (原地 vs 返回新)
nums = [3, 1, 4, 1, 5, 9]
sorted_nums = sorted(nums)              # 返回新 list，原 list 不变
nums.sort()                              # 原地排序，原 list 被改了
print(f"sorted 返回新: {sorted_nums}")
print(f".sort() 原地:  {nums}")

# 反向
nums.reverse()                           # 原地反转
print(f"reverse: {nums}")

# 复制 — 最容易踩的坑
original = [1, 2, [3, 4]]
shallow = original.copy()               # 浅拷贝: 嵌套 list 还是共享的!
import copy
deep = copy.deepcopy(original)           # 深拷贝: 完全独立
original[2][0] = 999
print(f"浅拷贝也被改了: {shallow[2]}")    # [999, 4] ← 被牵连!
print(f"深拷贝不受影响: {deep[2]}")       # [3, 4]


# ============================================================================
# 3. dict 的高级操作（API 返回数据全是 dict 套 dict）
# ============================================================================
print("\n" + "=" * 50)
print("3. dict 操作全集")
print("=" * 50)

config = {"model": "gpt-4", "temp": 0.7, "max_tokens": 1024}

# 遍历
for key in config:                       # 默认遍历 keys
    pass
for key, value in config.items():        # ⭐ 最常用: 同时拿 key 和 value
    print(f"  {key} = {value}")

# 安全取值 vs 直接取值
print(config.get("api_key"))             # None — 没这个 key 不报错
print(config.get("api_key", "sk-xxx"))  # 给默认值
# config["api_key"]                      # ❌ KeyError! 直接挂

# 合并 (Python 3.9+)
defaults = {"temp": 0.5, "top_p": 0.9, "max_tokens": 512}
merged = {**defaults, **config}          # 后面的覆盖前面的
print(f"合并: {merged}")

# 删除
config.pop("temp")                       # 删并返回值
del config["max_tokens"]                 # 直接删


# ============================================================================
# 4. set — 去重、交集、差集
# ============================================================================
print("\n" + "=" * 50)
print("4. set 集合")
print("=" * 50)

# 场景: 找两个模型预测不同的 token
model_a_preds = {1, 2, 3, 4, 5}
model_b_preds = {4, 5, 6, 7, 8}

print(f"交集 (都预测对的): {model_a_preds & model_b_preds}")
print(f"并集:              {model_a_preds | model_b_preds}")
print(f"A 有 B 没有:        {model_a_preds - model_b_preds}")
print(f"对称差 (不同的):    {model_a_preds ^ model_b_preds}")

# 去重 — 最常用
items = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(items))
print(f"去重: {unique}")


# ============================================================================
# 5. 切片 (slicing) — [start:stop:step]
# ============================================================================
print("\n" + "=" * 50)
print("5. 切片 — Python 最屌的语法特性")
print("=" * 50)

seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f"前 3 个:    {seq[:3]}")          # [0, 1, 2]
print(f"后 3 个:    {seq[-3:]}")        # [7, 8, 9]
print(f"中间 [3:6]: {seq[3:6]}")        # [3, 4, 5]
print(f"隔一个取:   {seq[::2]}")        # [0, 2, 4, 6, 8]
print(f"倒序:      {seq[::-1]}")        # ⭐ 最常用的技巧

# 在 Tensor 上一模一样 (你已经在 02_tensors.py 看到了)
# tensor[0, :, -1]  ← 第0个batch, 所有行, 最后一列


# ============================================================================
# 6. enumerate, zip, map, filter — 函数式四件套
# ============================================================================
print("\n" + "=" * 50)
print("6. 函数式四件套")
print("=" * 50)

layers = ["attn_0", "attn_1", "ffn_0", "ffn_1"]
params = [4096, 4096, 16384, 16384]

# enumerate: 需要索引时用它
for i, name in enumerate(layers):
    print(f"  第 {i} 层: {name}")

# zip: 并行遍历多个 list
for name, n_params in zip(layers, params):
    print(f"  {name}: {n_params} 参数")

# map: 对每个元素做同样的事 (现在更多用列表推导式)
squared = list(map(lambda x: x**2, [1, 2, 3]))
print(f"map: {squared}")

# filter: 筛选
big_layers = list(filter(lambda x: x > 10000, params))
print(f"filter (>10k 参数的层): {big_layers}")

# 但列表推导式通常更可读:
big_layers_lc = [p for p in params if p > 10000]  # 和上面等价


# ============================================================================
# 7. import 机制 — 模块和包
# ============================================================================
print("\n" + "=" * 50)
print("7. import 的各种写法")
print("=" * 50)

# 你天天看到的:
import torch                             # 导入整个包
import torch.nn as nn                    # 起别名
from pathlib import Path                 # 只导入一个类
from torch import nn, optim              # 导入多个
import torch.nn.functional as F          # 子模块别名 (AI 代码标配)

# 这 3 行在 AI 代码里是固定组合:
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# 相对导入 vs 绝对导入 (写自己的库时才需要懂)
# from .utils import get_device   ← 相对导入 (在包里面用)
# from my_project.utils import get_device  ← 绝对导入


# ============================================================================
# 8. 可变 vs 不可变 — Python 最反直觉的地方
# ============================================================================
print("\n" + "=" * 50)
print("8. 可变 vs 不可变 — 面试必考, 日常必踩坑")
print("=" * 50)

# 不可变: int, float, str, tuple, bool — 传进函数不会变
def add_one(x):
    x = x + 1                             # 新创建一个 int，不影响外面的 x
    return x

a = 5
add_one(a)
print(f"int 传入函数后不变: a={a}")       # 还是 5

# 可变: list, dict, set — 传进函数，里面改了外面也改!
def append_item(lst):
    lst.append("surprise!")               # 直接改了原 list!

my_list = [1, 2, 3]
append_item(my_list)
print(f"list 传入函数后被改了: {my_list}") # [1, 2, 3, 'surprise!']

# 这就是为什么 PyTorch 的 optimizer.zero_grad() 能直接清零模型参数
# 因为参数存在 mutable 的 tensor 里


# ============================================================================
# 9. None — 空值的正确用法
# ============================================================================
print("\n" + "=" * 50)
print("9. None 的正确打开方式")
print("=" * 50)

# ❌ 错误写法
x = None
# if x == None:  # 不推荐

# ✅ 正确写法
if x is None:                             # is 检查身份，== 检查值
    print("  用 is None 而不是 == None")

# 函数默认参数别用可变对象! (经典陷阱)
# ❌ def foo(items=[]): ...   # 所有调用共享同一个 list!
# ✅ def foo(items=None):
#        if items is None:
#            items = []


# ============================================================================
# 10. Truthiness — Python 的真真假假
# ============================================================================
print("\n" + "=" * 50)
print("10. 哪些值是 False?")
print("=" * 50)

# 以下全是 False (记住这 7 个就够了):
falsy = [False, None, 0, 0.0, "", [], {}, set()]
for v in falsy:
    print(f"  bool({repr(v):12}) = {bool(v)}")

# 这意味着你可以这样写:
data = []
if not data:                              # 等价于 if len(data) == 0
    print("  数据为空!")

name = "Claude"
if name:                                  # 等价于 if name != ""
    print(f"  name 有值: {name}")


# ============================================================================
# 11. Path (pathlib) — 现代文件路径操作
# ============================================================================
print("\n" + "=" * 50)
print("11. pathlib — 别再用 os.path 了")
print("=" * 50)

from pathlib import Path

# 构建路径 (跨平台，不用管 / vs \)
model_dir = Path("/Users/nolan") / "models" / "checkpoints"
print(f"路径: {model_dir}")
print(f"父目录: {model_dir.parent}")
print(f"文件名: {model_dir.name}")
print(f"后缀: {Path('model.pt').suffix}")  # .pt

# 判断文件是否存在
print(f"存在吗: {model_dir.exists()}")

# 遍历目录下的所有 .py 文件
# for py_file in Path(".").glob("**/*.py"):  # ** = 递归
#     print(py_file)


# ============================================================================
# 12. 常见异常 + 如何看报错
# ============================================================================
print("\n" + "=" * 50)
print("12. 常见报错翻译")
print("=" * 50)

errors = {
    "NameError":   "变量没定义 —— 打错字了",
    "TypeError":   "类型错了 —— 比如把 str 传给了需要 int 的函数",
    "ValueError":  "值不合理 —— 比如 d_model=257 不能被 n_heads=8 整除",
    "KeyError":    "dict 里没有这个 key —— 用 .get() 代替 []",
    "IndexError":  "list 下标越界 —— 序列长度不够",
    "AttributeError": "对象没有这个属性 —— 记错方法名了",
    "ImportError": "找不到模块 —— 路径不对或没装包",
}
for err, trans in errors.items():
    print(f"  {err:18} = {trans}")


# ============================================================================
# 检查清单
# ============================================================================
print("\n" + "=" * 50)
print("✅ Python 基础完整版完成")
print("=" * 50)
print("""
学完这个文件 + 01_python_crash.py，你已经掌握了:

字符串处理     ✅  split, join, strip, replace, find
list 全家桶    ✅  append, extend, pop, remove, sort, reverse, copy
dict 全家桶    ✅  items, get, update, pop, 合并
set            ✅  去重, 交集, 差集
切片           ✅  [::-1] 倒序, [-3:] 后3个
enumerate/zip  ✅  遍历带索引, 并行遍历
import 各种写法 ✅  from X import Y as Z
可变vs不可变   ✅  list 传进去会改, int 不会
None           ✅  is None
Truthiness     ✅  7 个 False: False, None, 0, "", [], {}, set()
pathlib        ✅  Path / .parent / .suffix / .glob
常见报错       ✅  能看懂 traceback 了

现在去读任何 AI 项目的代码，90% 的 Python 语法难不倒你。
剩下的 10% 是: yield, decorator, async/await — 各花 5 分钟学，不急。
""")
