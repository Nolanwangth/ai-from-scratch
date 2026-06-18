# 运行: conda activate diffusion && python 06_harness/01_model_registry.py
"""
Model Registry — 生产级模型管理
==================================
Harness 工程师的核心工作: 管理多个模型，让它们可以被统一调用。

Model Registry 是 AI 基础设施的基石:
    - 注册模型: "这个名字对应这个模型类"
    - 创建模型: "给我一个 'mini-gpt' 模型"
    - 加载/保存: checkpoint 管理
    - 设备管理: 自动放到 GPU 上

这就是 Claude Code、LangChain 等框架底层做的事情。
"""

import torch
import torch.nn as nn
from typing import Dict, Type, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# 1. Config Dataclass — 类型安全的配置
# ============================================================================
@dataclass
class ModelConfig:
    """
    Python dataclass: 自动生成 __init__, __repr__, __eq__ 等方法。
    比普通 dict 好在:
        - IDE 有自动补全
        - 类型检查 (名字写错了 IDE 直接报错)
        - 可以加默认值
    """

    name: str                                        # 模型名字 (唯一标识)
    model_type: str                                  # 模型类型: "gpt", "unet", etc.
    d_model: int = 256                               # 隐藏维度
    n_layers: int = 6                               # 层数
    n_heads: int = 8                                # attention 头数
    dropout: float = 0.1
    checkpoint_path: Optional[str] = None            # 预训练权重路径

    def to_dict(self) -> dict:
        """转成 dict (用于保存 JSON)"""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "checkpoint_path": self.checkpoint_path,
        }


# ============================================================================
# 2. Model Registry — 核心类
# ============================================================================
class ModelRegistry:
    """
    模型注册中心。

    用法:
        registry = ModelRegistry()

        # 注册
        @registry.register("mini-gpt", model_type="gpt")
        class MiniGPT(nn.Module):
            ...

        # 创建
        model = registry.create("mini-gpt", config)

        # 列出所有模型
        print(registry.list_models())
    """

    def __init__(self):
        # _models: 存模型类的字典
        # key = 模型名, value = {"cls": 类, "type": 类型, "config_cls": 配置类}
        self._models: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, model_type: str, config_cls: Type = ModelConfig):
        """
        注册一个模型。可以用作装饰器。

        用法 1 (装饰器):
            @registry.register("mini-gpt", model_type="gpt")
            class MiniGPT(nn.Module):
                ...

        用法 2 (函数调用):
            registry.register("mini-gpt", model_type="gpt", MiniGPT)
        """
        def decorator(cls):
            self._models[name] = {
                "cls": cls,
                "type": model_type,
                "config_cls": config_cls,
            }
            print(f"✅ 注册模型: '{name}' (类型: {model_type}, 类: {cls.__name__})")
            return cls
        return decorator

    def create(
        self,
        name: str,
        config: Optional[ModelConfig] = None,
        device: str = "cpu",
        **kwargs
    ) -> nn.Module:
        """
        创建模型实例。

        Args:
            name:   模型名 (必须在 registry 里注册过)
            config: 模型配置
            device: "cpu" / "mps" / "cuda"
            **kwargs: 传给模型构造函数的额外参数

        Returns:
            nn.Module 实例

        Raises:
            KeyError: 模型名未注册
        """
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(
                f"模型 '{name}' 未注册! 可用模型: [{available}]"
            )

        model_info = self._models[name]
        model_cls = model_info["cls"]

        # 从 config 提取参数 (如果提供了)
        if config is not None:
            kwargs.update({
                k: v for k, v in config.__dict__.items()
                if k in model_cls.__init__.__code__.co_varnames
            })

        # 创建模型
        model = model_cls(**kwargs)

        # 移到设备
        model = model.to(device)

        # 如果提供了 checkpoint，加载权重
        if config and config.checkpoint_path:
            self.load_checkpoint(model, config.checkpoint_path, device)

        return model

    def load_checkpoint(self, model: nn.Module, path: str, device: str = "cpu") -> None:
        """
        加载预训练权重。

        为什么需要 map_location?
            GPU 训练的模型直接 load 到 CPU 会报错
            map_location 告诉 PyTorch 把 tensor 搬到当前设备
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"📦 加载权重: {path}")

    def list_models(self) -> Dict[str, str]:
        """列出所有已注册模型"""
        return {name: info["type"] for name, info in self._models.items()}

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """获取模型的详细信息"""
        if name not in self._models:
            raise KeyError(f"模型 '{name}' 未注册")
        info = self._models[name]
        return {
            "name": name,
            "type": info["type"],
            "class": info["cls"].__name__,
            "module": info["cls"].__module__,
        }


# ============================================================================
# 3. 使用示例
# ============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=" * 60)
    print("Model Registry 演示")
    print("=" * 60)

    registry = ModelRegistry()

    # ── 注册几个模型 ──
    @registry.register("mini-gpt", model_type="llm")
    class DummyGPT(nn.Module):
        def __init__(self, d_model=256, n_layers=6, n_heads=8, **kwargs):
            super().__init__()
            self.embed = nn.Embedding(1000, d_model)
            self.linear = nn.Linear(d_model, 1000)

        def forward(self, x):
            return self.linear(self.embed(x))

    @registry.register("simple-unet", model_type="diffusion")
    class DummyUNet(nn.Module):
        def __init__(self, in_channels=3, base_channels=64, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        def forward(self, x, t):
            return self.conv(x)

    # ── 列出所有模型 ──
    print(f"\n已注册模型: {registry.list_models()}")

    # ── 创建模型 ──
    print(f"\n创建模型:")
    gpt_config = ModelConfig(
        name="my-gpt",
        model_type="llm",
        d_model=128,
        n_layers=4,
    )
    gpt = registry.create("mini-gpt", gpt_config, device="cpu")
    print(f"  GPT 参数量: {sum(p.numel() for p in gpt.parameters()):,}")

    unet = registry.create("simple-unet", device="cpu")
    print(f"  UNet 参数量: {sum(p.numel() for p in unet.parameters()):,}")

    # ── 错误处理 ──
    print(f"\n错误处理:")
    try:
        registry.create("nonexistent-model")
    except KeyError as e:
        print(f"  ✅ 正确报错: {e}")

    # ── 获取模型信息 ──
    print(f"\n模型详情:")
    for name in registry.list_models():
        info = registry.get_model_info(name)
        print(f"  {info}")

    print("\n" + "=" * 60)
    print("✅ Model Registry 完成！")
    print("这就是 AI infra 的基石 — 统一管理所有模型")
    print("下一步: 运行 02_prompt_template.py")
    print("=" * 60)
