# 运行: conda activate diffusion && python 06_harness/02_prompt_template.py
#       (纯标准库，任意 env 都可以)
"""
Prompt Template — LLM 调用的标准模式
======================================
Harness 工程师的日常: 封装 API 调用, 管理 prompt, 处理异常。

本文件展示了企业级 LLM 调用的标准模式:
    1. Prompt Template (带占位符的模板)
    2. LLM Client (统一的 API 调用接口)
    3. Retry + Fallback (容错)
    4. Streaming (流式输出)
    5. Response Parser (解析返回)

这个模式适用于 Claude, GPT, Llama 等所有 LLM。
"""

from typing import Dict, List, Optional, Any, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import time
import json


# ============================================================================
# 1. Message 类型 — 统一的消息格式
# ============================================================================
class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"       # 系统指令 (给模型的最高优先级指令)
    USER = "user"           # 用户消息
    ASSISTANT = "assistant" # 模型回复
    TOOL = "tool"           # 工具调用结果


@dataclass
class Message:
    """
    统一的消息格式，所有 LLM API 都长这样。

    例:
        system_msg = Message(role=MessageRole.SYSTEM, content="你是 Claude")
        user_msg = Message(role=MessageRole.USER, content="你好")
    """
    role: MessageRole
    content: str
    name: Optional[str] = None                      # 可选的消息名 (用于工具调用)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 2. Prompt Template — 模板化 prompt
# ============================================================================
class PromptTemplate:
    """
    Prompt 模板引擎 (简化版)。

    用法:
        template = PromptTemplate(
            "你是一个 {role}，请用 {language} 回答: {question}"
        )
        result = template.format(role="教师", language="中文", question="什么是 AI?")

    和 Python 的 str.format() 一样，但加了:
        - 参数验证 (缺少变量会报错)
        - 变量列表 (方便 IDE 知道要传什么参数)
    """

    def __init__(self, template: str):
        self.template = template
        # 提取模板里的所有变量名 {xxx}
        self.variables = self._extract_variables(template)

    def _extract_variables(self, template: str) -> List[str]:
        """从模板字符串中提取所有 {变量名}"""
        import re
        return re.findall(r"\{(\w+)\}", template)

    def format(self, **kwargs) -> str:
        """
        填充模板，自动验证参数。

        Raises:
            KeyError: 缺少必要变量
        """
        # 检查: 所有变量都提供了吗?
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"缺少变量: {missing}")

        return self.template.format(**kwargs)

    def __repr__(self):
        return f"PromptTemplate(template='{self.template[:50]}...', variables={self.variables})"


# ============================================================================
# 3. LLM Client — 统一的 API 调用接口
# ============================================================================
class LLMClient:
    """
    统一的 LLM 客户端。

    特点:
        - 支持多个后端 (Claude / GPT / Ollama)
        - 自动重试 (retry)
        - 流式输出 (streaming)
        - 模型 fallback (主模型挂了自动换备选)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        default_model: str = "claude-sonnet-4-6",
        fallback_models: List[str] = None,           # 备选模型列表
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.fallback_models = fallback_models or []
        self.max_retries = max_retries
        self.timeout = timeout

        # Token 统计
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_calls = 0

    def call(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[dict]] = None,
    ) -> dict:
        """
        调用 LLM，带自动重试和 fallback。

        这是 Harness 工程师写得最多的函数 — 封装 API 调用，处理各种边界情况。

        Args:
            messages: 对话消息列表
            model: 模型名 (None = 用默认)
            temperature: 温度 (0=确定性, 1=正常, >1=创意)
            max_tokens: 最大输出 token 数
            tools: 工具定义列表 (Function Calling)

        Returns:
            {"content": str, "model": str, "usage": {"input": int, "output": int}}
        """
        model = model or self.default_model
        models_to_try = [model] + [m for m in self.fallback_models if m != model]

        last_error = None
        for attempt, current_model in enumerate(models_to_try):
            try:
                result = self._call_api(
                    messages, current_model, temperature, max_tokens, tools
                )
                # 成功! 更新统计
                self.total_calls += 1
                self.total_tokens_in += result.get("usage", {}).get("input", 0)
                self.total_tokens_out += result.get("usage", {}).get("output", 0)
                return result

            except Exception as e:
                last_error = e
                if attempt < len(models_to_try) - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️ 模型 {current_model} 失败: {e}")
                    print(f"   等待 {wait_time}s 后尝试 {models_to_try[attempt + 1]}...")
                    time.sleep(wait_time)

        # 所有模型都失败了
        raise RuntimeError(
            f"所有模型调用失败! 最后错误: {last_error}"
        )

    def _call_api(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[dict]],
    ) -> dict:
        """
        实际调用 API (这里用模拟实现代替真实的 HTTP 请求)。

        真实代码里这里会是:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers={"x-api-key": self.api_key},
                json={...},
                timeout=self.timeout,
            )
        """
        # 模拟 API 调用 (真实场景替换为实际的 HTTP 请求)
        formatted_msgs = [
            {"role": m.role.value, "content": m.content}
            for m in messages
        ]

        # 模拟返回 (真实 API 会返回实际的 LLM 响应)
        print(f"📡 调用 {model}...")
        print(f"   消息数: {len(formatted_msgs)}")
        print(f"   temperature: {temperature}")
        print(f"   max_tokens: {max_tokens}")

        return {
            "content": f"[{model} 的模拟回复]",
            "model": model,
            "usage": {"input": sum(len(m["content"]) // 4 for m in formatted_msgs), "output": 50},
        }

    def call_streaming(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """
        流式调用 — 逐 token 返回，用于打字机效果。

        真实实现会用 SSE (Server-Sent Events) 或者 WebSocket。
        """
        model = model or self.default_model

        # 模拟流式输出
        response = self.call(messages, model, temperature, max_tokens)
        content = response["content"]

        for char in content:
            yield char
            time.sleep(0.02)                         # 模拟延迟

    def get_stats(self) -> dict:
        """获取调用统计"""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_tokens_in,
            "total_output_tokens": self.total_tokens_out,
            "estimated_cost": f"${self.total_tokens_in * 3 / 1_000_000:.4f} + "
                              f"${self.total_tokens_out * 15 / 1_000_000:.4f}",
        }


# ============================================================================
# 4. 使用示例
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Template & LLM Client 演示")
    print("=" * 60)

    # ── Prompt Template ──
    print("\n1. Prompt Template:")
    template = PromptTemplate(
        "你是一个 {role} 专家，请用 {language} 解释: {topic}\n"
        "要求: {requirements}\n"
        "风格: {style}"
    )
    print(f"   模板: {template.template[:60]}...")
    print(f"   变量: {template.variables}")

    filled = template.format(
        role="机器学习",
        language="中文",
        topic="Transformer 的 Attention 机制",
        requirements="100 字以内，用比喻的方式",
        style="生动有趣",
    )
    print(f"\n   填充后:\n{filled}")

    # ── 错误处理 ──
    try:
        template.format(role="test")                 # 漏了很多变量
    except KeyError as e:
        print(f"\n   ✅ 参数验证: {e}")

    # ── LLM Client ──
    print(f"\n2. LLM Client:")
    client = LLMClient(
        default_model="claude-sonnet-4-6",
        fallback_models=["claude-haiku-4-5", "gpt-4o-mini"],
        max_retries=3,
    )

    # 构建对话
    messages = [
        Message(MessageRole.SYSTEM, "你是一个 Python 编程助手"),
        Message(MessageRole.USER, "写一个快速排序"),
    ]

    # 调用
    response = client.call(messages, temperature=0.3, max_tokens=500)
    print(f"\n   回复: {response['content']}")
    print(f"   模型: {response['model']}")

    # ── 流式输出 ──
    print(f"\n3. 流式输出:")
    for chunk in client.call_streaming(messages, max_tokens=100):
        print(chunk, end="", flush=True)
    print()

    # ── 调用统计 ──
    print(f"\n4. 调用统计:")
    import json
    print(json.dumps(client.get_stats(), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("✅ Prompt Template & LLM Client 完成！")
    print("核心: Template → Call(Retry+Fallback) → Parse → Stream")
    print("下一步: 运行 03_simple_agent_loop.py")
    print("=" * 60)
