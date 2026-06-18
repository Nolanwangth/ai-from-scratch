"""
Simple Agent Loop — AI Agent 的完整实现
==========================================
Harness 工程师最核心的工作: 让 AI 能自主完成任务。

Agent Loop (代理循环):
    ┌──────────────────────────────────────────┐
    │                                          │
    │  Think → Decide → Act → Observe → ...    │
    │    ↑                                  │
    │    └────────── Repeat ────────────────┘  │
    │                                          │
    └──────────────────────────────────────────┘

这就是 Claude Code 的工作方式:
    1. 用户说 "帮我创建一个网站"
    2. Agent 思考: "需要先创建 HTML 文件"
    3. Agent 调工具: write_file("index.html", ...)
    4. Agent 观察结果: "文件创建成功"
    5. Agent 再思考: "现在需要 CSS..."
    6. ... 循环到任务完成

你的目标:
    构建一个框架，让 Agent 能:
    - 自动分解任务
    - 选择合适的工具
    - 处理错误
    - 知道什么时候停止

这就是你要做的 Harness 工程!
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import traceback


# ============================================================================
# 1. Tool — Agent 可以使用的工具
# ============================================================================
@dataclass
class Tool:
    """
    工具定义 — Agent 可以调用的函数。

    一个工具包含:
        name:         工具名 (LLM 用于选择工具)
        description:  描述 (LLM 用于理解这个工具干什么)
        parameters:   JSON Schema (工具的参数定义)
        function:     实际执行的 Python 函数
    """
    name: str
    description: str
    parameters: Dict[str, Any]                        # JSON Schema 格式
    function: Callable
    #   function 的返回值可以是 str (成功) 或抛出异常 (失败)

    def to_openai_schema(self) -> dict:
        """转成 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }

    def execute(self, **kwargs) -> str:
        """
        执行工具并返回字符串结果。

        为什么返回字符串而不是任意类型?
            因为 LLM 只能理解文本 — 工具的返回必须是 LLM 能读懂的!
            这是 Agent 工程的核心约束。
        """
        try:
            result = self.function(**kwargs)
            return str(result)
        except Exception as e:
            # 返回错误信息给 LLM，让它能自我修正
            return f"❌ 工具执行失败: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# ============================================================================
# 2. Agent State — 追踪 Agent 的状态
# ============================================================================
class AgentState(str, Enum):
    """Agent 当前状态"""
    IDLE = "idle"                 # 空闲，等待任务
    THINKING = "thinking"         # 思考下一步
    ACTING = "acting"             # 执行工具
    OBSERVING = "observing"       # 观察工具结果
    DONE = "done"                 # 任务完成
    ERROR = "error"               # 出错


@dataclass
class AgentStep:
    """记录每一步的详细信息 (用于审计和调试)"""
    step_number: int
    state: AgentState
    thought: str                                   # LLM 的思考过程
    tool_name: Optional[str] = None                # 选择的工具 (None = 没有调用工具)
    tool_args: Optional[Dict] = None               # 工具参数
    observation: Optional[str] = None              # 工具执行结果
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# 3. Agent — 核心循环
# ============================================================================
class Agent:
    """
    AI Agent 的完整实现。

    Agent 不是 "调一次 LLM" — 而是一个循环:
        while not done:
            think → decide → act → observe

    这个循环让 Agent 能完成需要多步的复杂任务:
        "帮我创建一个 Web 应用" → Agent 自动分解为:
            Step 1: 创建 HTML 文件
            Step 2: 创建 CSS 文件
            Step 3: 创建 JS 文件
            Step 4: 验证所有文件正确
            Step 5: 启动服务器
    """

    def __init__(
        self,
        system_prompt: str,                          # 系统指令
        tools: List[Tool],                           # 可用工具列表
        max_steps: int = 20,                         # 最大循环步数 (防止无限循环)
        verbose: bool = True,                        # 是否打印详细日志
    ):
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}      # name → Tool 映射
        self.max_steps = max_steps
        self.verbose = verbose

        # 对话历史 (Agent 的记忆)
        self.messages: List[Dict] = []
        self.steps: List[AgentStep] = []             # 审计日志

        # 初始化: 添加系统指令
        self.messages.append({
            "role": "system",
            "content": self._build_system_prompt(),
        })

    def _build_system_prompt(self) -> str:
        """构建完整的系统指令 (包含工具列表)"""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        return f"""{self.system_prompt}

你可以使用以下工具:
{tool_descriptions}

工作方式:
1. 分析任务，思考下一步需要做什么
2. 如果需要信息或执行操作，调用合适的工具
3. 观察工具结果，决定下一步
4. 当任务完成时，在回复中说 "DONE" 并提供摘要

重要规则:
- 一次只调用一个工具
- 不要猜测 — 如果不确定，使用工具
- 如果工具失败，尝试另一种方法
- 如果实在无法完成，说 "DONE: cannot complete" 并说明原因
"""

    def _simulate_llm_call(self, messages: List[dict]) -> Tuple[str, Optional[str], Optional[Dict]]:
        """
        模拟 LLM 调用 (真实实现会调用 Claude/GPT API)。

        返回: (思考文本, 工具名或None, 工具参数或None)

        真实实现:
            1. 构建 prompt: messages → API format
            2. 调用 LLM API
            3. 解析返回: 是 text response 还是 function call?
            4. 如果是 function call → 返回工具名和参数
            5. 如果是 text → 返回文本

        这里用一个简单的规则逻辑模拟 LLM 决策。
        在你的 Harness 里，这里会替换为真正的 LLM API 调用。
        """
        # 简化逻辑: 取最后一条用户消息，模拟 LLM 的决策
        last_user_msg = messages[-1]["content"] if messages else ""

        # 规则: 如果任务里提到 "文件"，用 create_file 工具
        if "文件" in last_user_msg or "file" in last_user_msg.lower():
            return (
                "用户想要操作文件，我应该创建文件",
                "create_file",
                {"filename": "output.txt", "content": "模拟的文件内容\nHello, Agent!"},
            )

        # 规则: 如果提到 "搜索" 或 "查"
        if "搜索" in last_user_msg or "查" in last_user_msg:
            return (
                "用户想要查信息，我需要搜索",
                "web_search",
                {"query": last_user_msg},
            )

        # 默认: 不需要工具，直接回复
        return (
            "我理解了用户的需求，任务完成。",
            None,
            None,
        )

    def run(self, task_description: str) -> Dict[str, Any]:
        """
        运行 Agent 完成一个任务。

        这是整个 Agent 框架的核心 — 实现了 Think → Decide → Act → Observe 循环。

        Args:
            task_description: 用户任务 (自然语言)

        Returns:
            {"status": "success"|"failed", "steps": [...], "final_answer": str}
        """
        self.log(f"\n{'='*50}")
        self.log(f"🚀 收到任务: {task_description}")
        self.log(f"{'='*50}")

        # 把用户任务加入对话
        self.messages.append({"role": "user", "content": task_description})

        for step_num in range(self.max_steps):
            # ── Step 1: THINK — LLM 思考接下来做什么 ──
            self.log(f"\n--- Step {step_num + 1}/{self.max_steps} ---")

            thought, tool_name, tool_args = self._simulate_llm_call(self.messages)

            step = AgentStep(
                step_number=step_num + 1,
                state=AgentState.THINKING,
                thought=thought,
            )
            self.log(f"💭 思考: {thought}")

            # ── Step 2: DECIDE — 需要用工具吗? ──
            if tool_name is None:
                # 不需要工具 → 任务完成
                self.log(f"✅ 任务完成 (不需要更多工具)")
                self.steps.append(step)
                return self._finish("success", thought)

            # 检查工具是否存在
            if tool_name not in self.tools:
                error_msg = f"未知工具 '{tool_name}'"
                self.log(f"❌ {error_msg}")
                self.messages.append({"role": "user", "content": error_msg})
                step.state = AgentState.ERROR
                step.observation = error_msg
                self.steps.append(step)
                continue

            # ── Step 3: ACT — 执行工具 ──
            step.state = AgentState.ACTING
            step.tool_name = tool_name
            step.tool_args = tool_args

            self.log(f"🔧 执行: {tool_name}({tool_args})")
            observation = self.tools[tool_name].execute(**(tool_args or {}))

            # ── Step 4: OBSERVE — 观察结果 ──
            step.state = AgentState.OBSERVING
            step.observation = observation
            self.steps.append(step)

            self.log(f"👁️  结果: {observation[:200]}{'...' if len(observation) > 200 else ''}")

            # 把工具结果加入对话 (这是 Agent 的核心 — 让 LLM 看到工具的执行结果)
            self.messages.append({
                "role": "assistant",
                "content": f"[调用工具: {tool_name}]",
                "tool_calls": [{"name": tool_name, "arguments": tool_args}],
            })
            self.messages.append({
                "role": "tool",
                "content": observation,
            })

            # ── Check: 任务完成? ──
            if "DONE" in observation.upper() or step_num == self.max_steps - 1:
                return self._finish("success" if "DONE" in observation.upper() else "max_steps", observation)

        # 超出最大步数
        return self._finish("max_steps_reached", "超过最大步数限制")

    def _finish(self, status: str, final_answer: str) -> Dict[str, Any]:
        """输出最终结果"""
        return {
            "status": status,
            "total_steps": len(self.steps),
            "steps": self.steps,
            "final_answer": final_answer,
            "messages": self.messages,
        }

    def log(self, msg: str):
        """日志输出 (verbose 控制是否打印)"""
        if self.verbose:
            print(msg)


# ============================================================================
# 4. 示例工具 — 真实可用的函数
# ============================================================================
def create_file(filename: str, content: str) -> str:
    """创建文件"""
    path = Path(f"/tmp/{filename}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"✅ 文件已创建: {path} (大小: {len(content)} 字节)"

def web_search(query: str) -> str:
    """模拟网络搜索"""
    results = {
        "python": "Python 是一种高级编程语言，广泛用于 AI、数据科学和 Web 开发。最新版本是 3.13。",
        "transformer": "Transformer 是 2017 年由 Google 提出的神经网络架构，基于自注意力机制。",
        "diffusion": "Diffusion Model 是一种生成模型，通过逐步去噪来生成数据。",
        "claude": "Claude 是 Anthropic 开发的 AI 助手系列。",
        "default": f"关于 '{query}' 的搜索结果: 这是一个模拟的搜索。",
    }

    for key, value in results.items():
        if key in query.lower():
            return f"📊 搜索结果: {value}"
    return f"📊 {results['default']}"

def calculator(expression: str) -> str:
    """执行数学表达式"""
    try:
        # 安全地 eval (生产中要用 ast.literal_eval 或专门的表达式解析器)
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
        return f"🧮 {expression} = {result}"
    except Exception as e:
        return f"❌ 计算失败: {e}"


# ============================================================================
# 5. 运行演示
# ============================================================================
if __name__ == "__main__":
    from pathlib import Path

    print("=" * 60)
    print("Agent Loop 演示")
    print("=" * 60)

    # ── 定义工具集 ──
    tools = [
        Tool(
            name="create_file",
            description="创建一个新文件。参数: filename (文件名), content (文件内容)",
            parameters={
                "filename": {"type": "string", "description": "文件名"},
                "content": {"type": "string", "description": "文件内容"},
            },
            function=create_file,
        ),
        Tool(
            name="web_search",
            description="搜索互联网获取信息。参数: query (搜索查询)",
            parameters={
                "query": {"type": "string", "description": "搜索查询"},
            },
            function=web_search,
        ),
        Tool(
            name="calculator",
            description="执行数学计算。参数: expression (数学表达式, 如 '3*5+2')",
            parameters={
                "expression": {"type": "string", "description": "数学表达式"},
            },
            function=calculator,
        ),
    ]

    # ── 创建 Agent ──
    agent = Agent(
        system_prompt="你是一个有用的 AI 助手，可以创建文件、搜索信息和执行计算。",
        tools=tools,
        max_steps=10,
        verbose=True,
    )

    # ── 测试任务 1: 创建文件 ──
    print(f"\n{'='*60}")
    print("测试 1: 创建文件")
    print(f"{'='*60}")
    result = agent.run("帮我创建一个名为 hello.py 的 Python 文件，内容是 print('Hello from Agent!')")
    print(f"\n最终状态: {result['status']}")
    print(f"总步数: {result['total_steps']}")
    # 验证文件是否真的创建了
    expected_path = Path("/tmp/output.txt")
    if expected_path.exists():
        print(f"文件验证: {expected_path.read_text()}")

    # ── 测试任务 2: 搜索 ──
    print(f"\n{'='*60}")
    print("测试 2: 搜索信息")
    print(f"{'='*60}")
    agent2 = Agent(system_prompt="你是一个研究员助手。", tools=tools, max_steps=10)
    result2 = agent2.run("帮我搜索 Transformer 是什么")

    print(f"\n最终状态: {result2['status']}")
    print(f"总步数: {result2['total_steps']}")
    print(f"审计日志:")
    for step in result2['steps']:
        print(f"  Step {step.step_number}: {step.state.value} | 思考: {step.thought[:60]}...")

    # ── Agent 循环架构总结 ──
    print(f"\n{'='*60}")
    print("Agent Loop 架构总结")
    print(f"{'='*60}")
    print("""
    while not done:
        # 1. THINK: LLM 分析当前状态
        thought = llm.think(messages)

        # 2. DECIDE: 选择工具
        if needs_tool(thought):
            tool = select_tool(thought)

            # 3. ACT: 执行工具
            result = tool.execute(args)

            # 4. OBSERVE: 记录结果
            messages.append(result)

            # 5. LOOP: 回到思考
        else:
            # 不需要工具 → 任务完成
            done = True
    """)

    print("=" * 60)
    print("✅ Agent Loop 完成！")
    print("这就是 Claude Code、LangChain Agent 等框架的核心循环!")
    print("你现在已经理解了 AI Harness 工程的基石。")
    print("=" * 60)
    print()
    print("🎉 恭喜！你完成了整个 AI From Scratch 学习项目！")
    print()
    print("回顾你学到了什么:")
    print("  01_foundations:  Python 速成 + Tensor + Autograd + 训练循环")
    print("  02_transformer:  Attention → PE → Block → Mini-GPT → 训练生成文本")
    print("  03_diffusion:    Forward Noise → U-Net → DDPM → 训练生成图像")
    print("  04_harness:      Registry → Prompt Template → LLM Client → Agent Loop")
    print()
    print("你已经从 Python 新手成长为掌握了:")
    print("  ✅ Transformer 完整实现")
    print("  ✅ Diffusion 完整实现")
    print("  ✅ Agent Harness 工程")
    print("  ✅ MPS GPU 加速")
    print()
    print("下一步: 用这些知识构建你自己的 Harness 框架吧! 🚀")
    print("=" * 60)
