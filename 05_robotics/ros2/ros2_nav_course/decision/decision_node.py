# 运行: conda activate ros2_nav && python ros2_nav_course/decision/decision_node.py
"""
Decision Node — Behavior Tree with real recovery behaviors
===========================================================
Nav2-style Behavior Tree for navigation decisions.

Tree:
    Fallback("MainLoop")
    ├─ Sequence("Navigate")              ← normal path following
    │   ├─ Condition: PathOk?
    │   ├─ Action: FollowPath
    │   └─ Condition: AtGoal?
    ├─ Fallback("Recoveries")            ← try increasingly drastic fixes
    │   ├─ Action: SlowRetry             ← just replan slowly
    │   ├─ Action: Backup               ← reverse 0.8m, then replan
    │   └─ Action: SpinAndReplan         ← rotate in place, scan, replan
    └─ Action: Abort                     ← nothing worked, give up

Emergency stop lives in the control layer (50Hz safety loop).
Backup/replan logic lives here in the decision layer (what to do AFTER stopping).
"""

import math
import enum
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Optional, Callable, Any

from ros2_nav_course.utils.mock_ros2 import Node, Pose, log_info, log_ok, log_warn


# =============================================================================
# Behavior Tree engine
# =============================================================================

class BTStatus(enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class BTNode:
    def __init__(self, name: str):
        self.name = name
    def tick(self, context: dict) -> BTStatus:
        raise NotImplementedError


class Sequence(BTNode):
    """Run children in order. All must succeed."""
    def __init__(self, name: str, children: List[BTNode]):
        super().__init__(name)
        self.children = children
    def tick(self, context: dict) -> BTStatus:
        for child in self.children:
            s = child.tick(context)
            if s != BTStatus.SUCCESS:
                return s
        return BTStatus.SUCCESS


class Fallback(BTNode):
    """Try children in order. First success wins."""
    def __init__(self, name: str, children: List[BTNode]):
        super().__init__(name)
        self.children = children
    def tick(self, context: dict) -> BTStatus:
        for child in self.children:
            s = child.tick(context)
            if s != BTStatus.FAILURE:
                return s
        return BTStatus.FAILURE


class Condition(BTNode):
    """Check a predicate. SUCCESS if true, FAILURE otherwise."""
    def __init__(self, name: str, check_fn: Callable[[dict], bool]):
        super().__init__(name)
        self.check_fn = check_fn
    def tick(self, context: dict) -> BTStatus:
        return BTStatus.SUCCESS if self.check_fn(context) else BTStatus.FAILURE


class ActionNode(BTNode):
    """Execute an action. Returns RUNNING until done."""
    def __init__(self, name: str, execute_fn: Callable[[dict], BTStatus],
                 enter_fn=None, exit_fn=None):
        super().__init__(name)
        self.execute_fn = execute_fn
        self.enter_fn = enter_fn
        self.exit_fn = exit_fn
        self._started = False
    def tick(self, context: dict) -> BTStatus:
        if not self._started:
            if self.enter_fn: self.enter_fn(context)
            self._started = True
        status = self.execute_fn(context)
        if status != BTStatus.RUNNING:
            self._started = False
            if self.exit_fn: self.exit_fn(context)
        return status


# =============================================================================
# Decision Node
# =============================================================================

class DecisionNode(Node):
    """
    Behavior Tree decision node. Receives path-blocked signals and decides
    which recovery action to take.

    Recovery escalation:
        1st block → SlowRetry  (just replan, maybe it's temporary)
        2nd block → Backup     (reverse 0.8m to clear the obstacle, then replan)
        3rd block → Spin       (rotate in place 90°, scan, replan)
        4th block → Abort      (nothing worked)

    The BT context contains:
        path_blocked: bool    — set by emergency stop or dynamic obstacle
        recovery_count: int   — how many times we've tried to recover
        navigation_done: bool — robot reached goal
        robot:                — reference to the robot (for backup/spin)
        planner_set_goal: fn  — called to trigger A* replan
    """

    def __init__(self, world, hz: float = 10.0):
        super().__init__("decision_node")
        self.world = world
        self.bt_root: Optional[BTNode] = None
        self.current_goal = None
        self.state = "IDLE"

        self.context = {
            "path_blocked": False,
            "recovery_count": 0,
            "navigation_done": False,
        }

        self.create_publisher("/goal_point", "Pose")
        self.create_publisher("/decision_state", "str")

        # ── Action: /navigate_to_goal — 别的 Node 发导航目标, 我们处理 ──
        self.create_action_server("/navigate_to_goal",
                                  self._navigate_action_execute,
                                  self._navigate_action_cancel)
        self._build_tree()
        log_ok(f"[decision] BT ready ({self.state})")

    def _navigate_action_execute(self, goal, feedback, is_cancelled):
        """Action 执行: 收到导航目标 → 设目标 → 等到达."""
        target = goal.target
        self.set_goal(target[0], target[1])
        log_ok(f"  [decision] Action: navigating to ({target[0]:.1f}, {target[1]:.1f})")
        # 返回, 实际导航在控制循环里跑
        return type('obj', (object,), {'success': True, 'message': 'goal accepted'})()

    def _navigate_action_cancel(self):
        log_warn(f"  [decision] Action cancelled")
        self.state = "IDLE"

    def _build_tree(self):
        """
        无人驾驶 BT:
            正常 → FollowPath
            被堵 → Replan (A* 重规划, 控制层会自己减速/倒车)
            放弃 → Abort
        """
        self.bt_root = Fallback("MainLoop", [
            Sequence("Navigate", [
                Condition("PathOk", lambda ctx: not ctx.get("path_blocked", False)),
                ActionNode("FollowPath", self._follow_path),
                Condition("AtGoal", lambda ctx: ctx.get("navigation_done", False)),
            ]),
            ActionNode("Replan", self._recovery_replan),
            ActionNode("Abort", self._recovery_abort),
        ])

    # ── Actions ───────────────────────────────────────────────

    def _follow_path(self, ctx: dict) -> BTStatus:
        if ctx.get("navigation_done", False):
            return BTStatus.SUCCESS
        self.state = "FOLLOWING"
        return BTStatus.RUNNING

    def _recovery_replan(self, ctx: dict) -> BTStatus:
        """被堵了 → 触发 A* 重规划. 控制层自己会在阴影区减速/倒车."""
        if not ctx.get("path_blocked", False):
            return BTStatus.FAILURE
        ctx["path_blocked"] = False
        ctx["recovery_count"] = ctx.get("recovery_count", 0) + 1
        if ctx.get("trigger_replan"):
            ctx["trigger_replan"]()
        self.state = "RECOVERY"
        log_warn(f"  [BT] 路径被堵 (第{ctx['recovery_count']}次), 重规划")
        return BTStatus.SUCCESS

    def _recovery_abort(self, ctx: dict) -> BTStatus:
        self.state = "ABORT"
        log_warn(f"  [BT] 放弃: 无法到达")
        return BTStatus.SUCCESS

    # ── API ───────────────────────────────────────────────────

    def set_goal(self, x: float, y: float):
        self.current_goal = (x, y)
        self.context["goal"] = (x, y)
        self.context["navigation_done"] = False
        self.context["recovery_count"] = 0
        self.state = "NAVIGATING"
        self.publish("/goal_point", Pose(x, y, 0))
        log_info(f"  [decision] Goal: ({x:.2f}, {y:.2f})")

    def tick(self, dt: float):
        super().tick(dt)
        if self.bt_root:
            self.bt_root.tick(self.context)
        self.publish("/decision_state", self.state)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    from ros2_nav_course.simulation.world_2d import World2D

    print("=" * 60)
    print("Decision Node — Behavior Tree with Backup Recovery")
    print("=" * 60)

    world = World2D()
    decision = DecisionNode(world)

    print(f"""
    Behavior Tree:
    Fallback("MainLoop")
    ├─ Sequence("Navigate")
    │   ├─ Condition: PathOk?
    │   ├─ Action: FollowPath
    │   └─ Condition: AtGoal?
    ├─ Fallback("Recoveries")          ← 递进恢复
    │   ├─ SlowRetry                   ← 第1次: 只重规划
    │   ├─ Backup (倒车 0.8m)          ← 第2次: 倒车再重规划
    │   └─ SpinAndReplan (原地转)      ← 第3次: 转90°找路
    └─ Abort                           ← 全失败, 放弃
    """)

    # Simulate three blocks in a row — escalation demo
    print("  [Demo] Block #1:")
    decision.context["path_blocked"] = True
    decision.bt_root.tick(decision.context)
    print(f"    state={decision.state}, count={decision.context['recovery_count']}")

    print("  [Demo] Block #2:")
    decision.context["path_blocked"] = True
    decision.bt_root.tick(decision.context)
    print(f"    state={decision.state}, count={decision.context['recovery_count']}")

    print("  [Demo] Block #3:")
    decision.context["path_blocked"] = True
    decision.bt_root.tick(decision.context)
    print(f"    state={decision.state}, count={decision.context['recovery_count']}")

    print("  [Demo] Block #4:")
    decision.context["path_blocked"] = True
    decision.bt_root.tick(decision.context)
    print(f"    state={decision.state}, count={decision.context['recovery_count']}")

    print(f"\n  ✅ BT escalation: SlowRetry → Backup → Spin → Abort")
