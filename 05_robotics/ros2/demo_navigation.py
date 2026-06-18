#!/usr/bin/env python3
# 运行: conda activate ros2_nav && python demo_navigation.py --map
"""
🦾 完整 ROS2 导航系统 — 教学演示
===================================

感知 → SLAM(代价地图) → 决策(Behavior Tree) → A* → MPC+PID

所有模块通过 ROS2 Topic 通信 (mock 版, Mac 能跑)

输出:
    --map:    只画地图 (教室布局)
    --run:    跑完整导航, ASCII 动画实时显示
    --stats:  跑完打印各模块统计

用法:
    conda activate ros2_nav
    python demo_navigation.py --map      # 先看地图
    python demo_navigation.py --run      # 跑导航
    python demo_navigation.py --stats    # 看统计
"""

import sys
import time
import math
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ros2_nav_course.simulation.world_2d import World2D
from ros2_nav_course.simulation.robot_model import DifferentialDriveRobot
from ros2_nav_course.perception.sensor_node import PerceptionNode
from ros2_nav_course.slam.rtabmap_node import RTABMapNode
from ros2_nav_course.slam.costmap_node import CostmapNode
from ros2_nav_course.decision.decision_node import DecisionNode
from ros2_nav_course.planning.astar_node import AStarPlannerNode
from ros2_nav_course.control.control_node import ControlNode
from ros2_nav_course.utils.mock_ros2 import (
    Node, OccupancyGrid, Pose, Path, log_info, log_ok, log_warn, log_error,
    Rate, explain_ros2_dataflow
)
from ros2_nav_course.utils.visualizer import NavigationVisualizer


def print_separator(title: str):
    w = 70
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")


def print_banner():
    print(r"""
   ╔═══════════════════════════════════════════════════════╗
   ║                                                       ║
   ║     🦾  ROS2 导航教学系统                               ║
   ║                                                       ║
   ║     感知 (D435) → RTAB-Map SLAM → 代价地图              ║
   ║          ↓                                            ║
   ║     决策 (Behavior Tree)                               ║
   ║          ↓                                            ║
   ║     A* 全局规划                                        ║
   ║          ↓                                            ║
   ║     MPC (横向) + PID (纵向) → 小车前进                  ║
   ║                                                       ║
   ║     全部用 ROS2 Topic 通信, Node 模块化                 ║
   ║                                                       ║
   ╚═══════════════════════════════════════════════════════╝
    """)

# =============================================================================
# Demo: 只看地图
# =============================================================================

def demo_show_map():
    world = World2D()
    print("\n教室 (10m × 10m):")
    print(f"  @ = 障碍物  . = 自由  R = 机器人  T = 目标  * = 规划路径\n")
    print(world.render_ascii(
        robot_x=1.0, robot_y=1.0,
        target_x=8.0, target_y=8.0
    ))
    print(f"\n  地标:")
    for name, pos in world.landmarks.items():
        print(f"    {name:>8} @ ({pos[0]:.1f}, {pos[1]:.1f})")

    # 展示代价地图
    print_separator("代价地图 (障碍物周围有膨胀层)")
    raw_grid, ox, oy = world.to_occupancy_grid(resolution=0.1)
    H, W = raw_grid.shape
    show_grid = raw_grid.copy()
    # 标出机器人和目标
    rx = int(1.0 / 0.1); ry = int(1.0 / 0.1)
    tx = int(8.0 / 0.1); ty = int(8.0 / 0.1)
    if 0 <= ry < H and 0 <= rx < W: show_grid[ry, rx] = 50
    if 0 <= ty < H and 0 <= tx < W: show_grid[ty, tx] = 50
    chars = {-1: '.', 0: '.', 50: '●', 100: '█'}
    for iy in range(H-1, -1, -2):
        line = ''.join(chars.get(v, '?') for v in show_grid[iy, ::2])
        print(f"  {line}")



# =============================================================================
# Demo: 跑完整导航
# =============================================================================

def demo_explain_architecture():
    """Print ROS2 dataflow + algorithm table"""
    explain_ros2_dataflow()

    print_separator("Algorithm Summary")
    print("""
    Layer           Algorithm                         Frequency
    ─────────       ────────────────────────────       ─────────
    Perception      D435 stereo IR + noise             30Hz
    SLAM            RTAB-Map 3-level memory (STM→WM→LTM)  30Hz
    Costmap         Distance transform + inflation     10Hz
    Decision        Behavior Tree (Navigate/Replan/Abort)  on demand
    Global Plan     A* (f=g+h, 8-neighbor, diagonal)    on path change
    Local Plan      SpeedProfiler (curve+goal+obstacle)  on A* replan

    ── split mode (--mode=split) ──
    Lateral Ctrl    MPC: 41 ω candidates, 10-step rollout  50Hz
    Longitudinal    PID: Kp*e + Ki*∫e + Kd*de/dt           50Hz

    ── unified mode (--mode=unified) ──
    Joint Ctrl      UnifiedMPC: 77 (v,ω) candidates         50Hz
                    一个代价函数 → 方向盘+油门一起选
                    不需要 PID, 不需要 SpeedProfiler
    """)


def demo_run_navigation_viz(control_mode="split"):
    """带可视化窗口的导航"""
    print_banner()

    world = World2D()
    robot = DifferentialDriveRobot(x=1.0, y=1.0, theta=0.0)
    robot.max_v = 1.0
    robot.max_omega = 2.0

    # 初始化所有节点
    perception = PerceptionNode(world, robot, hz=30.0)
    slam = RTABMapNode(world, robot, keyframe_min_dist=0.3, hz=10.0)
    costmap = CostmapNode(robot_radius=0.3, inflation_radius=0.5)
    decision = DecisionNode(world)
    decision.context["robot"] = robot
    planner = AStarPlannerNode(world)
    controller = ControlNode(world, robot, hz=50.0, mode=control_mode)

    # Wire BT recovery to replanning: when BT says "replan", invalidate plan
    def _bt_trigger_replan():
        nonlocal plan
        plan = None
    decision.context["trigger_replan"] = _bt_trigger_replan

    # 设置目标
    goal_x, goal_y = world.landmarks["workstation"]
    decision.set_goal(goal_x, goal_y)

    # 初始化可视化
    viz = NavigationVisualizer(world)
    viz.goal = (goal_x, goal_y)
    log_ok("Visualization window open — robot is running!")

    # 初始地图
    from ros2_nav_course.utils.mock_ros2 import Odometry as OdomMsg
    for _ in range(5):
        x, y, t = robot.get_pose()
        slam._odom_callback(OdomMsg(pose=Pose(x, y, t)))
    slam._tick()
    costmap._inflate_and_publish()
    planner.latest_costmap = costmap.costmap
    controller.costmap = costmap.costmap


    # 初算路径
    planner.set_pose(robot.x, robot.y)
    planner.set_goal(goal_x, goal_y)
    plan = planner.plan()
    if plan:
        controller.set_path(plan)

    # 主循环
    max_steps = 2000
    reached = False
    dynamic_obstacle = None
    bt_triggered = False

    for step in range(max_steps):
        if step % 2 == 0:
            perception._sensor_callback()

        if step % 5 == 0:
            x, y, t = robot.get_pose()
            slam._odom_callback(OdomMsg(pose=Pose(x, y, t)))

        # Dynamic obstacle: drop at step 400 (~8s), 2/3 along path (ahead of robot)
        if step == 400:
            from ros2_nav_course.simulation.world_2d import Obstacle
            idx = min(len(plan) * 2 // 3, len(plan) - 1) if plan else 0
            bx, by = plan[idx] if plan else (5.0, 5.0)
            if math.sqrt((bx-robot.x)**2 + (by-robot.y)**2) < 1.5:
                bx, by = robot.x + 2.0, robot.y + 2.0
            dynamic_obstacle = Obstacle(bx, by, 1.0, 1.0, "BOX!")
            world.obstacles.append(dynamic_obstacle)
            # Immediately regenerate costmap so A* sees the new obstacle
            slam._tick()
            costmap._inflate_and_publish()
            planner.latest_costmap = costmap.costmap
            controller.costmap = costmap.costmap

            viz.update_costmap(costmap.costmap)
            viz.refresh_obstacles()
            log_warn(f"  ⚡ Dynamic obstacle appeared at ({bx:.1f}, {by:.1f})!")
            plan = None  # force replan
            decision.context["path_blocked"] = True

        if step % 20 == 0:
            slam._tick()
            costmap._inflate_and_publish()
            planner.latest_costmap = costmap.costmap
            controller.costmap = costmap.costmap


        # 每 2 秒重规划 (正常维护) 或 path_blocked 时立即重规划
        if plan is None or step % 100 == 0:
            was_blocked = decision.context.get("path_blocked", False)

            # BT 运行: 被堵则递进恢复 (SlowRetry → Backup → Spin → Abort)
            if was_blocked:
                decision.bt_root.tick(decision.context)
                bt_triggered = True
            else:
                # 路段通畅: 重置恢复计数
                decision.context["recovery_count"] = 0

            planner.set_pose(robot.x, robot.y)
            new_plan = planner.plan(verbose=was_blocked)
            if new_plan:
                plan = new_plan
                controller.set_path(plan)
                decision.context["path_blocked"] = False
                if was_blocked:
                    log_ok(f"  ↻ Replanned: {len(plan)} waypoints (recovery #{decision.context.get('recovery_count', 0)})")
            elif was_blocked:
                log_warn("  ⚠ A* still can't find a path after recovery")

        controller._control_loop()

        # Emergency stop fired → immediate BT replan
        if controller._needs_replan:
            controller._needs_replan = False
            decision.context["path_blocked"] = True
            plan = None

        # Visualization: ~6Hz render
        if step % 8 == 0:
            if costmap.costmap is not None:
                viz.update_costmap(costmap.costmap)

            kf_list = [(kf.x, kf.y) for kf in slam.wm.values()]
            kf_list += [(kf.x, kf.y) for kf in slam.stm]
            viz.draw(
                robot.x, robot.y, robot.theta,
                robot.v, robot.omega,
                path=plan,
                kfs=kf_list,
                n_loops=slam.n_loops,
                wm_size=len(slam.wm),
                goal=(goal_x, goal_y),
                step=step,
            )

        dist_to_goal = math.sqrt((robot.x - goal_x)**2 + (robot.y - goal_y)**2)
        if dist_to_goal < 0.3:
            viz.draw(robot.x, robot.y, robot.theta, robot.v, robot.omega,
                       path=plan, kfs=[], n_loops=slam.n_loops,
                       wm_size=len(slam.wm), goal=(goal_x, goal_y), step=step)
            viz.finalize(True)
            log_ok(f"\n🏁 到达工位! (误差 {dist_to_goal:.2f}m) 总步数={step}")
            reached = True
            break

    if not reached:
        viz.finalize(False)
        log_warn(f"未到达 (距离 {dist_to_goal:.2f}m)")

    return reached


def demo_run_navigation(control_mode="split"):
    print_banner()

    world = World2D()
    robot = DifferentialDriveRobot(x=1.0, y=1.0, theta=0.0)
    robot.max_v = 1.0
    robot.max_omega = 2.0

    # ── 初始化所有节点 ──
    log_ok("初始化 ROS2 节点...")

    perception = PerceptionNode(world, robot, hz=30.0)
    slam = RTABMapNode(world, robot, keyframe_min_dist=0.5, hz=10.0)
    costmap = CostmapNode(robot_radius=0.3, inflation_radius=0.5)
    decision = DecisionNode(world)
    decision.context["robot"] = robot
    planner = AStarPlannerNode(world)
    controller = ControlNode(world, robot, hz=50.0, mode=control_mode)

    log_ok("初始化 ROS2 节点... done")

    # ── 设置目标 ──
    goal_x, goal_y = world.landmarks["workstation"]
    decision.set_goal(goal_x, goal_y)

    log_ok(f"目标: workstation ({goal_x:.1f}, {goal_y:.1f})")
    log_ok(f"起点: 充电桩 (1.0, 1.0)")
    log_ok("开始导航! 50Hz 控制循环\n")

    # ── 先建初始地图 + costmap ──
    # 让 SLAM 跑几帧建地图
    from ros2_nav_course.utils.mock_ros2 import Odometry as OdomMsg
    for _ in range(5):
        x, y, t = robot.get_pose()
        odom = OdomMsg(pose=Pose(x, y, t))
        slam._odom_callback(odom)

    slam._tick()  # 发布 /map → costmap 订阅回调
    costmap._inflate_and_publish()
    planner.latest_costmap = costmap.costmap
    controller.costmap = costmap.costmap


    # ── 初算路径 ──
    planner.set_pose(robot.x, robot.y)
    planner.set_goal(goal_x, goal_y)
    plan = planner.plan()
    if plan:
        controller.set_path(plan)
        log_ok(f"  初始路径: {len(plan)} 个航点 (A*)")
    else:
        log_warn("  A* 无路! 目标被挡住了, 继续尝试...")

    bt_triggered = False
    # ── 主控制循环 (50Hz) ──
    max_steps = 2000
    dt = 1.0 / 50
    reached = False
    dist_to_goal = 999.0

    for step in range(max_steps):
        # 1. 感知: 读传感器, 发布 /odom (30Hz = 隔帧)
        if step % 2 == 0:
            perception._sensor_callback()

        # 2. SLAM: 处理 odom, 建图 (10Hz)
        if step % 5 == 0:
            x, y, t = robot.get_pose()
            odom = OdomMsg(pose=Pose(x, y, t))
            slam._odom_callback(odom)

        # 3. Costmap: 收到新地图后膨胀
        if step % 20 == 0:
            slam._tick()  # 发布 /map
            costmap._inflate_and_publish()
            planner.latest_costmap = costmap.costmap
            controller.costmap = costmap.costmap


        # 4. 决策 + 重规划: 每 100 步 (2 秒) 检查
        if plan is None or step % 100 == 0:
            planner.set_pose(robot.x, robot.y)
            new_plan = planner.plan()
            if new_plan:
                plan = new_plan
                controller.set_path(plan)

        # 5. 控制: MPC + PID
        controller._control_loop()
        if controller._needs_replan:
            controller._needs_replan = False
            plan = None

        # 6. 检查到达
        dist_to_goal = math.sqrt((robot.x - goal_x)**2 + (robot.y - goal_y)**2)
        if dist_to_goal < 0.3:
            log_ok(f"\n🏁 到达工位! (误差 {dist_to_goal:.2f}m) 总步数={step}")
            decision.context["navigation_done"] = True
            reached = True
            break

    if not reached:
        log_warn(f"未到达目标 (最后距离 {dist_to_goal:.2f}m)")

    # 打印最终统计
    print_separator("最终统计")
    print(f"  总步数:          {step}")
    print(f"  到达目标:        {'✅ 是' if reached else '❌ 否'}")
    print(f"  最终距离:        {dist_to_goal:.2f}m")
    print(f"  最终位置:        ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f}rad)")
    print(f"  SLAM:           {slam.memory_summary()}")
    print(f"  Costmap:        {'✅' if costmap.costmap else '❌'}")
    if planner.latest_path:
        print(f"  A* 路径:         {len(planner.latest_path)} 航点")
    print(f"  控制步数:        {controller.n_control_steps}")
    if controller.track_error_history:
        import numpy as np
        errors = controller.track_error_history
        print(f"  跟踪误差:        平均={np.mean(errors):.3f}m, 最大={np.max(errors):.3f}m")

    print_separator("节点状态")
    print(f"  Perception:     ✅ (D435 -> /odom, /camera/*)")
    print(f"  SLAM:           ✅ (RTAB-Map -> /map, /pose)")
    print(f"  Costmap:        ✅ (inflation -> /costmap)")
    print(f"  Decision:       ✅ (Behavior Tree: {decision.state})")
    print(f"    -> BT triggered: {'YES - dynamic obstacle!' if bt_triggered else 'no (path was clear)'}")
    print(f"  Planner:        ✅ (A* -> /plan)")
    print(f"  Controller:     ✅ (MPC lateral + PID longitudinal -> /cmd_vel)")

    if bt_triggered:
        print(f"\n  🎯 Behavior Tree just did its job:")
        print(f"    1. Path blocked -> Condition 'PathOk?' returned FAILURE")
        print(f"    2. Fallback -> tried RecoverySpin branch")
        print(f"    3. A* replanned around the dynamic obstacle")
        print(f"    4. Robot continued on new path -> reached goal")

    return reached


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    args = sys.argv[1:]

    # 控制模式: --mode=split 或 --mode=unified
    control_mode = "split"
    for a in args:
        if a.startswith("--mode="):
            control_mode = a.split("=", 1)[1]
            assert control_mode in ("split", "unified"), \
                f"--mode must be 'split' or 'unified', got '{control_mode}'"

    if len(args) == 0 or "--help" in args:
        print("Usage: conda activate ros2_nav && python demo_navigation.py [--map|--explain|--run|--viz] [--mode=split|unified]")
        print("  --map         Static map (room layout)")
        print("  --explain     ROS2 dataflow + algorithm overview")
        print("  --run         Run navigation (terminal text)")
        print("  --viz         Run navigation (3-panel matplotlib window)")
        print("  --mode=split   Horizontal MPC + Vertical PID     (default, learn PID)")
        print("  --mode=unified Joint MPC samples (v,ω) together  (like Waymo)")
        sys.exit(0)

    if "--map" in args:
        demo_show_map()

    if "--explain" in args:
        demo_explain_architecture()

    if "--viz" in args:
        import numpy as np
        demo_run_navigation_viz(control_mode)

    elif "--run" in args:
        import numpy as np
        demo_run_navigation(control_mode)
