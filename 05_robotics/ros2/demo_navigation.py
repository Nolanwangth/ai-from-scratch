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


def demo_run_navigation(control_mode="split", use_viz=False):
    """跑完整导航。use_viz=True 弹出 matplotlib 三面板可视化窗口。"""
    print_banner()

    world = World2D()
    robot = DifferentialDriveRobot(x=1.0, y=1.0, theta=0.0)
    robot.max_v = 1.0
    robot.max_omega = 2.0

    # ── 初始化所有节点 ──
    perception = PerceptionNode(world, robot, hz=30.0)
    slam = RTABMapNode(world, robot, keyframe_min_dist=0.3, hz=10.0)
    costmap = CostmapNode(robot_radius=0.3, inflation_radius=0.5)
    decision = DecisionNode(world)
    # NOTE: BT 恢复动作(BACKUP/SPIN)需要控制小车, 保留 robot 引用
    #       真 ROS2 里这会是一个 /cmd_vel Service, 不直接传引用
    decision.context["robot"] = robot
    planner = AStarPlannerNode(world)
    # NOTE: planner 和 controller 通过订阅 /costmap 和 /plan 获取数据,
    #       不再需要 planner.latest_costmap = ... 这种后门赋值
    controller = ControlNode(world, robot, hz=50.0, mode=control_mode)

    # ── 设置目标 (通过 Action: /navigate_to_goal) ──
    goal_x, goal_y = world.landmarks["workstation"]
    decision.set_goal(goal_x, goal_y)

    # ── 演示: Service + Action + Parameter ──
    # Service: 调用 SLAM 的 /get_map
    map_client = planner.create_client("/get_map")
    _ = map_client.call(None)  # 演示 Service 调用
    # Action: 通过 /navigate_to_goal 发送目标
    nav_client = decision.create_action_client("/navigate_to_goal")
    # Parameter: 控制节点声明了 kp/ki/kd/max_v, 运行中可改
    controller.set_parameter("kp", 1.5)  # 演示运行时改参数

    # ── 可视化 (可选) ──
    viz = None
    if use_viz:
        viz = NavigationVisualizer(world)
        viz.goal = (goal_x, goal_y)
        log_ok("可视化窗口已打开")

    # ── 初始地图 — 原地扫 5 帧建前方视野, 边走边继续建图 ──
    from ros2_nav_course.utils.mock_ros2 import Odometry as OdomMsg
    for _ in range(5):
        perception._sensor_callback()
        slam._tick()
    costmap._inflate_and_publish()
    if controller.unified_mpc and controller.unified_mpc.cm_data is None:
        controller.unified_mpc.set_costmap(costmap.costmap)
    # 初始 SLAM 地图推给可视化 (显示扫描后的扇形覆盖)
    if viz:
        viz.update_slam_map(slam._grid)

    # ── 初算路径: planner 通过 /pose_corrected 获 SLAM 估计位姿 (不是 true robot pose) ──
    planner.set_goal(goal_x, goal_y)
    plan = planner.plan()
    if plan and not use_viz:
        log_ok(f"  初始路径: {len(plan)} 个航点 (A*) — 基于 SLAM 估计位姿")

    # ── 主控制循环 (50Hz) ──
    max_steps = 2000
    reached = False
    bt_triggered = False

    for step in range(max_steps):
        # 感知: 发布 /odom on topic → SLAM 订阅
        if step % 2 == 0:
            perception._sensor_callback()

        # 动态障碍物: 第 400 步在路径前方掉落一个 BOX
        if step == 400:
            from ros2_nav_course.simulation.world_2d import Obstacle
            idx = min(len(plan) * 2 // 3, len(plan) - 1) if plan else 0
            bx, by = plan[idx] if plan else (5.0, 5.0)
            if math.sqrt((bx-robot.x)**2 + (by-robot.y)**2) < 1.5:
                bx, by = robot.x + 2.0, robot.y + 2.0
            world.obstacles.append(Obstacle(bx, by, 1.0, 1.0, "BOX!"))
            slam._tick()
            costmap._inflate_and_publish()
            if viz:
                viz.update_costmap(costmap.costmap)
                viz.refresh_obstacles()
            log_warn(f"  ⚡ 动态障碍物出现在 ({bx:.1f}, {by:.1f})!")
            plan = None
            decision.context["path_blocked"] = True

        # SLAM 建图 + Costmap 更新
        if step % 20 == 0:
            slam._tick()
            costmap._inflate_and_publish()
            # costmap 发布 /costmap → planner+controller 通过订阅自动获取

        # 重规划: path_blocked 时立即, 否则每 2 秒维护
        if plan is None or step % 100 == 0:
            was_blocked = decision.context.get("path_blocked", False)
            if was_blocked:
                decision.bt_root.tick(decision.context)
                bt_triggered = True
            else:
                decision.context["recovery_count"] = 0

            # planner 通过 /pose_corrected 获取 SLAM 估计位姿
            new_plan = planner.plan(verbose=was_blocked)  # publishes /plan → controller._plan_cb
            if new_plan:
                plan = new_plan
                decision.context["path_blocked"] = False
                if was_blocked:
                    log_ok(f"  ↻ 重规划: {len(plan)} 航点 (第{decision.context.get('recovery_count',0)}次恢复)")
            elif was_blocked:
                log_warn("  ⚠ A* 仍无路!")

        controller._control_loop()

        # 控制层触发紧急重规划 (障碍物太近)
        if controller._needs_replan:
            controller._needs_replan = False
            decision.context["path_blocked"] = True
            plan = None

        # 可视化渲染 (~6Hz)
        if viz and step % 8 == 0:
            if costmap.costmap is not None:
                viz.update_costmap(costmap.costmap)
            # SLAM 建图过程: 把原始栅格发给可视化, 显示逐步 reveal 效果
            if slam._grid is not None:
                viz.update_slam_map(slam._grid)
            kf_list = [(kf.x, kf.y) for kf in slam.wm.values()]
            kf_list += [(kf.x, kf.y) for kf in slam.stm]
            viz.draw(robot.x, robot.y, robot.theta, robot.v, robot.omega,
                     path=plan, kfs=kf_list, n_loops=slam.n_loops,
                     wm_size=len(slam.wm), goal=(goal_x, goal_y), step=step)

        dist_to_goal = math.sqrt((robot.x - goal_x)**2 + (robot.y - goal_y)**2)
        if dist_to_goal < 0.3:
            if viz:
                viz.draw(robot.x, robot.y, robot.theta, robot.v, robot.omega,
                         path=plan, kfs=[], n_loops=slam.n_loops,
                         wm_size=len(slam.wm), goal=(goal_x, goal_y), step=step)
                viz.finalize(True)
            log_ok(f"\n🏁 到达工位! (误差 {dist_to_goal:.2f}m) 步数={step}")
            reached = True
            break

    if not reached:
        if viz:
            viz.finalize(False)
        log_warn(f"未到达 (距离 {dist_to_goal:.2f}m)")

    # ── 最终统计 (终端模式打印详细表格) ──
    if not use_viz:
        print_separator("最终统计")
        print(f"  总步数:          {step}")
        print(f"  到达目标:        {'✅ 是' if reached else '❌ 否'}")
        print(f"  最终距离:        {dist_to_goal:.2f}m")
        print(f"  最终位置:        ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f}rad)")
        print(f"  SLAM:           {slam.memory_summary()}")
        if planner.latest_path:
            print(f"  A* 路径:         {len(planner.latest_path)} 航点")
        print(f"  控制步数:        {controller.n_control_steps}")
        if controller.track_error_history:
            import numpy as np
            errors = controller.track_error_history
            print(f"  跟踪误差:        平均={np.mean(errors):.3f}m, 最大={np.max(errors):.3f}m")
        print_separator("节点状态")
        print(f"  Perception:     ✅  /odom, /camera/*")
        print(f"  SLAM:           ✅  /map, /pose (STM→WM→LTM)")
        print(f"  Costmap:        ✅  /costmap (膨胀)")
        print(f"  Decision:       ✅  BT: {decision.state}")
        print(f"  Planner:        ✅  A* → /plan")
        print(f"  Controller:     ✅  {control_mode} → /cmd_vel")
        if bt_triggered:
            print(f"\n  🎯 BT 触发: 路径被堵 → 重规划 → 绕行成功")

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
        demo_run_navigation(control_mode, use_viz=True)

    elif "--run" in args:
        import numpy as np
        demo_run_navigation(control_mode, use_viz=False)
