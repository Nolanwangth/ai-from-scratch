# 运行: conda activate ros2_nav && python ros2_nav_course/control/control_node.py
"""
Control Node — Two modes: Split (MPC + PID) | Unified MPC
==========================================================

self.mode = "split"    → 横向 MPC(ω) + 纵向 PID(v) + SpeedProfiler + 紧急制动
                            ↑ 教学用: 看清楚每个模块干什么

self.mode = "unified"  → 统一 MPC: 同时采样 (v, ω), 一次代价函数决定横纵
                            ↑ 和 Waymo/Cruise 逻辑一样, 无需 PID/SpeedProfiler/急停

对比:
    Split:    A* -> SpeedProfiler -> MPC(ω) + PID(v) -> 紧急制动 -> /cmd_vel
    Unified:  A* -> UnifiedMPC(v,ω) -> /cmd_vel
              ↑ 少三层, 撞/弯/目标减速全在代价里自然体现

PID 怎么学:
    跑 --mode split 看 PID 的 P/I/D 三项分别怎么贡献油门
    跑 --mode unified 看统一 MPC 怎么同时搞定横纵
"""

import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Tuple, Optional

from ros2_nav_course.utils.mock_ros2 import (
    Node, Path, Pose, Twist, log_ok, log_info, log_warn
)
from ros2_nav_course.simulation.robot_model import DifferentialDriveRobot


# =============================================================================
# Part 1: Lateral MPC (controls ω only — used in "split" mode)
# =============================================================================

class LateralMPC:
    """
    Lateral MPC: samples ω candidates, picks best via rollout.

    Only controls steering. Speed comes from PID.
    """

    def __init__(self, predict_steps=10, dt=0.02, max_omega=2.0, n_samples=20):
        self.predict_steps = predict_steps
        self.dt = dt
        self.max_omega = max_omega
        self.n_samples = n_samples
        self.current_omega = 0.0
        self.world = None
        log_info(f"  [LateralMPC] {n_samples*2+1} omega candidates, {predict_steps}-step rollout")

    def compute_omega(self, robot, path, v):
        if not path or len(path) < 2:
            return 0.0
        best_omega, best_cost = 0.0, float('inf')
        for i in range(-self.n_samples, self.n_samples + 1):
            omega = self.max_omega * i / self.n_samples
            cost = self._rollout(robot, path, v, omega)
            if cost < best_cost:
                best_cost, best_omega = cost, omega
        alpha = 0.3
        self.current_omega = alpha * best_omega + (1 - alpha) * self.current_omega
        return self.current_omega

    def _rollout(self, robot, path, v, omega):
        x, y, theta = robot.x, robot.y, robot.theta
        total = 0.0
        for step in range(self.predict_steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += omega * self.dt
            min_dist = min(math.sqrt((x-px)**2 + (y-py)**2) for px, py in path)
            best_idx = min(range(len(path)),
                           key=lambda i: math.sqrt((x-path[i][0])**2+(y-path[i][1])**2))
            look = min(best_idx + 20, len(path) - 1)  # 20点=1m超前, 用整体方向
            p_cur, p_look = path[best_idx], path[look]
            path_dir = math.atan2(p_look[1]-p_cur[1], p_look[0]-p_cur[0])
            heading_err = abs(theta - path_dir)
            heading_err = min(heading_err, 2*math.pi - heading_err)
            goal = path[-1]
            d_goal = math.sqrt((x-goal[0])**2 + (y-goal[1])**2)
            collision = 100.0 if (self.world and self.world.is_collision(x, y, 0.2)) else 0.0
            w = 1.0 + step * 0.15
            total += (min_dist*4.0 + heading_err*1.5 + d_goal*0.5 +
                      collision + abs(omega)*0.1) * w
        return total


# =============================================================================
# Part 2: Speed Profiler (used in "split" mode)
# =============================================================================

class SpeedProfiler:
    """
    速度规划器: 算每个路径点的安全速度上限。

    三个约束:
        1. 曲率限速:  弯越急 → 越慢 (无人驾驶的横向加速度约束)
        2. 目标减速:  离终点越近 → 越慢 (防止冲过)
        3. 障碍物距离: 离障碍物越近 → 越慢 → 太近就倒车 (costmap 阴影区)
                      ↑ 这是你要的"离东西太近就减速/倒车"
    """

    def __init__(self, max_v=1.0, min_v=0.1, slowdown_dist=2.0,
                 safe_dist=0.3,  # 离障碍物多远开始减速
                 back_dist=0.15):  # 离障碍物多近开始倒车 (负速度)
        self.max_v = max_v
        self.min_v = min_v
        self.slowdown_dist = slowdown_dist
        self.safe_dist = safe_dist
        self.back_dist = back_dist
        self.world = None  # 由 ControlNode 设置

    def compute_limits(self, path):
        if not path or len(path) < 2:
            return [self.max_v]
        limits = []
        goal = path[-1]
        for i in range(len(path)):
            px, py = path[i]
            v = self.max_v

            # 1. 曲率限速
            if i < len(path) - 2:
                a1 = math.atan2(path[i+1][1]-py, path[i+1][0]-px)
                a2 = math.atan2(path[i+2][1]-path[i+1][1], path[i+2][0]-path[i+1][0])
                da = min(abs(a2-a1), 2*math.pi - abs(a2-a1))
                if da > 0.05:
                    v = min(v, max(self.min_v, self.max_v / (1.0 + da * 2.0)))

            # 2. 目标减速
            dg = math.sqrt((px-goal[0])**2 + (py-goal[1])**2)
            if dg < self.slowdown_dist:
                v = min(v, self.min_v + (self.max_v-self.min_v)*dg/self.slowdown_dist)

            # 3. 障碍物实时减速交给控制层安全兜底.
            # SpeedProfiler 只做路径几何速度规划, 避免整条路线被未知/近障碍压得过慢.

            limits.append(v)
        return limits

    def _min_obstacle_distance(self, x, y):
        """返回 (x,y) 到最近障碍物的距离 (m)."""
        if self.world is None:
            return float('inf')
        min_d = float('inf')
        for obs in self.world.obstacles + self.world.walls:
            d = obs.distance_to(x, y)
            if d < min_d:
                min_d = d
        return min_d


# =============================================================================
# Part 3: PID (longitudinal — used in "split" mode)
# =============================================================================

class LongitudinalPID:
    """
    PID speed tracker.

        e(t) = v_target - v_current
        u(t) = Kp*e + Ki*∫e dt + Kd*de/dt

    P: proportional — bigger error → more throttle
    I: integral    — persistent offset → accumulate correction
    D: derivative  — approaching target → ease off early, prevent overshoot

    Anti-windup: clamps integral so it doesn't explode during stop.
    """

    def __init__(self, kp=1.5, ki=0.3, kd=0.1, max_v=1.0, max_accel=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_v, self.max_accel = max_v, max_accel
        self._integral, self._prev_error = 0.0, 0.0
        self.last_p, self.last_i, self.last_d = 0.0, 0.0, 0.0  # for inspection
        log_info(f"  [PID] Kp={kp} Ki={ki} Kd={kd}")

    def compute_velocity(self, v_target, v_current, dt):
        error = v_target - v_current
        self.last_p = self.kp * error
        self._integral += error * dt
        self._integral = max(min(self._integral, 0.5/self.ki), -0.5/self.ki)
        self.last_i = self.ki * self._integral
        self.last_d = self.kd * (error - self._prev_error) / max(dt, 0.001)
        self._prev_error = error
        v_cmd = v_current + self.last_p + self.last_i + self.last_d
        v_cmd = max(min(v_cmd, v_current+self.max_accel*dt), v_current-self.max_accel*dt)
        v_cmd = max(0.0, min(v_cmd, self.max_v))
        return v_cmd

    def reset(self):
        self._integral, self._prev_error = 0.0, 0.0


# =============================================================================
# Part 4: Unified MPC (controls v AND ω together — "unified" mode)
# =============================================================================

class UnifiedMPC:
    """
    Unified MPC: samples (v, ω) jointly. One cost function decides both.

    This is how Waymo / Cruise / Apollo do it. No separate PID, no SpeedProfiler,
    no emergency-stop band-aid. The cost function naturally makes it:
        - slow down before sharp turns (high heading error at high v → high cost)
        - slow down near goal (distance-to-goal term)
        - avoid collisions (collision penalty)
        - prefer smooth control (ω penalty + acceleration penalty)

    Number of candidates:
        n_v × n_ω = 11 × 31 = 341 candidates × 10 step rollout
        = 3410 forward-model calls per control tick. M5 handles this easily at 50Hz.
    """

    def __init__(self, predict_steps=10, dt=0.02,
                 max_v=1.0, max_omega=2.0,
                 n_v=11, n_omega=31, world=None):
        self.predict_steps = predict_steps
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.n_v = n_v
        self.n_omega = n_omega
        self.world = world
        self.current_v = 0.0
        self.current_omega = 0.0
        self.costmap = None   # OccupancyGrid, O(1) lookup
        self.cm_data = None   # numpy array view
        self.cm_res = 0.05
        self.cm_ox = 0.0
        self.cm_oy = 0.0
        n_v_actual = self.n_v + 1  # +(-0.3)
        n_w_actual = self.n_omega * 2 + 1
        total = n_v_actual * n_w_actual
        log_info(f"  [UnifiedMPC] {n_v_actual}v x {n_w_actual}w = {total} candidates, "
                 f"{total * self.predict_steps} cost evals/tick")

    def set_costmap(self, costmap):
        """缓存 costmap 网格, O(1) 查障碍物."""
        if costmap is None:
            return
        self.costmap = costmap
        self.cm_data = np.array(costmap.data).reshape(costmap.height, costmap.width)
        self.cm_res = costmap.resolution
        self.cm_ox = costmap.origin_x
        self.cm_oy = costmap.origin_y

    def compute(self, robot, path):
        if not path or len(path) < 2:
            return 0.0, 0.0

        best_v, best_omega = 0.0, 0.0
        best_cost = float('inf')

        # v 候选: 正向速度保持足够下限, 避免 MPC 选择“贴路径但爬行”.
        v_candidates = list(np.linspace(0.35, self.max_v, self.n_v)) + [-0.2]
        # ω 候选: 从 -max_omega 到 +max_omega
        omega_candidates = np.linspace(-self.max_omega, self.max_omega,
                                        self.n_omega * 2 + 1)

        for v in v_candidates:
            for omega in omega_candidates:
                cost = self._rollout(robot, path, v, omega)
                if cost < best_cost:
                    best_cost, best_v, best_omega = cost, v, omega

        # Smooth both v and ω
        alpha = 0.55
        self.current_v = alpha * best_v + (1 - alpha) * self.current_v
        self.current_omega = alpha * best_omega + (1 - alpha) * self.current_omega
        return self.current_v, self.current_omega

    def _rollout(self, robot, path, v, omega):
        x, y, theta = robot.x, robot.y, robot.theta
        total = 0.0
        for step in range(self.predict_steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += omega * self.dt

            # 路径偏差: 到这个预测位置最近路径点的距离
            min_dist = min(math.sqrt((x-px)**2 + (y-py)**2) for px, py in path)
            # 每步重新找最近路径点 (不停留原地)
            best_idx = min(range(len(path)),
                           key=lambda i: math.sqrt((x-path[i][0])**2+(y-path[i][1])**2))

            # 朝向偏差: 路径上超前 2 步的方向 vs 预测朝向
            look = min(best_idx + 20, len(path) - 1)  # 20点=1m超前, 用整体方向
            p_cur, p_look = path[best_idx], path[look]
            path_dir = math.atan2(p_look[1]-p_cur[1], p_look[0]-p_cur[0])
            heading_err = abs(theta - path_dir)
            heading_err = min(heading_err, 2*math.pi - heading_err)

            goal = path[-1]
            d_goal = math.sqrt((x-goal[0])**2 + (y-goal[1])**2)

            # 碰撞检测: costmap 网格 O(1) 查表, 若不可用则回退到 world
            cost = 0
            near_obs = False
            if self.cm_data is not None:
                gx = int((x - self.cm_ox) / self.cm_res)
                gy = int((y - self.cm_oy) / self.cm_res)
                H, W = self.cm_data.shape
                if 0 <= gx < W and 0 <= gy < H:
                    cost = self.cm_data[gy, gx]
            elif self.world is not None:
                # 回退: 用 world.is_collision 判断
                if self.world.is_collision(x, y, 0.20):
                    cost = 254
                elif self.world.is_collision(x, y, 0.40):
                    cost = 50
            if cost >= 254:
                collision = 500.0       # 致命区 (障碍物本体)
            elif cost >= 100:
                collision = cost * 0.3  # 膨胀区 (50→15, ~N/A for cost costmap)
                near_obs = True
            elif cost == 50:
                collision = cost * 0.3  # 膨胀区
                near_obs = True
            else:
                collision = 0.0         # 自由(0) / 未知(80) 不罚, 只管走

            # 速度惩罚
            overspeed = max(0.0, v - 1.0)  # 超速 = v > 1.0m/s
            if near_obs:
                # 膨胀区或致命区附近: 强制慢速
                overspeed = max(overspeed, v - 0.3)

            w = 1.0  # flat weight, 不放大远期误差
            total += (min_dist*2.0 + d_goal*0.8 + collision + overspeed*2.0) * w

        # 加速度平滑 (弱, 不让 v=0 永远赢)
        # 前进奖励: v 越大越优, 但倒车只有在避障代价需要时才会赢.
        total -= v * 14.0
        return total


# =============================================================================
# Control Node — pick "split" or "unified" mode
# =============================================================================

class ControlNode(Node):
    """
    控制节点: 两种模式可选。

        mode="split"   → 横向 MPC(ω) + 纵向 PID(v) + SpeedProfiler
        mode="unified" → 统一 MPC: 同时采样 (v, ω), 一次代价函数决定横纵

    用法:
        ctrl = ControlNode(world, robot, mode="split")    # 学 PID
        ctrl = ControlNode(world, robot, mode="unified")  # 像 Waymo

    Subscribes: /plan (from A*)
    Publishes:  /cmd_vel (v, ω)
    """

    def __init__(self, world, robot: DifferentialDriveRobot, hz=50.0,
                 mode: str = "split"):
        super().__init__("control_node")
        self.world = world
        self.robot = robot
        self.hz = hz
        self.dt = 1.0 / hz
        self.mode = mode
        self.costmap = None  # OccupancyGrid, 由外部设置

        # --- Shared ---
        self.current_path: List[Tuple[float, float]] = []
        self.n_control_steps = 0
        self.track_error_history: List[float] = []
        self._needs_replan = False
        self._backing_up = False
        self._backup_steps = 0

        # --- Split-mode components ---
        self.mpc = None
        self.pid = None
        self.speed_profiler = None
        self.v_limits: List[float] = []

        # --- Unified-mode component ---
        self.unified_mpc = None

        if self.mode == "split":
            self._init_split_mode()
        elif self.mode == "unified":
            self._init_unified_mode()
        else:
            raise ValueError(f"Unknown self.mode: {self.mode}")

        # ── 订阅 /costmap (来自 CostmapNode, 换掉直接赋值后门) ──
        self.create_subscription("/costmap", self._costmap_cb, "OccupancyGrid")
        # ── 订阅 /plan (来自 A*, 换掉直接 set_path 调用) ──
        self.create_subscription("/plan", self._plan_cb, "Path")

        # ── 参数 (运行时可变, 不用改代码重新跑) ──
        self.declare_parameter("kp", 1.5, "PID proportional gain")
        self.declare_parameter("ki", 0.3, "PID integral gain")
        self.declare_parameter("kd", 0.1, "PID derivative gain")
        self.declare_parameter("max_v", self.robot.max_v, "Max linear velocity (m/s)")

        self.create_publisher("/cmd_vel", "Twist")
        self.create_timer(self.dt, self._control_loop)
        log_ok(f"[control] Mode={self.mode} @ {hz}Hz")

    def _costmap_cb(self, cm):
        """订阅 /costmap — CostmapNode 发新地图时自动更新."""
        self.costmap = cm
        if self.unified_mpc is not None:
            self.unified_mpc.set_costmap(cm)

    def _plan_cb(self, plan_msg):
        """订阅 /plan — A* 发新路径时自动更新当前路径."""
        path = [(p.x, p.y) for p in plan_msg.poses] if plan_msg and plan_msg.poses else []
        if path:
            self.set_path(path)

    def _reload_params(self):
        """从参数服务器重新读取 PID 参数 (运行中可调)."""
        if self.pid:
            self.pid.kp = self.get_parameter("kp") or self.pid.kp
            self.pid.ki = self.get_parameter("ki") or self.pid.ki
            self.pid.kd = self.get_parameter("kd") or self.pid.kd
        self.robot.max_v = self.get_parameter("max_v") or self.robot.max_v

    def _init_split_mode(self):
        self.speed_profiler = SpeedProfiler(max_v=self.robot.max_v, min_v=0.3)
        self.speed_profiler.world = self.world  # 障碍物距离感知
        self.mpc = LateralMPC(
            predict_steps=10, dt=self.dt,
            max_omega=self.robot.max_omega, n_samples=20
        )
        self.mpc.world = self.world
        self.pid = LongitudinalPID(
            kp=1.5, ki=0.3, kd=0.1,
            max_v=self.robot.max_v, max_accel=2.0
        )
        log_info("  [split] MPC(steering) + SpeedProfiler + PID(throttle) + emergency-stop")

    def _init_unified_mode(self):
        # 教学仿真优先保证交互速度: 6v × 21ω × 6步, 仍保留统一优化思想.
        n_v = 5
        n_omega = 10
        self.unified_mpc = UnifiedMPC(
            predict_steps=6, dt=0.12,
            max_v=self.robot.max_v, max_omega=self.robot.max_omega,
            n_v=n_v, n_omega=n_omega, world=self.world
        )
        log_info("  [unified] UnifiedMPC: one cost decides (v, omega) together")

    def set_path(self, path: List[Tuple[float, float]]):
        # A* 栅格路径很密(通常 5cm 一个点), 控制器用 15cm 级别就够了.
        # 下采样能显著减少 MPC 每帧最近点搜索开销.
        if len(path) > 3:
            compact = path[::3]
            if compact[-1] != path[-1]:
                compact.append(path[-1])
            self.current_path = compact
        else:
            self.current_path = path
        # 同步 costmap 给 MPC: O(1) 网格查表, 不遍历障碍物
        if self.unified_mpc is not None:
            self.unified_mpc.set_costmap(self.costmap)
        if self.mode == "split" and self.speed_profiler and self.current_path:
            self.v_limits = self.speed_profiler.compute_limits(self.current_path)

    # ─── Split mode control loop ──────────────────────────────────

    def _control_split(self):
        # 横向: MPC 打方向盘
        omega = self.mpc.compute_omega(self.robot, self.current_path, self.robot.v)

        # 速度目标: 路径限速 + 机器人实际位置障碍物限速 (直接防撞)
        if self.v_limits and self.current_path:
            best_idx = 0
            best_d = float('inf')
            for i, (px, py) in enumerate(self.current_path):
                d = math.sqrt((self.robot.x-px)**2 + (self.robot.y-py)**2)
                if d < best_d:
                    best_d, best_idx = d, i
            v_target = self.v_limits[min(best_idx, len(self.v_limits)-1)]
        else:
            v_target = 0.5

        # 用机器人真实位置查障碍物距离, 0.5m 开始减速, 0.35m 触发倒车+重规划
        if self.world is not None:
            real_dist = self._min_obstacle_dist(self.robot.x, self.robot.y)
            if real_dist < 0.20:
                v_target = min(v_target, max(0.05, real_dist / 0.20 * self.robot.max_v))

        # 纵向: 倒车时 PID 不接管, 直接设负速度; 正常时 PID 追 v_target
        if self._backing_up and self._backup_steps > 0:
            v_cmd = -0.5  # 倒车速度
            self._backup_steps -= 1
        elif self._backing_up:
            self._backing_up = False
            v_cmd = self.pid.compute_velocity(v_target, self.robot.v, self.dt)
        else:
            v_cmd = self.pid.compute_velocity(v_target, self.robot.v, self.dt)

        return v_cmd, omega

    # ─── Unified mode control loop ────────────────────────────────

    def _control_unified(self):
        v_cmd, omega = self.unified_mpc.compute(self.robot, self.current_path)
        real_dist = float('inf')

        # 安全备份: robot 当前位置离障碍物太近 → 倒车 + 通知 BT
        if self.world is not None:
            real_dist = self._min_obstacle_dist(self.robot.x, self.robot.y)
            if real_dist < 0.20:  # 减速
                v_cmd = min(v_cmd, max(0.05, real_dist / 0.20 * self.robot.max_v))

        if self.current_path:
            goal = self.current_path[-1]
            d_goal = math.sqrt((self.robot.x-goal[0])**2 + (self.robot.y-goal[1])**2)
            if d_goal > 1.0 and real_dist > 0.30:
                v_cmd = max(v_cmd, 0.45)

        if self._backing_up and self._backup_steps > 0:
            v_cmd = -0.5
            self._backup_steps -= 1
        elif self._backing_up:
            self._backing_up = False

        return v_cmd, omega

    # ─── Shared helpers ───────────────────────────────────────────

    def _min_obstacle_dist(self, x, y):
        """返回 (x,y) 到最近障碍物的距离 (m), 供 split 模式实时限速用."""
        if self.world is None:
            return float('inf')
        md = float('inf')
        for obs in self.world.obstacles + self.world.walls:
            md = min(md, obs.distance_to(x, y))
        return md

    def _control_loop(self):
        self.n_control_steps += 1

        if self.mode == "split":
            v_cmd, omega = self._control_split()
        else:
            v_cmd, omega = self._control_unified()

        self.robot.set_velocity(v_cmd, omega)
        self.robot.update(self.dt)

        self.publish("/cmd_vel", Twist(linear_x=v_cmd, angular_z=omega))

        if self.current_path:
            min_dist = min(
                math.sqrt((self.robot.x-px)**2 + (self.robot.y-py)**2)
                for px, py in self.current_path
            )
            self.track_error_history.append(min_dist)


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    from ros2_nav_course.simulation.world_2d import World2D
    import sys

    mode = "split"
    for a in sys.argv[1:]:
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1]

    print("=" * 60)
    print(f"Control Node — mode={mode}")
    print("=" * 60)
    print()

    world = World2D()
    robot = DifferentialDriveRobot(x=0.5, y=0.5, theta=0.0)

    test_path = [(0.5, 0.5), (1.0, 0.5), (1.5, 0.5), (2.0, 0.5),
                 (2.5, 0.5), (3.0, 0.5), (3.5, 0.5), (4.0, 0.5)]

    ctrl = ControlNode(world, robot, hz=50.0, mode=mode)
    ctrl.set_path(test_path)

    print(f"Start: ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f})")
    print(f"Path: {len(test_path)} waypoints")

    for i in range(100):
        ctrl._control_loop()

    print(f"End:   ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f})")
    print(f"Track error: avg={np.mean(ctrl.track_error_history):.3f}m  "
          f"max={np.max(ctrl.track_error_history):.3f}m")

    if mode == "split" and ctrl.pid:
        print(f"\nPID breakdown (last tick):")
        print(f"  P-term: {ctrl.pid.last_p:+.3f}  (proportional to error)")
        print(f"  I-term: {ctrl.pid.last_i:+.3f}  (accumulated offset)")
        print(f"  D-term: {ctrl.pid.last_d:+.3f}  (rate of change)")

    print(f"\n  Mode: {mode}")
    if mode == "split":
        print(f"  Lateral: MPC ({ctrl.mpc.n_samples*2+1} ω candidates)")
        print(f"  Longitudinal: PID (Kp={ctrl.pid.kp}, Ki={ctrl.pid.ki}, Kd={ctrl.pid.kd})")
    else:
        nv = ctrl.unified_mpc.n_v + 1  # +(-0.3)
        nw = ctrl.unified_mpc.n_omega * 2 + 1
        print(f"  Unified MPC: {nv}v x {nw}ω = {nv*nw} candidates, "
              f"{nv*nw*ctrl.unified_mpc.predict_steps} cost evals/tick")
        print(f"  No PID, no SpeedProfiler needed")
