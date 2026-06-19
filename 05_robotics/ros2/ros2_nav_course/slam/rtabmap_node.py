# 运行: conda activate ros2_nav && python ros2_nav_course/slam/rtabmap_node.py
"""
SLAM 节点 — RTAB-Map 原理 + 模拟实现
======================================
教学版 RTAB-Map SLAM。

输入:  /odom (里程计), /camera/rgb, /camera/depth
输出:  /map (占据栅格), /tf (map→base_link), /pose_estimate

核心教学: RTAB-Map 的三级记忆 (STM→WM→LTM)
         + 关键帧选择 + 图优化 (简化)

算法流程:
    1. 前端: 检查是否关键帧 → 是就加入 STM
    2. STM 满了 → 转去 WM
    3. WM 满了 → 转去 LTM
    4. 回环: 只在 WM 里搜相似关键帧
"""

import math
import random
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Tuple, Optional

from ros2_nav_course.utils.mock_ros2 import (
    Node, Pose, OccupancyGrid, Odometry, log_info, log_ok, log_warn
)

# =============================================================================
# 关键帧 (KeyFrame): RTAB-Map 的"记忆单元"
# =============================================================================

class KeyFrame:
    """SLAM 里的一帧记忆"""

    def __init__(self, id: int, x: float, y: float, theta: float,
                 landmarks: Optional[Dict] = None):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
        self.landmarks = landmarks or {}
        self.weight = 1.0       # 重要性 (越高越不容易被踢出 WM)

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)

    def similarity(self, other: 'KeyFrame') -> float:
        """两个关键帧的相似度 (0~1)"""
        dist = self.distance_to(other.x, other.y)
        # 距离越近越像, 1m 以内 >0.7
        return math.exp(-dist * 0.5)


class RTABMapNode(Node):
    """
    RTAB-Map 教学版 — 三级记忆 SLAM。

    订阅:
        /odom: 带噪声的里程计

    发布:
        /map:         占据栅格 (建图)
        /pose_corrected: SLAM 修正后的位姿
        /loop_detected: 回环检测事件

    参数:
        stm_size:   短期记忆 STM 大小 (默认 10)
        wm_size:    工作记忆 WM 大小 (默认 50)
        keyframe_min_dist: 创建关键帧的最小距离 (默认 0.5m)
        loop_threshold: 回环检测阈值 (默认 0.6)
    """

    def __init__(self, world, robot,
                 stm_size: int = 10, wm_size: int = 50,
                 keyframe_min_dist: float = 0.5,
                 loop_threshold: float = 0.6,
                 hz: float = 30.0, resolution: float = 0.05):
        super().__init__("rtabmap_slam_node")
        self.world = world
        self.robot = robot

        # ── 三级记忆 ──
        self.stm: List[KeyFrame] = []
        self.wm:  Dict[int, KeyFrame] = {}
        self.ltm: Dict[int, KeyFrame] = {}
        self.stm_max = stm_size
        self.wm_max = wm_size
        self.next_id = 0

        # ── 参数 ──
        self.keyframe_min_dist = keyframe_min_dist
        self.loop_threshold = loop_threshold
        self.last_kf_pos: Optional[Tuple[float, float]] = None

        # ── 统计 + 位姿估计 ──
        self.n_keyframes = 0
        self.n_loops = 0
        self._true_pose = (0.0, 0.0, 0.0)       # 真实位姿 (上帝视角)
        self._drifted_pose = (0.0, 0.0, 0.0)     # SLAM 估计位姿 (带漂移)
        self._drift_x = 0.0                       # 累积漂移量
        self._drift_y = 0.0
        self._drift_theta = 0.0
        self._last_true_pose = (0.0, 0.0, 0.0)
        # 漂移参数: 教学 demo 保留“误差会累积”的 SLAM 语义,
        # 但默认值不能大到把规划坐标系和真实机器人坐标系撕开.
        self._drift_trans_ratio = 0.008
        self._drift_rot_ratio = 0.015

        # ── 增量建图 — 地图随视野逐步 reveal, 不是一次性真值 ──
        self.resolution = resolution
        self.map_W = int(self.world.width / resolution) + 1
        self.map_H = int(self.world.height / resolution) + 1
        self.map_ox = 0.0
        self.map_oy = 0.0
        # 占据栅格: -1=未知, 0=自由, 100=占据
        self._grid = np.full((self.map_H, self.map_W), -1, dtype=np.int8)
        # D435 模拟参数
        self.d435_fov_deg = 87.0        # D435 水平视场角
        self.d435_max_range = 10.0      # 教学仿真里覆盖 10m 房间
        self.d435_rays = 180            # 每帧射线数
        self._depth_noise_std = 0.03    # 深度噪声标准差 (m)
        self._obs_noise_prob = 0.02     # 2% 概率把自由格子误报为占据

        # ── 发布者 ──
        self.create_publisher("/map", "OccupancyGrid")
        self.create_publisher("/pose_corrected", "Pose")
        self.create_publisher("/loop_detected", "any")

        # ── 订阅 /odom ──
        self.create_subscription("/odom", self._odom_callback, "Odometry")
        self.create_timer(1.0 / hz, self._tick)

        # ── Service: /get_map — 别的 Node 随时查询当前地图 ──
        self._latest_grid = None
        self.create_service("/get_map", self._get_map_handler)

        log_ok(f"[SLAM] RTAB-Map ready (STM={stm_size}, WM={wm_size}, "
               f"incremental map {self.map_W}x{self.map_H})")

    def _get_map_handler(self, _request):
        """Service handler: 返回最新的占据栅格."""
        return dict(grid=self._latest_grid) if self._latest_grid else dict(error="no map yet")

    def _odom_callback(self, odom: Odometry):
        """收到里程计数据 → 累积漂移 + 关键帧 + 回环检测."""
        x, y, theta = odom.pose.x, odom.pose.y, odom.pose.theta

        # 第一帧 odom: 初始化, 不累积漂移
        if self._last_true_pose == (0.0, 0.0, 0.0):
            self._last_true_pose = (x, y, theta)
            self._true_pose = (x, y, theta)
            self._drifted_pose = (x, y, theta)
            return

        self._true_pose = (x, y, theta)
        lx, ly, lt = self._last_true_pose

        # 累积 odom 漂移: 误差随运动量增长
        dx = x - lx
        dy = y - ly
        dtheta = theta - lt
        dist = math.sqrt(dx**2 + dy**2)
        self._drift_x += dist * self._drift_trans_ratio * (1.0 if random.random() > 0.5 else -1.0)
        self._drift_y += dist * self._drift_trans_ratio * (1.0 if random.random() > 0.5 else -1.0)
        if abs(dtheta) > 0.01:
            self._drift_theta += abs(dtheta) * self._drift_rot_ratio * (1.0 if random.random() > 0.5 else -1.0)

        # SLAM 估计位姿 = 真值 + 累积漂移 (漂移量累加, 不是每帧独立噪声)
        dx_drifted = odom.pose.x + self._drift_x
        dy_drifted = odom.pose.y + self._drift_y
        dt_drifted = odom.pose.theta + self._drift_theta
        self._drifted_pose = (dx_drifted, dy_drifted, dt_drifted)
        self._last_true_pose = self._true_pose

        # Step 1: 关键帧判定 (基于漂移后的估计位置)
        if self._is_keyframe(dx_drifted, dy_drifted):
            kf = KeyFrame(self.next_id, dx_drifted, dy_drifted, dt_drifted)
            # 存储真值和漂移位置用于回环修正
            kf._true_pose = self._true_pose
            kf._drifted_pose = self._drifted_pose
            self.next_id += 1
            self.n_keyframes += 1

            # Step 2: 加入 STM
            self.stm.append(kf)
            self.last_kf_pos = (dx_drifted, dy_drifted)

            # Step 3: STM→WM 转移
            self._manage_memory()

            # Step 4: 回环检测
            self._detect_loop(kf)

    def _is_keyframe(self, x: float, y: float) -> bool:
        """判断是否创建关键帧"""
        if self.last_kf_pos is None:
            return True
        return math.sqrt((x - self.last_kf_pos[0])**2 +
                         (y - self.last_kf_pos[1])**2) > self.keyframe_min_dist

    def _manage_memory(self):
        """三级记忆管理"""
        # STM → WM
        while len(self.stm) > self.stm_max:
            oldest = self.stm.pop(0)
            self.wm[oldest.id] = oldest

        # WM → LTM (踢掉最不重要的)
        while len(self.wm) > self.wm_max:
            min_id = min(self.wm.keys(), key=lambda k: self.wm[k].weight)
            self.ltm[min_id] = self.wm.pop(min_id)

    def _detect_loop(self, new_kf: KeyFrame):
        """回环检测: 只在 WM 里搜. 命中则修正 drift (模拟图优化)"""
        best_sim = 0
        best_kf = None
        for kf in self.wm.values():
            if kf.id == new_kf.id:
                continue
            sim = new_kf.similarity(kf)
            if sim > best_sim:
                best_sim = sim
                best_kf = kf

        if best_sim > self.loop_threshold and best_kf is not None:
            self.n_loops += 1
            if best_kf.id in self.ltm:
                self.wm[best_kf.id] = self.ltm.pop(best_kf.id)

            # 模拟图优化: 修正 drifts
            # 回环匹配的旧关键帧记录了自己的真值位置, 用这个来纠正累积误差
            if hasattr(best_kf, '_true_pose'):
                tx, ty, tt = best_kf._true_pose
                correction_x = tx - new_kf._drifted_pose[0] if hasattr(new_kf, '_drifted_pose') else 0
                correction_y = ty - new_kf._drifted_pose[1] if hasattr(new_kf, '_drifted_pose') else 0
                correction_t = tt - new_kf._drifted_pose[2] if hasattr(new_kf, '_drifted_pose') else 0
                self._drift_x += correction_x * 0.7  # 70% 修正, 不完全拉回
                self._drift_y += correction_y * 0.7
                self._drift_theta += correction_t * 0.7

            self.publish("/loop_detected",
                         f"回环! {new_kf.id}↔{best_kf.id} (sim={best_sim:.2f})")
            new_kf.weight *= 1.2

    def _tick(self):
        """增量建图: 真值位姿测量 + 估计位姿插图 (物理正确的 SLAM 模拟)."""
        # 真值位姿产生传感器测量 (物理: 传感器看到的世界是真实世界)
        tx, ty, ttheta = self._true_pose
        # 估计位姿把测量插入地图 (SLAM: 我不知道自己在哪, 往漂移位姿写)
        rx, ry, rtheta = self._drifted_pose

        angles = np.linspace(-self.d435_fov_deg/2, self.d435_fov_deg/2, self.d435_rays)
        for a_deg in angles:
            # ── 从真值位姿发射射线, 测量真实深度 ──
            true_a = ttheta + math.radians(a_deg)
            tdx, tdy = math.cos(true_a), math.sin(true_a)
            step = self.resolution
            hit_dist = self.d435_max_range
            hit = False
            for s in np.arange(step, self.d435_max_range + step, step):
                wx = tx + s * tdx
                wy = ty + s * tdy
                if not (0 <= wx < self.world.width and 0 <= wy < self.world.height):
                    break
                if self.world.is_collision(wx, wy, 0.05):
                    hit_dist = s + random.gauss(0, self._depth_noise_std)
                    hit = True
                    break

            # ── 把测量插入地图时, 使用估计位姿 (漂移后) ──
            est_a = rtheta + math.radians(a_deg)
            edx, edy = math.cos(est_a), math.sin(est_a)

            # 射线经过的格子标记自由 (基于估计位姿)
            for s in np.arange(step, hit_dist, step):
                gx = int((rx + s*edx - self.map_ox) / self.resolution)
                gy = int((ry + s*edy - self.map_oy) / self.resolution)
                if 0 <= gx < self.map_W and 0 <= gy < self.map_H:
                    if self._grid[gy, gx] == -1:
                        self._grid[gy, gx] = 0  # 自由

            # 命中点标记占据 (基于估计位姿)
            if hit:
                hit_gx = int((rx + hit_dist*edx - self.map_ox) / self.resolution)
                hit_gy = int((ry + hit_dist*edy - self.map_oy) / self.resolution)
                if 0 <= hit_gx < self.map_W and 0 <= hit_gy < self.map_H:
                    self._grid[hit_gy, hit_gx] = 0 if random.random() < self._obs_noise_prob else 100

        # 发布当前部分地图
        og = OccupancyGrid(
            width=self.map_W, height=self.map_H,
            resolution=self.resolution,
            origin_x=self.map_ox, origin_y=self.map_oy,
            data=list(self._grid.flatten())
        )
        self._latest_grid = og
        self.publish("/map", og)

        self.publish("/pose_corrected", Pose(rx, ry, rtheta))

    def memory_summary(self) -> str:
        return (f"STM={len(self.stm)} | WM={len(self.wm)} | "
                f"LTM={len(self.ltm)} | KFs={self.n_keyframes} | "
                f"Loops={self.n_loops}")


if __name__ == "__main__":
    from ros2_nav_course.simulation.world_2d import World2D
    from ros2_nav_course.simulation.robot_model import DifferentialDriveRobot
    from ros2_nav_course.utils.mock_ros2 import spin_once

    print("=" * 60)
    print("SLAM 节点 — RTAB-Map 原理演示")
    print("=" * 60)

    world = World2D()
    robot = DifferentialDriveRobot(1.0, 2.0, 0.5)
    slam = RTABMapNode(world, robot, stm_size=10, wm_size=20, hz=10.0)

    from ros2_nav_course.perception.sensor_node import PerceptionNode
    perception = PerceptionNode(world, robot, hz=10.0)

    robot.set_velocity(0.4, 0.15)
    steps = 100
    for i in range(steps):
        robot.update(0.1)
        perception.tick(0.1)
        slam.tick(0.1)
        if i % 10 == 0 and i > 0:
            print(f"  步{i:3d}: {slam.memory_summary()}")

    print(f"\n最终: {slam.memory_summary()}")
    print(f"\n✅ SLAM 节点演示完成")
    print(f"   关键帧数: {slam.n_keyframes} (从 {steps} 个原始帧)")
    print(f"   回环检测: {slam.n_loops} 次")
    print(f"   三级记忆: STM(最近)={len(slam.stm)}, WM={len(slam.wm)}, LTM={len(slam.ltm)}")
