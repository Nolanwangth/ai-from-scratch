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
                 hz: float = 30.0):
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

        # ── 统计 ──
        self.n_keyframes = 0
        self.n_loops = 0
        self.corrected_pose = (0.0, 0.0, 0.0)

        # ── 发布者 ──
        self.create_publisher("/map", "OccupancyGrid")
        self.create_publisher("/pose_corrected", "Pose")
        self.create_publisher("/loop_detected", "any")

        # ── 订阅 /odom ──
        self.create_subscription("/odom", self._odom_callback, "Odometry")
        self.create_timer(1.0 / hz, self._tick)
        log_ok(f"[SLAM] RTAB-Map ready (STM={stm_size}, WM={wm_size})")

    def _odom_callback(self, odom: Odometry):
        """收到里程计数据 → 处理一帧"""
        x, y, theta = odom.pose.x, odom.pose.y, odom.pose.theta

        # Step 1: 关键帧判定
        if self._is_keyframe(x, y):
            kf = KeyFrame(self.next_id, x, y, theta)
            self.next_id += 1
            self.n_keyframes += 1

            # Step 2: 加入 STM
            self.stm.append(kf)
            self.last_kf_pos = (x, y)

            # Step 3: STM→WM 转移
            self._manage_memory()

            # Step 4: 回环检测
            self._detect_loop(kf)

        self.corrected_pose = (x, y, theta)

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
        """回环检测: 只在 WM 里搜!"""
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
            self.publish("/loop_detected",
                         f"回环! {new_kf.id} ↔ {best_kf.id} (sim={best_sim:.2f})")
            # 模拟图优化: 修正当前位置 (真实的交给 g2o)
            new_kf.weight *= 1.2  # 回环帧多重要 → 更难被踢

    def _tick(self):
        """定时发布地图"""
        grid, ox, oy = self.world.to_occupancy_grid(resolution=0.05)
        og = OccupancyGrid(
            width=grid.shape[1], height=grid.shape[0],
            resolution=0.05, origin_x=ox, origin_y=oy,
            data=list(grid.flatten())
        )
        self.publish("/map", og)

        px, py, pt = self.corrected_pose
        self.publish("/pose_corrected", Pose(px, py, pt))

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
