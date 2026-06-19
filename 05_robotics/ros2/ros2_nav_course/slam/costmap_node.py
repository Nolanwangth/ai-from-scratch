# 运行: conda activate ros2_nav && python ros2_nav_course/slam/costmap_node.py
"""
代价地图节点 — 占据栅格 + 膨胀
================================
SLAM 出原始占据栅格, 但原始地图不能直接给 A*:

    原始地图:   0=空, 100=障碍物
    代价地图:   0=自由, 50=靠近障碍物, 254=致命

需要膨胀 (inflation) 的原因:
    机器人有半径 (30cm), 不能当场是一个点。
    在障碍物周围画一个 "禁止区" = 膨胀层
    → A* 自然就会绕开障碍物

这和 Nav2 的 costmap_2d 原理一样。
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from ros2_nav_course.utils.mock_ros2 import (
    Node, OccupancyGrid, log_ok, log_info
)


class CostmapNode(Node):
    """
    代价地图节点: 原始地图 → 带膨胀层的代价地图。

    订阅: /map (占据栅格)
    发布: /costmap (带膨胀层)

    膨胀算法: 距离变换
        对每个占据格子做 BFS 距离变换
        离障碍物越近 → 代价越高 (但未全部禁止)
        离障碍物足够近 → 致命 (禁止进入)
    """

    def __init__(self, robot_radius: float = 0.3,  # 30cm 底盘半径
                 inflation_radius: float = 0.5,     # 额外 20cm 安全距离
                 hz: float = 10.0):
        super().__init__("costmap_node")
        self.robot_radius = robot_radius
        self.inflation_radius = inflation_radius
        self.latest_map: OccupancyGrid = None
        self.costmap: OccupancyGrid = None

        self.create_subscription("/map", self._map_callback, "OccupancyGrid")
        self.create_publisher("/costmap", "OccupancyGrid")
        self.create_timer(1.0 / hz, self._inflate_and_publish)
        log_ok(f"[costmap] ready (radius={robot_radius}m, inflation=+{inflation_radius}m)")

    def _map_callback(self, og: OccupancyGrid):
        self.latest_map = og

    def _inflate_and_publish(self):
        """膨胀算法: 只对占据格子膨胀, 不污染未知/自由."""
        if self.latest_map is None:
            return

        raw = np.array(self.latest_map.data).reshape(
            self.latest_map.height, self.latest_map.width
        ).astype(np.float32)

        H, W = raw.shape
        inflate_px = int(self.inflation_radius / self.latest_map.resolution)

        # 只膨胀占据格子, 分 mask 处理避免 unknown 被稀释
        occ_mask = (raw >= 100).astype(np.float32)  # 只有障碍物本体

        from scipy import ndimage
        y, x = np.ogrid[-inflate_px:inflate_px+1, -inflate_px:inflate_px+1]
        kernel = (x*x + y*y <= inflate_px*inflate_px).astype(np.float32)
        dilated = ndimage.maximum_filter(occ_mask, footprint=kernel)

        # 分层构建 costmap:
        #   默认=未知(80, 高代价可过)
        #   raw==0 且不在膨胀区内 → 自由(0)
        #   膨胀区内且 raw!=100 → 膨胀区(50)
        #   raw>=100 → 致命(254)
        costmap = np.full((H, W), 80, dtype=np.int16)
        costmap[(raw == 0) & (dilated == 0)] = 0      # 自由
        costmap[(dilated > 0) & (raw < 100)] = 50      # 膨胀区
        costmap[raw >= 100] = 254                       # 致命
        costmap_out = costmap

        self.costmap = OccupancyGrid(
            width=W, height=H,
            resolution=self.latest_map.resolution,
            origin_x=self.latest_map.origin_x,
            origin_y=self.latest_map.origin_y,
            data=list(costmap_out.flatten().astype(int))
        )
        self.publish("/costmap", self.costmap)


if __name__ == "__main__":
    from ros2_nav_course.simulation.world_2d import World2D

    print("=" * 60)
    print("代价地图节点 — 占据栅格膨胀")
    print("=" * 60)

    world = World2D()
    costmap = CostmapNode(robot_radius=0.3, inflation_radius=0.5)

    # 模拟收到地图
    raw_grid, ox, oy = world.to_occupancy_grid(resolution=0.05)
    og = OccupancyGrid(
        width=raw_grid.shape[1], height=raw_grid.shape[0],
        resolution=0.05, origin_x=ox, origin_y=oy,
        data=list(raw_grid.flatten())
    )
    costmap._map_callback(og)
    costmap._inflate_and_publish()

    cm = costmap.costmap
    data = np.array(cm.data).reshape(cm.height, cm.width)
    n_free = (data == 0).sum()
    n_inflated = (data == 50).sum()
    n_occ = (data >= 100).sum()
    total = data.size

    print(f"\n  地图 ({cm.width}×{cm.height}, {cm.resolution}m/格):")
    print(f"    自由空间:   {n_free:6d}  ({n_free/total*100:.1f}%)")
    print(f"    膨胀区:     {n_inflated:6d}  ({n_inflated/total*100:.1f}%)")
    print(f"    占据/致命:  {n_occ:6d}  ({n_occ/total*100:.1f}%)")

    print(f"\n  ✅ 代价地图节点演示完成")
    print(f"  → 膨胀半径 = 底盘(0.3m) + 安全(0.2m) = 0.5m")
    print(f"  → 障碍物周围 10 格内都是膨胀区")
    print(f"  → A* 规划时会在膨胀区内找路, 不会进入致命区")
