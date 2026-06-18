"""
2D 仿真世界 — 一个带障碍物的室内环境。
=======================================
ROS2 里的仿真通常用 Gazebo。
这里用简单的网格世界替代, 让你在 Mac 上就能跑。

代表了: 你家客厅/工位的简化版。
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Set


class Obstacle:
    """一个障碍物 (矩形)"""

    def __init__(self, x: float, y: float, w: float, h: float,
                 name: str = ""):
        self.x = x          # 中心 x
        self.y = y          # 中心 y
        self.w = w          # 宽度
        self.h = h          # 高度
        self.name = name

    @property
    def left(self): return self.x - self.w / 2

    @property
    def right(self): return self.x + self.w / 2

    @property
    def top(self): return self.y + self.h / 2

    @property
    def bottom(self): return self.y - self.h / 2

    def contains(self, px: float, py: float) -> bool:
        """点 (px, py) 是否在障碍物内?"""
        return (self.left <= px <= self.right and
                self.bottom <= py <= self.top)

    def distance_to(self, px: float, py: float) -> float:
        """点到障碍物的最近距离 (0 = 在里面)"""
        if self.contains(px, py):
            return 0.0
        dx = max(self.left - px, 0, px - self.right)
        dy = max(self.bottom - py, 0, py - self.top)
        return math.sqrt(dx**2 + dy**2)


class World2D:
    """
    2D 室内世界。

    模拟一个 10m × 10m 的房间:
        - 墙壁 (边界)
        - 桌子 (障碍物)
        - 目标点 (工位/工具箱)
        - 可选的动态障碍物

    这个世界的 [占据栅格] 会传给 SLAM → 决策 → A* → MPC。
    """

    def __init__(self, width: float = 10.0, height: float = 10.0):
        self.width = width
        self.height = height

        # 障碍物列表
        self.obstacles: List[Obstacle] = []
        self._build_default_scene()

        # 语义标签
        self.landmarks = {
            "charger":     (1.0, 1.0),
            "toolbox":     (8.0, 1.5),
            "workstation": (8.0, 8.0),
            "kitchen":     (1.5, 8.0),
        }

    def _build_default_scene(self):
        """建一个 10×10 的房间布局"""
        # 桌子
        self.obstacles.append(Obstacle(3, 3, 1.5, 1.0, "table1"))
        self.obstacles.append(Obstacle(7, 5, 1.0, 1.5, "table2"))
        self.obstacles.append(Obstacle(4, 7, 2.0, 0.8, "table3"))
        self.obstacles.append(Obstacle(3.5, 2.0, 0.4, 0.4, "chair1"))
        self.obstacles.append(Obstacle(7.5, 4.5, 0.4, 0.4, "chair2"))
        self.obstacles.append(Obstacle(1.0, 5.0, 0.8, 2.0, "cabinet"))

        # 墙壁 = 边界障碍物 (用 0.1m 厚的墙)
        wall_t = 0.2
        w, h = self.width, self.height
        self.walls = [
            Obstacle(w/2, -wall_t/2, w, wall_t, "wall-S"),
            Obstacle(w/2, h+wall_t/2, w, wall_t, "wall-N"),
            Obstacle(-wall_t/2, h/2, wall_t, h, "wall-W"),
            Obstacle(w+wall_t/2, h/2, wall_t, h, "wall-E"),
        ]

    def is_collision(self, x: float, y: float, radius: float = 0.2) -> bool:
        """检查 (x,y) 以 radius 半径是否碰撞"""
        # 墙壁
        for w in self.walls:
            if w.distance_to(x, y) < radius:
                return True
        # 障碍物
        for o in self.obstacles:
            if o.distance_to(x, y) < radius:
                return True
        # 边界
        if not (0 < x < self.width and 0 < y < self.height):
            return True
        return False

    def get_cost_at(self, x: float, y: float, robot_radius: float = 0.2) -> float:
        """
        返回 (x,y) 的成本:
            0.0 = 自由空间 (白色)
            0.5 = 靠近障碍物 (浅灰)
            1.0 = 碰撞 (黑色)
        """
        if self.is_collision(x, y, robot_radius):
            return 1.0  # 致命

        min_dist = min(
            (o.distance_to(x, y) for o in self.obstacles + self.walls),
            default=float('inf')
        )

        if min_dist < robot_radius * 1.5:
            return 0.5  # 靠近障碍物
        return 0.0  # 自由

    def to_occupancy_grid(self, resolution: float = 0.05,
                          robot_radius: float = 0.2) -> Tuple[np.ndarray, float, float]:
        """
        把世界转成占据栅格 (ROS2 OccupancyGrid 格式)

        返回:
            grid: (H, W) numpy array, -1=未知, 0=空, 100=占
            origin_x: 地图左下角 x
            origin_y: 地图左下角 y
        """
        W = int(self.width / resolution) + 1
        H = int(self.height / resolution) + 1
        grid = np.zeros((H, W), dtype=np.int8)

        origin_x = 0.0
        origin_y = 0.0

        for iy in range(H):
            for ix in range(W):
                wx = origin_x + ix * resolution
                wy = origin_y + iy * resolution
                cost = self.get_cost_at(wx, wy, robot_radius)
                if cost >= 1.0:
                    grid[iy, ix] = 100  # 占据
                elif cost > 0:
                    grid[iy, ix] = 50   # 膨胀
                else:
                    grid[iy, ix] = 0    # 自由

        return grid, origin_x, origin_y

    def render_ascii(self, robot_x: float = -1, robot_y: float = -1,
                     target_x: float = -1, target_y: float = -1,
                     path: Optional[List[Tuple[float, float]]] = None,
                     resolution: float = 0.3) -> str:
        """
        用 ASCII 打印地图 (教学用)

        @ = 障碍物
        . = 自由空间
        R = 机器人
        T = 目标
        * = 规划路径
        """
        W = int(self.width / resolution)
        H = int(self.height / resolution)
        grid = [['.' for _ in range(W)] for _ in range(H)]

        # 画障碍物
        for iy in range(H):
            for ix in range(W):
                wx = ix * resolution
                wy = iy * resolution
                if self.is_collision(wx, wy):
                    grid[iy][ix] = '@'

        # 画路径
        if path:
            for px, py in path:
                ix = int(px / resolution)
                iy = int(py / resolution)
                if 0 <= ix < W and 0 <= iy < H and grid[iy][ix] == '.':
                    grid[iy][ix] = '*'

        # 画目标
        if target_x >= 0:
            ix = int(target_x / resolution)
            iy = int(target_y / resolution)
            if 0 <= ix < W and 0 <= iy < H:
                grid[iy][ix] = 'T'

        # 画机器人
        if robot_x >= 0:
            ix = int(robot_x / resolution)
            iy = int(robot_y / resolution)
            if 0 <= ix < W and 0 <= iy < H:
                grid[iy][ix] = 'R'

        # 转字符串 (y 反转, 因为屏幕左上=0, 地图下=0)
        lines = [''.join(row) for row in reversed(grid)]
        return '\n'.join(lines)


# ============================================================================
# Demo: 画世界
# ============================================================================

if __name__ == "__main__":
    world = World2D()
    print("=" * 60)
    print("2D 仿真世界 (10m × 10m)")
    print("=" * 60)
    print(f"\n障碍物: {[o.name for o in world.obstacles]}")
    print(f"地标: {world.landmarks}")
    print(f"\n地图 (俯视图):")
    print(f"  @ = 障碍物  . = 自由  * = 路径  R = 机器人  T = 目标\n")
    print(world.render_ascii(robot_x=0.5, robot_y=0.5,
                              target_x=8.0, target_y=8.0,
                              path=[(1, 1), (2, 1.5), (3, 2), (4, 3),
                                    (5, 4), (6, 5), (7, 6), (8, 7)]))
    print(f"\n地图上的标注:")
    print(f"  R = 充电桩 (0.5, 0.5)")
    print(f"  T = 工位   (8.0, 8.0)")
    print(f"  * = A* 规划的路径 (绕过桌子)")
