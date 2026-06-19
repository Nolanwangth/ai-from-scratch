# 运行: conda activate ros2_nav && python ros2_nav_course/planning/astar_node.py
"""
A* 全局规划节点
================
经典 A* 算法: f(n) = g(n) + h(n)

    g(n): 从起点到节点 n 的实际代价 (走过的距离)
    h(n): 从 n 到终点的预估代价 (直线距离, 启发式)
    f(n): 总估计代价

A* = Dijkstra + 方向感

直观理解:
    你在一个十字路口:
        Dijkstra: 同等探索每个方向 (慢)
        A*:       "目标在东北方, 优先往东和往北走" (快)

为什么 A* 保证最优:
    只要 h(n) ≤ 真实距离 (admissible), A* 找的就是最短路径。
    直线距离永远 ≤ 真实距离 → A* 最优。

输入:
    - /costmap: 代价地图
    - /goal_point: 目标点 (来自决策)
    - /pose_corrected: 当前位置

输出:
    - /plan: 全局路径 (nav_msgs/Path)
"""

import heapq
import math
from typing import List, Tuple, Optional, Callable

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from ros2_nav_course.utils.mock_ros2 import (
    Node, Pose, Path, OccupancyGrid, log_ok, log_info, log_warn
)


class AStarPlanner:
    """
    纯 A* 算法实现。

    输入: 代价地图 (2D 数组), 起点, 终点
    输出: 最短路径 (或 None)

    这里把它从节点里拆出来, 方便你理解 A* 本身。
    真实 Nav2 的 A* 插件也是这样: 一个纯粹的算法类。
    """

    def __init__(self, cost_threshold: int = 100):
        """
        cost_threshold: 代价 >= 此值的格子不可通行 (默认 100 = 膨胀区不可走)
        """
        self.cost_threshold = cost_threshold

    def plan(self,
             costmap: OccupancyGrid,
             start_x: float, start_y: float,
             goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """
        A* 主函数。

        Returns:
            路径 [(x0,y0), (x1,y1), ..., (goal_x,goal_y)] 或 None
        """
        W, H = costmap.width, costmap.height
        res = costmap.resolution
        ox, oy = costmap.origin_x, costmap.origin_y
        data = np.array(costmap.data).reshape(H, W)

        # 世界坐标 → 网格坐标
        sx = int((start_x - ox) / res)
        sy = int((start_y - oy) / res)
        gx = int((goal_x - ox) / res)
        gy = int((goal_y - oy) / res)

        # 边界检查
        if not (0 <= sx < W and 0 <= sy < H):
            return None
        if not (0 <= gx < W and 0 <= gy < H):
            return None

        # 起点或终点被占了
        if data[sy, sx] >= self.cost_threshold:
            return None
        if data[gy, gx] >= self.cost_threshold:
            return None

        # A* 核心数据结构
        # open_set = 优先队列 (f, counter, cell)
        # g_score = 从起点到每个格子的实际代价
        # came_from = 回溯路径

        counter = 0  # 中断 tie-breaking
        open_set = [(0, counter, (sx, sy))]
        g_score = { (sx, sy): 0.0 }
        came_from = {}

        # 8 邻域 (包括对角线)
        neighbors_8 = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        n_explored = 0

        while open_set:
            f, _, current = heapq.heappop(open_set)
            cx, cy = current
            n_explored += 1

            # 到达目标
            if cx == gx and cy == gy:
                path = self._reconstruct_path(came_from, current)
                world_path = [(ox + px * res, oy + py * res)
                              for px, py in path]
                return world_path

            # 探索邻域
            for dx, dy in neighbors_8:
                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if data[ny, nx] >= self.cost_threshold:
                    continue

                # g(n) = g(parent) + step_cost
                step_dist = math.sqrt(dx**2 + dy**2) * res
                cell_cost = 1.0 + data[ny, nx] / 100.0  # 代价越高越慢
                tentative_g = g_score[current] + step_dist * cell_cost

                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g

                    # h(n) = 对角线距离 (比曼哈顿更紧, 仍 admissible)
                    h = self._heuristic(nx, ny, gx, gy, res)
                    f_score = tentative_g + h

                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, (nx, ny)))

        return None  # 没找到路

    def _heuristic(self, nx: int, ny: int, gx: int, gy: int,
                   res: float) -> float:
        """
        启发式函数 h(n): 对角线距离。

        对角线距离 ≤ 真实最短距离 → admissible → A* 最优。

        三种常用:
            曼哈顿: |dx| + |dy|                    (4 邻域, 不准)
            对角线: max(dx,dy) + (√2-1)*min(dx,dy)  (8 邻域, 紧)
            欧几里得: √(dx²+dy²)                     (最紧, 但需要 sqrt)
        """
        dx = abs(nx - gx) * res
        dy = abs(ny - gy) * res
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]):
        """从 goal 回溯到 start"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


class AStarPlannerNode(Node):
    """
    A* 规划节点: A* 算法的 ROS2 包装。

    订阅:
        /costmap: 代价地图
        /goal_point: 目标点 (决策层给的)
        /pose_corrected: 当前位置 (SLAM 给的)

    发布:
        /plan: 全局路径
    """

    def __init__(self, world, hz: float = 10.0):
        super().__init__("astar_planner_node")
        self.world = world
        self.planner = AStarPlanner(cost_threshold=200)  # 254=致命, 50=膨胀可过
        self.latest_costmap: Optional[OccupancyGrid] = None
        self.current_goal: Optional[Tuple[float, float]] = None
        self.current_pose: Tuple[float, float] = (0.0, 0.0)
        self.latest_path: Optional[List[Tuple[float, float]]] = None

        self.create_subscription("/costmap", self._costmap_callback, "OccupancyGrid")
        # SLAM 估计位姿 (带漂移), 不是真实位姿 — 这才是真 SLAM 的语义
        self.create_subscription("/pose_corrected", self._pose_callback, "Pose")
        self.create_publisher("/plan", "Path")
        log_ok(f"[planner] A* planner ready")

    def _costmap_callback(self, cm: OccupancyGrid):
        self.latest_costmap = cm

    def _pose_callback(self, pose):
        """订阅 SLAM 的 /pose_corrected — 用估计位姿规划, 不是真值."""
        self.current_pose = (pose.x, pose.y)

    def set_goal(self, x: float, y: float):
        self.current_goal = (x, y)

    def plan(self, verbose: bool = True) -> Optional[List[Tuple[float, float]]]:
        """算路径"""
        if self.latest_costmap is None or self.current_goal is None:
            return None
        path = self.planner.plan(
            self.latest_costmap,
            self.current_pose[0], self.current_pose[1],
            self.current_goal[0], self.current_goal[1],
        )
        self.latest_path = path
        if path:
            if verbose:
                log_ok(f"  [planner] Path found: {len(path)} waypoints")
            self.publish("/plan", Path(poses=[Pose(x, y, 0) for x, y in path]))
        return path


if __name__ == "__main__":
    from ros2_nav_course.simulation.world_2d import World2D
    from ros2_nav_course.slam.costmap_node import CostmapNode

    print("=" * 60)
    print("A* 全局路径规划 演示")
    print("=" * 60)

    world = World2D()

    # 1. 生成代价地图
    costmap_node = CostmapNode()
    raw_grid, ox, oy = world.to_occupancy_grid(resolution=0.05)
    og = OccupancyGrid(
        width=raw_grid.shape[1], height=raw_grid.shape[0],
        resolution=0.05, origin_x=ox, origin_y=oy,
        data=list(raw_grid.flatten())
    )
    costmap_node._map_callback(og)
    costmap_node._inflate_and_publish()
    cm = costmap_node.costmap

    # 2. A* 规划
    planner_node = AStarPlannerNode(world)
    planner_node.latest_costmap = cm
    planner_node.set_goal(8.0, 8.0)
    planner_node.set_pose(0.5, 0.5)

    path = planner_node.plan()

    if path:
        print(f"\n  A* 结果: 路径长度 = {len(path)} 个路径点")
        print(f"  前 3 点: {[(round(x,2), round(y,2)) for x,y in path[:3]]}")
        print(f"  后 3 点: {[(round(x,2), round(y,2)) for x,y in path[-3:]]}")
    else:
        print(f"\n  ❌ 没找到路径 (被障碍物挡住)")

    print(f"\n  ✅ A* 规划节点演示完成")
    print(f"  → 算法: f(n) = g(n) + h(n)")
    print(f"  → heuristics: 对角线距离 (admissible)")
    print(f"  → 8 邻域探索, O(N log N) with heap")
