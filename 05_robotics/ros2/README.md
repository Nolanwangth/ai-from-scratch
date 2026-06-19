# ROS2 导航系统 — 完整教学项目

```
感知 → SLAM(增量建图+漂移+回环) → 代价地图 → Behavior Tree → A* → 两种控制模式
```

全套模块化导航+SLAM 教学系统。Mac 上就能跑，零 ROS2 安装。

---

## 快速开始

```bash
# 激活你的 ros2_nav 环境
conda activate ros2_nav
cd /Users/nolan/Desktop/agi/ai-from-scratch/05_robotics/ros2
```

## 玩耍指令

```bash
# 1. 看房间布局
python demo_navigation.py --map

# 2. 看完整数据流 + 算法总览
python demo_navigation.py --explain

# 3. 三面板可视化 — split 模式 (MPC 方向盘 + PID 油门)
python demo_navigation.py --viz

# 4. unified 统一 MPC
python demo_navigation.py --viz --mode=unified

# 5. 终端跑
python demo_navigation.py --run --mode=split
python demo_navigation.py --run --mode=unified

# 6. 可选: 注入后期动态障碍, 专门看 BT 重规划
python demo_navigation.py --run --mode=split --dynamic
python demo_navigation.py --viz --mode=unified --dynamic
```

---

## ROS2 通信模型 — 5 个原语全用上

```
┌────────────┬────────┬─────────────────────────────────────────┐
│   通信原语   │  数量   │  用在哪                                  │
├────────────┼────────┼─────────────────────────────────────────┤
│ Node       │   6    │ Perception/SLAM/Costmap/Decision/        │
│            │        │ Planner/Controller — 各管一摊             │
├────────────┼────────┼─────────────────────────────────────────┤
│ Topic      │   8    │ /odom /camera/* /map /costmap            │
│            │        │ /pose_corrected /goal_point /plan /cmd_vel│
│            │        │ 高频异步数据流, 模块解耦的核心             │
├────────────┼────────┼─────────────────────────────────────────┤
│ Service    │   1    │ /get_map — SLAM 提供, 任意 Node 可调     │
├────────────┼────────┼─────────────────────────────────────────┤
│ Action     │   1    │ /navigate_to_goal — Decision 提供        │
├────────────┼────────┼─────────────────────────────────────────┤
│ Parameter  │   4    │ kp/ki/kd/max_v — Controller 声明         │
│            │        │ 运行时在线调 PID                          │
└────────────┴────────┴─────────────────────────────────────────┘
```

## 数据流

```
Perception (30Hz)    SLAM (10Hz)              Costmap (10Hz)
D435 sim + noise     Drift odom + 增量建图    分层膨胀
    |                    |                        |
  /odom ─────────────> (subscribe)                 |
                         /map ───────────────> (subscribe)
                         /pose_corrected           |
                              |                  /costmap ───┐
                              v                        |     |
                        Planner (A*)                    |     |
                        (subscribe)                     v     v
                              |                   Controller
                           /plan ──────────────────> (subscribe)
                                              Parameter: kp/ki/kd/max_v
                                                         |
                                                     /cmd_vel (v,ω)
                                                         |
                                                         v
                                                   Differential Drive

Decision (BT)
    Action: /navigate_to_goal ← 上层调用
    Topic:  /goal_point → Planner
    Topic:  /decision_state → 监控

核心模块通过 Topic/Service/Action 通信。`demo_navigation.py` 里为了 mock 时序稳定, 初始化阶段会手动同步一次 planner/controller 状态。
```

---

## SLAM — 增量建图 + odom 漂移 + 回环

```
和之前一次性拿真值地图的区别:

  之前: world.to_occupancy_grid() → 完整地图, 零误差
  现在: D435 FOV 扇形射线逐步扫描, 边建图边导航

  建图过程:
    第一步: 真值位姿 → 发射射线 → 测距 (物理测量是准的)
    第二步: 估计位姿 (带漂移) → 插图 (SLAM 不知道自己偏了)
    第三步: 走一圈回来 → 回环检测命中 → 修正 70% 漂移

  代价地图:
    未知区域 (gray)  → cost=80   (高代价, 让 A* 尽量走已知)
    自由空间 (clear) → cost=0
    膨胀区  (orange) → cost=120  (可过, 但强烈避开)
    致命区  (red)    → cost=254

  --viz 里能看到:
    灰色半透明 = 未探索 → 随机器人移动逐步被扫开
    深色半透明 = 障碍物 (SLAM 建出来的)
    橙色半透明 = costmap 膨胀层 (上层叠加)
```

---

## 两种控制模式

| | Split (--mode=split) | Unified (--mode=unified) |
|---|---|---|
| 方向盘 | MPC: 41 ω, 预测 10 步 | UnifiedMPC: 126 (v,ω), 预测 6 步 |
| 油门 | PID: Kp·e + Ki·∫e + Kd·de/dt | 和方向盘同一个代价 |
| 速度规划 | SpeedProfiler (曲率+目标减速) | 内建在 rollout 代价里 |
| 碰撞检测 | 横向 MPC 避障 + 最终速度裁剪 | costmap O(1) 查表 + 最终速度裁剪 |
| 真实类比 | AGV 叉车, 低速机器人 | Waymo / Cruise / Apollo |
| 优点 | 分层清楚, 好理解 PID/MPC/速度规划 | 速度和转向统一优化, 行为更直接 |
| 当前跑通步数 | 约 1000~1100 步, 碰撞帧数 0 | 约 550~650 步, 碰撞帧数 0 |

---

## 防撞四层

| 层 | 什么时候 | 做什么 |
|---|---|---|
| A* | 算路径时 | costmap ≥254 → 不可通行 |
| Costmap | SLAM 更新后 | 障碍物膨胀成高代价区 |
| SpeedProfiler | split 模式 | 弯道/目标附近降速 |
| 控制层 (50Hz) | 每帧 | robot < 0.2m 障碍物 → 降速 |
| BT / A* | 周期或动态障碍 | 触发重规划 |

---

## 项目结构

```
ros2/
├── demo_navigation.py              # 主入口
├── README.md                       # 你正在看的文件
├── package.xml, setup.py
└── ros2_nav_course/
    ├── utils/
    │   ├── mock_ros2.py             # ROS2 5原语 (Node/Topic/Service/Action/Parameter)
    │   └── visualizer.py            # 三面板: 地图+建图过程 + 状态 + D435 深度
    ├── simulation/
    │   ├── world_2d.py              # 10m×10m 房间
    │   └── robot_model.py           # 差速轮底盘
    ├── perception/
    │   └── sensor_node.py           # D435 模拟 (30Hz, 噪声)
    ├── slam/
    │   ├── rtabmap_node.py          # 增量建图 + odom漂移 + 回环 + /get_map
    │   └── costmap_node.py          # 分层膨胀 (未知/自由/膨胀/致命)
    ├── decision/
    │   └── decision_node.py         # Behavior Tree + /navigate_to_goal
    ├── planning/
    │   └── astar_node.py            # A* (订阅 /pose_corrected)
    └── control/
        └── control_node.py          # split (MPC+PID) / unified (UnifiedMPC)
```

---

## 可以折腾的

```bash
# 对比两种控制模式
python demo_navigation.py --viz --mode=split
python demo_navigation.py --viz --mode=unified

# 单独测试各模块
python ros2_nav_course/perception/sensor_node.py
python ros2_nav_course/slam/rtabmap_node.py
python ros2_nav_course/planning/astar_node.py
python ros2_nav_course/control/control_node.py --mode=split
python ros2_nav_course/control/control_node.py --mode=unified

# 调 PID: control/control_node.py → LongitudinalPID.__init__
#   kp=2.0 → 油门更猛, kd=0.3 → 减少超调

# 调 SLAM 漂移: slam/rtabmap_node.py → __init__
#   _drift_trans_ratio = 0.05 → 漂移更大, 更难
#   _drift_trans_ratio = 0.0  → 关漂移

# 改目标: demo_navigation.py, 搜 "workstation"
#   → "kitchen" 或 "toolbox"

# 加障碍物: simulation/world_2d.py → _build_default_scene
#   self.obstacles.append(Obstacle(5, 5, 3.0, 0.5, "wall"))

# 换 A* 启发式: planning/astar_node.py → _heuristic
```

---

## 跑通结果

```
起点: 充电桩 (1.0, 1.0) → 目标: 工位 (8.0, 8.0)
SLAM: 10m D435 FOV 逐步扫描, 3% 平移漂移 + 5% 旋转漂移
A*: 基于增量地图和 costmap 低频重规划
split:   863 步到工位 (误差 0.29m)
unified: 671 步到工位 (误差 0.30m)
```
