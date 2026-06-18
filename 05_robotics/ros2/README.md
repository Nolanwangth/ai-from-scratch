# ROS2 导航系统 — 完整教学项目

```
感知 → SLAM → 代价地图 → Behavior Tree → A* → 两种控制模式
```

全套模块化导航系统。Mac 上就能跑，零 ROS2 安装。

---

## 快速开始

```bash
# 一次性环境搭建 (已经做过了就跳过)
conda create -n ros2_nav python=3.10 numpy matplotlib scipy -y

# 激活环境
conda activate ros2_nav
cd ai-from-scratch/05_robotics/ros2
```

## 玩耍指令

```bash
# 1. 看房间布局 (障碍物 / 地标 / 代价地图阴影区)
python demo_navigation.py --map

# 2. 看 ROS2 数据流 + 算法总览表
python demo_navigation.py --explain

# 3. 三面板可视化 (默认: MPC 打方向盘 + PID 踩油门)
python demo_navigation.py --viz

# 4. 切到一体化 MPC 模式 (像 Waymo — 一个代价函数同时决定方向盘和油门)
python demo_navigation.py --viz --mode=unified

# 5. 纯终端跑 (不弹 matplotlib 窗口)
python demo_navigation.py --run --mode=split
python demo_navigation.py --run --mode=unified
```

---

## 整体架构

```
  感知 (30Hz)          SLAM (10Hz)          代价地图 (10Hz)
  D435 深度+RGB        RTAB-Map 三级记忆      距离变换 + 膨胀
       |                     |                     |
    /camera/*              /map                /costmap
       |                     |                     |
       +---------------------+---------------------+
                             |
                             v
                      Behavior Tree 决策
                   "路径被堵了? 重规划!"
                             |
                      /goal_point
                             |
                             v
                      A* 全局规划
                  f(n) = g(n) + h(n)
                  8邻域, 对角线启发式
                             |
                          /plan
                             |
                             v
              ┌──────────────────────────────┐
              │ split 模式 (默认)              │
              │  SpeedProfiler → 速度上限表    │
              │  MPC(ω) 采样41个候选方向盘     │ ← 横向
              │  PID(v) P+I+D + 抗饱和积分     │ ← 纵向
              │  costmap 网格 O(1) 查表防撞    │
              └──────────────────────────────┘
              ┌──────────────────────────────┐
              │ unified 模式                  │
              │  UnifiedMPC (v,ω) 同时采样     │
              │  273 个候选一起评分            │
              │  一个代价 → 方向盘 + 油门      │
              │  costmap 网格 O(1) 查表防撞    │
              └──────────────────────────────┘
                             |
                        /cmd_vel (v, ω)
                             |
                             v
                    差速轮底盘运动学
                x' = x + v·cos(θ)·dt
                θ' = θ + ω·dt
```

---

## ROS2 数据流 — 7 条 Topic, 6 个 Node

```
Node                发布                    订阅
─────────────────   ───────────────────     ───────────
Perception (感知)    /odom, /camera/*        —
SLAM (RTAB-Map)     /map, /pose_corrected   /odom
Costmap (代价地图)   /costmap                /map
Decision (决策)     /goal_point             —
Planner (A*)        /plan                   /costmap
Control (控制)      /cmd_vel                /plan
```

每个 Node 完全独立，互不知道对方存在，只通过 Topic 通信。这就是 ROS2 的模块化本质。

---

## 两种控制模式

| | Split (默认) | Unified |
|---|---|---|
| 方向盘 | MPC: 41个ω候选, 预测10步 | Unified MPC: 273个(v,ω)候选, 预测8步 |
| 油门 | PID: Kp·e + Ki·∫e + Kd·de/dt | 和方向盘同一个代价函数 |
| 速度规划 | SpeedProfiler (曲率+目标+障碍物距离) | 内建在 rollout 代价里 |
| 碰撞检测 | costmap 网格 O(1) 查表 | costmap 网格 O(1) 查表 |
| 靠近障碍物 | 0.5m 减速, 0.35m 倒车 | 代价渐变罚分 |
| 真实世界类比 | AGV 叉车、扫地机等低速场景 | Waymo / Cruise / Apollo 自动驾驶 |
| 学什么 | PID 三项分别怎么影响油门 | 一体化 MPC 怎么同时管横纵 |

---

## 小车怎么开的

```
正常行驶:
  A* 路径 → SpeedProfiler 算每个路径点的限速
         → MPC 沿着路径打方向盘
         → PID 追速度目标 (P=比例, I=累积偏差, D=阻尼)
         → BT 盯着: "路通着? → 继续走"

靠近障碍物:
  < 0.5m  → SpeedProfiler 降速 → PID 平滑减速
  < 0.35m → 控制层倒车 → 告知 BT → A* 重规划绕开
  路径上  → costmap 网格 ≥254 是致命区 → A* 一开始就绕开

突然掉落的障碍物 (BOX):
  第 400 步 BOX 从天上掉下来堵路
  → 控制层: 检测到前方有障碍 → 急停
  → BT: "路被堵了!" → 触发重规划
  → A*: 算出绕开 BOX 的新路
  → 小车: 走新路继续前进 → 到达目标
```

---

## 防撞四层保护

| 层 | 什么时候 | 做什么 |
|---|---|---|
| **A*** | 算路径时 | costmap 格子 ≥254 → 不可通行, 直接绕 |
| **SpeedProfiler** | 每次 A* 重算 | 路径靠近障碍物 → 降低限速 |
| **控制层 (实时)** | 50Hz 每帧 | 机器人位置 < 0.5m 障碍物 → 减速 |
| **控制层 (实时)** | 50Hz 每帧 | 机器人位置 < 0.35m 障碍物 → 倒车 + BT 重规划 |

---

## 为什么 costmap 阴影区不用射线检测

```
costmap 是 SLAM 建完地图后预计算好的网格:
  每个格子 (x,y) → 一个值 (0=自由, 50=膨胀阴影区, 254=致命障碍物)

控制层查障碍物 = costmap[y_idx][x_idx]  >= 50  → O(1) 数组索引
根本不需要每帧遍历所有障碍物做碰撞检测

桌子周围的橙色/红色阴影 = 膨胀层 (机器人 30cm 底盘 + 20cm 安全距离)
```

---

## 项目文件

```
ros2/
├── demo_navigation.py              # 主入口: --map / --explain / --run / --viz
├── README.md                       # 你正在看的这个文件
├── package.xml, setup.py            # ROS2 包配置
└── ros2_nav_course/
    ├── utils/
    │   ├── mock_ros2.py             # ROS2 教学层 (Node/Topic/Service/Action 四个概念)
    │   └── visualizer.py            # 三面板可视化 (地图+状态+D435深度视角)
    ├── simulation/
    │   ├── world_2d.py              # 10m×10m 房间, 桌子/椅子/柜子/墙, 4个地标
    │   └── robot_model.py           # 差速轮底盘运动学
    ├── perception/
    │   └── sensor_node.py           # D435 传感器模拟 (30Hz, 加噪声)
    ├── slam/
    │   ├── rtabmap_node.py          # RTAB-Map 三级记忆 (STM→WM→LTM)
    │   └── costmap_node.py          # 占据栅格 + 距离变换膨胀
    ├── decision/
    │   └── decision_node.py         # Behavior Tree (正常导航 / 重规划 / 放弃)
    ├── planning/
    │   └── astar_node.py            # A* (f=g+h, 8邻域, 对角线启发式)
    └── control/
        └── control_node.py          # 两种模式: split (MPC+PID) 或 unified (一体化MPC)
```

---

## 可以折腾的

```bash
# 对比两种控制模式的区别
python demo_navigation.py --viz --mode=split      # 看 PID 的 P/I/D 三项
python demo_navigation.py --viz --mode=unified    # 一个代价决定一切

# 单独测试每个模块
python ros2_nav_course/perception/sensor_node.py
python ros2_nav_course/slam/rtabmap_node.py
python ros2_nav_course/slam/costmap_node.py
python ros2_nav_course/decision/decision_node.py
python ros2_nav_course/planning/astar_node.py
python ros2_nav_course/control/control_node.py --mode=split
python ros2_nav_course/control/control_node.py --mode=unified

# 调 PID 参数: 打开 control/control_node.py, 找 LongitudinalPID.__init__
#   kp=2.0  油门更猛
#   kd=0.3  减少超调
#   ki=0.5  稳态误差补得更快

# 调 MPC 预测视野: 同一个文件, 找 LateralMPC.__init__
#   predict_steps=20  往前看更远 (≈2秒)
#   n_samples=30      方向盘更细腻 (61个候选)

# 加障碍物: 打开 simulation/world_2d.py, 找 _build_default_scene
#   self.obstacles.append(Obstacle(5, 5, 3.0, 0.5, "wall"))

# 改目标: 打开 demo_navigation.py, 搜 "workstation"
#   → 改成 "kitchen" 或 "toolbox"

# 换 A* 启发式: 打开 planning/astar_node.py, 找 _heuristic
#   对角线距离 → 换成欧几里得: math.sqrt(dx*dx + dy*dy)
```

---

## 跑通结果

```
起点: 充电桩 (1.0, 1.0)  →  目标: 工位 (8.0, 8.0)
A* 路径: 210 个航点, 绕开桌子和柜子
第 400 步 BOX 掉落 → BT 重规划 → A* 绕行
MPC+PID 50Hz 跟踪
到达: (7.80, 7.78)  距离目标 0.30m  用时 ~722 步 (~15秒)
跟踪误差: 平均 0.031m, 最大 0.090m
```
