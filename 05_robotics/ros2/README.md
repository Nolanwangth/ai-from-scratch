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

# 2. 看完整数据流 + 算法总览 (Topic/Service/Action/Parameter 全列)
python demo_navigation.py --explain

# 3. 三面板可视化 — split 模式 (MPC 打方向盘 + PID 踩油门)
python demo_navigation.py --viz

# 4. 切到 unified 统一 MPC (一个代价函数同时决定方向盘和油门)
python demo_navigation.py --viz --mode=unified

# 5. 纯终端跑 (不弹 matplotlib 窗口)
python demo_navigation.py --run --mode=split
python demo_navigation.py --run --mode=unified
```

---

## ROS2 通信模型 — 5 个原语全用上

```
┌────────────┬────────┬─────────────────────────────────────────────┐
│   通信原语   │  数量   │  用在哪                                      │
├────────────┼────────┼─────────────────────────────────────────────┤
│ Node       │   6    │ Perception / SLAM / Costmap / Decision /     │
│            │        │ Planner / Controller — 各管一摊, 边界清晰     │
├────────────┼────────┼─────────────────────────────────────────────┤
│ Topic      │   7    │ /odom  /camera/*  /map  /costmap  /goal_point│
│            │        │ /plan  /cmd_vel  /pose_corrected             │
│            │        │ 高频异步数据流, 一对多, 模块解耦的核心         │
├────────────┼────────┼─────────────────────────────────────────────┤
│ Service    │   1    │ /get_map — SLAM 提供, 任意 Node 可调         │
│            │        │ "给我当前地图" → "这是地图数据"                │
├────────────┼────────┼─────────────────────────────────────────────┤
│ Action     │   1    │ /navigate_to_goal — Decision 提供             │
│            │        │ 长任务: 发目标 → 持续反馈 → 到达/失败/取消     │
├────────────┼────────┼─────────────────────────────────────────────┤
│ Parameter  │   4    │ kp / ki / kd / max_v — Controller 声明       │
│            │        │ 运行时在线调 PID, 不需要改代码重跑             │
└────────────┴────────┴─────────────────────────────────────────────┘
```

## 数据流

```
Perception (30Hz)      SLAM (10Hz)             Costmap (10Hz)
D435 sim + noise       RTAB-Map STM→WM→LTM     Distance transform
    |                      |                        |
  /odom ──────────────> (subscribe)                 |
  /camera/*                |                        |
                         /map ─────────────────> (subscribe)
                         /pose_corrected            |
                         Service: /get_map       /costmap ──────┐
                                                    |            |
                                                    v            v
                                              Planner (A*)   Controller
                                              (subscribe)    (subscribe)
                                                    |
                                                 /plan ──────> (subscribe)
                                                               |
                                             Parameter: kp/ki/kd/max_v
                                                               |
                                                           /cmd_vel (v,ω)
                                                               |
                                                               v
                                                         Differential Drive
                                                      x' = x+v·cosθ·dt
                                                      θ' = θ+ω·dt

Decision (BT) —— 不直接连任何 Node
    Action: /navigate_to_goal  ← 上层/用户调用
    Topic:  /goal_point        → Planner 收
    Topic:  /decision_state    → 外部监控

各 Node 完全独立, 互不知道对方存在, 只通过 Topic/Service/Action 通信。
这就是 ROS2 的模块化本质。
```

---

## 两种控制模式

| | Split (默认) | Unified |
|---|---|---|
| 方向盘 | MPC: 41 ω 候选, 预测 10 步 | UnifiedMPC: ~248 (v,ω) 候选, 预测 8 步 |
| 油门 | PID: Kp·e + Ki·∫e + Kd·de/dt | 和方向盘同一个代价函数 |
| 速度规划 | SpeedProfiler (曲率+目标+障碍物) | 内建在 rollout 代价里 |
| 碰撞检测 | costmap 网格 O(1) 查表 | costmap 网格 O(1) 查表 |
| 真实类比 | AGV 叉车、低速机器人 | Waymo / Cruise / Apollo |
| 跑通步数 | 1991 步到工位 | 1099 步到工位 |

---

## 小车怎么开的

```
正常行驶:
  A* 路径 → SpeedProfiler 算限速 → MPC 打方向盘 → PID 追速度
  BT 盯着: "路通着? → 继续"

靠近桌子 < 0.5m:
  SpeedProfiler 降速 → PID 减速
  < 0.35m: 控制层倒车 → BT 通知重规划 → A* 绕行

BOX 突然掉落:
  控制层检测障碍物 → 倒车 → BT 触发 Replan → A* 绕开 → 继续
```

---

## 防撞四层

| 层 | 什么时候 | 做什么 |
|---|---|---|
| A* | 算路径时 | costmap ≥254 → 不可通行, 绕开 |
| SpeedProfiler | A* 重算时 | 路径靠近障碍物 → 降限速 |
| 控制层 (50Hz) | 每帧 | 机器人 < 0.5m 障碍物 → 减速 |
| 控制层 (50Hz) | 每帧 | 机器人 < 0.35m → 倒车 + BT 重规划 |

---

## 项目结构

```
ros2/
├── demo_navigation.py              # 主入口: --map / --explain / --run / --viz
├── README.md                       # 你正在看的这个文件
├── package.xml, setup.py            # ROS2 包配置
└── ros2_nav_course/
    ├── utils/
    │   ├── mock_ros2.py             # ROS2 5原语: Node/Topic/Service/Action/Parameter
    │   └── visualizer.py            # 三面板: 地图 + 状态 + D435 深度视角
    ├── simulation/
    │   ├── world_2d.py              # 10m×10m 房间, 桌子/椅子/柜子/墙
    │   └── robot_model.py           # 差速轮底盘 x'=x+v·cosθ·dt
    ├── perception/
    │   └── sensor_node.py           # D435 模拟 (30Hz, 噪声)
    ├── slam/
    │   ├── rtabmap_node.py          # RTAB-Map 三级记忆 + /get_map Service
    │   └── costmap_node.py          # 占据栅格 + 距离变换膨胀
    ├── decision/
    │   └── decision_node.py         # Behavior Tree + /navigate_to_goal Action
    ├── planning/
    │   └── astar_node.py            # A* (f=g+h, 8邻域, 对角线启发式)
    └── control/
        └── control_node.py          # split (MPC+PID) / unified (UnifiedMPC)
                                     # Parameter: kp/ki/kd/max_v
```

---

## 可以折腾的

```bash
# 对比两种控制模式
python demo_navigation.py --viz --mode=split      # 看 PID P/I/D
python demo_navigation.py --viz --mode=unified    # 一体化 MPC

# 单独测试每个模块
python ros2_nav_course/perception/sensor_node.py
python ros2_nav_course/slam/rtabmap_node.py
python ros2_nav_course/planning/astar_node.py
python ros2_nav_course/control/control_node.py --mode=split
python ros2_nav_course/control/control_node.py --mode=unified

# 调 PID 参数: 打开 control/control_node.py, 搜 LongitudinalPID.__init__
#   kp=2.0  油门更猛
#   kd=0.3  减少超调

# 改目标: 打开 demo_navigation.py, 搜 "workstation"
#   → 改成 "kitchen" 或 "toolbox"

# 加障碍物: 打开 simulation/world_2d.py, 搜 _build_default_scene
#   self.obstacles.append(Obstacle(5, 5, 3.0, 0.5, "wall"))
```

---

## 跑通结果

```
Mode        Steps    Tracking error     Notes
────        ─────    ──────────────      ─────
split        1991    avg 0.031m          先被桌子堵, 倒车后绕开
unified      1099    avg 0.016m          更快, 一体化MPC
```
