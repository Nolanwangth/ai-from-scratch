# 这个项目怎么学

目标很简单：学会当前这套代码怎么跑、每个模块负责什么、数据怎么流、split 和 unified 有什么区别。

不要先逐行读，先按主链路学。

## 1. 先看总流程

先读：

```text
demo_navigation.py
```

这个文件是整个项目的“导演”。它把世界、机器人、SLAM、规划、控制都创建出来，然后一帧一帧往前跑。

先只抓住这条链：

```text
Perception
→ SLAM
→ Costmap
→ Decision
→ A*
→ Control
→ Robot
```

这条链的意思是：

```text
Perception 负责产生传感器/里程计数据。
SLAM 负责根据机器人看到的东西慢慢建地图。
Costmap 负责把地图变成“哪里安全、哪里危险”。
A* 负责在地图上找一条路。
Control 负责让机器人真的沿着这条路走。
Robot update 负责更新机器人位置。
```

你要先明白频率：

```text
A* 不是 50Hz，控制器才是 50Hz。
A* 平时低频重规划，遇到障碍事件立即重规划。
```

也就是说：

```text
控制器像司机的手，每一小步都在修方向和速度。
A* 像导航软件，不需要每一毫秒都重新算路线。
```

现在代码里的大概节奏是：

```text
Control: 50Hz
Perception: 约 25Hz
SLAM + costmap: 约 2.5Hz
A*: 每 2 秒维护重规划一次, 动态障碍出现时额外重规划
```

当前跑通基线：

```text
split:   863 步到目标, 误差约 0.29m
unified: 671 步到目标, 误差约 0.30m
```

## 2. 学 ROS 核心思想

读：

```text
utils/mock_ros2.py
```

这个文件是项目里的“假 ROS2”。它让你不用真的装 ROS，也能理解 ROS 的核心思想。

只学这 5 个概念：

```text
Node       一个模块一个节点
Topic      高频数据流
Service    请求一次，返回一次
Action     长任务，比如导航到目标
Parameter  运行时调参数
```

在这个项目里可以这样理解：

```text
Node:
  Perception、SLAM、Costmap、Planner、Controller 都是独立模块。

Topic:
  /odom、/map、/costmap、/plan、/cmd_vel 这种持续流动的数据。

Service:
  /get_map，别人问 SLAM：“当前地图给我一份。”

Action:
  /navigate_to_goal，导航到目标点是一个长任务，不是瞬间完成。

Parameter:
  kp、ki、kd、max_v，可以运行时调控制参数。
```

看完这里，你要懂一句话：

```text
ROS 的核心不是某个算法，而是把机器人系统拆成很多节点，用标准通信方式连接起来。
```

## 3. 学地图线：地图怎么来

按顺序看：

```text
perception/sensor_node.py
slam/rtabmap_node.py
slam/costmap_node.py
planning/astar_node.py
```

这一组文件回答的是：

```text
机器人怎么知道哪里能走？
```

理解成这条线：

```text
传感器/里程计
→ SLAM 增量建图
→ costmap 加安全边界
→ A* 找路径
```

每个文件看什么：

```text
perception/sensor_node.py:
  看它怎么发布 /odom。
  这里的 /odom 是带噪声的机器人位姿。

slam/rtabmap_node.py:
  看它怎么维护一张 -1/0/100 的地图。
  -1 = 未知
  0 = 自由
  100 = 障碍

slam/costmap_node.py:
  看它怎么把障碍物膨胀。
  因为机器人有半径，不能贴着桌子边缘走。
  当前 costmap:
    unknown = 80
    free = 0
    inflated = 120
    lethal = 254

planning/astar_node.py:
  看 A* 怎么在 costmap 上找路径。
```

重点结论：

```text
SLAM 现在是增量 2D occupancy mapping，不是真 RTAB-Map。
Costmap 是给 A* 和控制器用的安全地图。
A* 输出的是 route/reference path，不直接开车。
```

小白理解版：

```text
SLAM: 慢慢画地图。
Costmap: 在地图上标出危险区。
A*: 在危险区外找一条路。
```

## 4. 学控制线：车怎么动

重点读：

```text
control/control_node.py
```

这个文件回答的是：

```text
有了 A* 路径之后，机器人下一步到底该给多少速度和角速度？
```

先学 split，因为它最好理解：

```text
A* path
→ SpeedProfiler 速度规划
→ LateralMPC 控方向
→ PID 控速度
```

split 的意思是把问题拆开：

```text
SpeedProfiler:
  这段路该快还是慢？
  当前它只管曲率和接近目标减速。
  障碍物实时减速交给控制层。

LateralMPC:
  方向该怎么转？

PID:
  怎么把当前速度追到目标速度？
```

再学 unified：

```text
A* path
→ UnifiedMPC 同时决定 v 和 omega
```

unified 的意思是不再把速度和转向分开，而是一起想：

```text
我现在应该用什么 v？
我现在应该用什么 omega？
这组 v/omega 未来几步会不会撞？
会不会偏离路径？
会不会太快？
```

一句话记住：

```text
split = 显式分层，容易理解。
unified = 统一优化，更像现代 MPC 思想。
```

你要最终明白：

```text
split 里有单独 SpeedProfiler。
unified 里没有单独 SpeedProfiler，因为速度选择被合进 MPC 了。
```

现在 unified 为了跑得快一点，做了两个工程化处理：

```text
1. path 下采样: A* 点太密, 控制器不用每个 5cm 点都看。
2. MPC 候选减少: 126 个 (v, omega) 候选, 预测 6 步。
```

这不改变 unified 的核心思想：还是同时决定速度和转向。

## 5. 学决策：什么时候重规划

读：

```text
decision/decision_node.py
```

这个文件回答的是：

```text
如果路被堵了，系统应该继续走、重规划，还是放弃？
```

只记住：

```text
BT 不负责开车。
BT 负责继续走、重规划、恢复、放弃。
```

BT 是 Behavior Tree，行为树。

你可以把它理解成一个高层判断器：

```text
路径正常？
  继续跟踪路径。

路径被堵？
  触发 A* 重规划。

恢复失败？
  放弃。
```

注意：

```text
控制层现在不会频繁清空全局路径。
它只做近障碍降速。
真正的重规划主要由周期维护和动态障碍触发。
```

## 6. 最后做三个实验

### 实验 1：换目标

改：

```text
workstation → kitchen / toolbox
```

看 A* 路径怎么变。

### 实验 2：对比 split / unified

跑：

```bash
python demo_navigation.py --run --mode=split
python demo_navigation.py --run --mode=unified

# 单独看动态障碍 + BT 重规划时再加这个
python demo_navigation.py --run --mode=split --dynamic
```

看：

```text
步数
跟踪误差
是否重规划
是否更平滑
```

当前正常结果应该接近：

```text
split:   1000~1100 步左右到达, 碰撞帧数 0
unified: 550~650 步左右到达, 碰撞帧数 0
```

这个实验的目的：

```text
看传统分层控制和统一 MPC 的行为差异。
```

### 实验 3：改 unified cost

改 `UnifiedMPC._rollout()` 里的权重。

看：

```text
更贴路径？
更保守？
更快？
更容易靠近障碍？
```

这个实验的目的：

```text
理解 MPC 的核心就是：预测未来，然后用 cost function 选最好的控制。
```

## 你最终要能说清楚

```text
1. ROS 为什么要拆 Node？
2. Topic / Service / Action / Parameter 分别适合什么？
3. SLAM、costmap、A*、control 各自负责什么？
4. split 和 unified 的区别是什么？
5. 为什么 A* 低频，control 高频？
```

能讲清楚这 5 个问题，这个项目就学到核心了。
