"""
ROS2 核心概念教学 + 双模式运行 (mock / real)
=============================================
这是本项目的"教学层"。它模拟了 ROS2 的 4 个核心概念:

    1. Node    — 独立的进程/模块, 一个机器人系统 = N 个 Node
    2. Topic   — 发布/订阅通信, 一对多, 异步
    3. Service — 请求/响应通信, 一对一, 同步
    4. Action  — 带反馈的长时间任务, 可取消

没有 ROS2 (Mac):  用这个 mock 跑, 所有概念一致
有 ROS2 (Linux):  可以切换到真实 rclpy, 代码不改

使用方法:
    from ros2_nav_course.utils.mock_ros2 import Node, log_info, ...

Node 是你唯一需要 import 的基类。
其他 (Topic, Service, Action) 都在 Node 内部创建。
"""

import time
import threading
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import traceback

# ============================================================================
# 全局设置 — 控制用 mock 还是 real ROS2
# ============================================================================
USE_REAL_ROS2 = False  # 没装 ROS2 的 Mac 保持 False

# 如果装了 rclpy 且用户想用真实的, 可以 import
try:
    if USE_REAL_ROS2:
        import rclpy
        from rclpy.node import Node as RclpyNode
        from rclpy.qos import QoSProfile
        HAS_REAL_ROS2 = True
    else:
        HAS_REAL_ROS2 = False
except ImportError:
    HAS_REAL_ROS2 = False


# ============================================================================
# 教学基础: ROS2 的 4 个核心概念
# ============================================================================
# ┌────────────────────────────────────────────────────────────────────────┐
# │ ROS2 的哲学:                                                          │
# │                                                                       │
# │   "一个机器人系统 = 多个独立的小程序(Node)互相通信"                     │
# │                                                                       │
# │   感知 Node 只管: 读 D435 → 发 RGB + 深度图                         │
# │   SLAM Node 只管: 收 RGB + 深度 → 出位姿 + 地图                     │
# │   规划 Node 只管: 收位姿 + 地图 + 目标 → 出路径                      │
# │   控制 Node 只管: 收路径 + 位姿 → 出速度指令                         │
# │                                                                       │
# │   每个 Node 可以独立启动、独立调试、独立崩溃。                          │
# │   这就是 ROS2 的 "模块化" 的本质。                                     │
# └────────────────────────────────────────────────────────────────────────┘


# ────────────────────────────────────────────────────────────────────────
# 概念 1: Topic (话题) — 发布/订阅通信
# ────────────────────────────────────────────────────────────────────────
# 场景: D435 每帧发布一次 /camera/image 话题
#       SLAM Node 订阅同一个话题, 收到每帧图像
#
# Pub: "我每 33ms 发一次图像, 谁想收谁收"
# Sub: "我对图像感兴趣, 来了告诉我"
#
# 特点:
#   - 一对多: 一个发布者, 多个订阅者
#   - 异步: 发布者不等订阅者处理完就继续
#   - 匿名: 双方都不知道对面是谁

@dataclass
class Topic:
    """
    一个话题 = 一个"消息通道"

    属性:
        name:  话题名 (如 /camera/image, /cmd_vel)
        msg_type: 消息类型 (如 Image, Twist)
        msg:    最新一帧消息

    ROS2 里的命名约定:
        /camera/image   — 图像话题
        /scan           — 激光雷达
        /odom           — 里程计
        /cmd_vel        — 速度指令 (控制用)
        /map            — SLAM 地图
    """
    name: str
    msg_type: str = "any"
    msg: Any = None
    _subscribers: List[Callable] = field(default_factory=list)


class TopicManager:
    """
    全局话题管理器 — 模拟 ROS2 的 "话题总线"。

    在真正的 ROS2 里, 话题总线由 ROS2 Daemon 管理。
    这里用全局 dict 模拟。
    """

    _topics: Dict[str, Topic] = {}

    @classmethod
    def advertise(cls, name: str, msg_type: str = "any") -> Topic:
        """创建一个话题 (ROS2 里是自动的)"""
        if name not in cls._topics:
            cls._topics[name] = Topic(name=name, msg_type=msg_type)
        return cls._topics[name]

    @classmethod
    def publish(cls, name: str, msg: Any, msg_type: str = "any"):
        """发布一条消息到话题"""
        if name not in cls._topics:
            cls.advertise(name, msg_type)
        topic = cls._topics[name]
        topic.msg = msg
        # 通知所有订阅者
        for cb in topic._subscribers:
            try:
                cb(msg)
            except Exception as e:
                print(f"  [Topic:{name}] 订阅者错误: {e}")
                traceback.print_exc()

    @classmethod
    def subscribe(cls, name: str, callback: Callable, msg_type: str = "any"):
        """订阅一个话题"""
        topic = cls.advertise(name, msg_type)
        topic._subscribers.append(callback)

    @classmethod
    def get_topic(cls, name: str) -> Optional[Topic]:
        return cls._topics.get(name)


# ────────────────────────────────────────────────────────────────────────
# 概念 2: Service (服务) — 请求/响应通信
# ────────────────────────────────────────────────────────────────────────
# 场景: 规划 Node 问 SLAM Node "当前地图能发我一份吗?"
#       SLAM Node 回复: "给你"
#
# 和 Topic 的区别:
#   Topic: 主动发布, 订阅者被动收
#   Service: 显式请求, 等回复
#
# 像函数调用: 你传参, 它返回结果

class ServiceServer:
    """
    服务端: 收到请求 → 处理 → 回复

    场景: SLAM Node 提供 /get_map 服务
          规划 Node 请求这个服务 → SLAM 回复地图
    """

    def __init__(self, service_name: str, handler: Callable):
        self.name = service_name
        self.handler = handler
        ServiceManager.register(service_name, self)
        log_info(f"Service /{service_name} ready")

    def call(self, request: Any) -> Any:
        return self.handler(request)


class ServiceClient:
    """
    客户端: 发请求 → 等回复

    场景: 规划 Node 调用 /get_map 服务
    """

    def __init__(self, service_name: str):
        self.name = service_name
        server = ServiceManager.get(service_name)
        if server is None:
            log_warn(f"Service /{service_name} not available yet")

    def call(self, request: Any, timeout: float = 5.0) -> Optional[Any]:
        """调用服务, 等回复"""
        server = ServiceManager.get(self.name)
        if server is None:
            log_warn(f"Service /{self.name} not available")
            return None
        return server.call(request)


class ServiceManager:
    """全局服务注册表"""
    _services: Dict[str, ServiceServer] = {}

    @classmethod
    def register(cls, name: str, server: ServiceServer):
        cls._services[name] = server

    @classmethod
    def get(cls, name: str) -> Optional[ServiceServer]:
        return cls._services.get(name)


# ────────────────────────────────────────────────────────────────────────
# 概念 3: Action (动作) — 带反馈的长时间任务
# ────────────────────────────────────────────────────────────────────────
# 场景: "导航到工位"  — 这是一个 Action
#       启动时: 给你目标点
#       过程中: 每帧给你反馈 "走到哪了, 还有多远"
#       完成时: "到了"
#       可取消: "停!"
#
# 和 Service 的区别:
#   Service: 请求 → 等 → 回复 (短任务)
#   Action:  请求 → 过程中持续反馈 → 完成/取消 (长任务)

@dataclass
class ActionGoal:
    """Action 的请求"""
    target: Any

@dataclass
class ActionResult:
    """Action 的最终结果"""
    success: bool
    message: str = ""

@dataclass
class ActionFeedback:
    """Action 的中间反馈"""
    progress: float = 0.0
    status: str = ""

class ActionServer:
    """
    Action 服务端: 接收目标 → 持续跑 → 反馈进度 → 完成

    场景: 导航 Node
        receive_goal: 收到 "去工位"
        execute:      开始跑 A* + MPC, 每帧发反馈
        cancel:       收到 "停!"
    """

    def __init__(self, action_name: str, execute_callback: Callable,
                 cancel_callback: Optional[Callable] = None):
        self.name = action_name
        self.execute_callback = execute_callback
        self.cancel_callback = cancel_callback
        self._current_goal: Optional[ActionGoal] = None
        self._cancelled = False
        ActionManager.register(action_name, self)
        log_info(f"Action /{action_name} ready")

    def start(self, goal: ActionGoal):
        """开始执行动作 (在新线程)"""
        self._current_goal = goal
        self._cancelled = False
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def cancel(self):
        """取消当前动作"""
        self._cancelled = True
        if self.cancel_callback:
            self.cancel_callback()

    def _run(self):
        feedback = ActionFeedback()
        try:
            result = self.execute_callback(self._current_goal, feedback,
                                            lambda: self._cancelled)
            if not self._cancelled:
                log_info(f"Action /{self.name} 完成: {result.message}")
        except Exception as e:
            log_error(f"Action /{self.name} 错误: {e}")


class ActionClient:
    """
    Action 客户端: 发目标 → 收反馈 → 等完成

    场景: 决策 Node 发 "去工位" 给导航 Node
    """

    def __init__(self, action_name: str):
        self.name = action_name
        self._server = ActionManager.get(action_name)

    def send_goal(self, target: Any, feedback_callback: Optional[Callable] = None):
        """发送目标, 注册反馈回调"""
        if self._server is None:
            log_warn(f"Action /{self.name} not available")
            return
        self._feedback_callback = feedback_callback
        self._server.start(ActionGoal(target=target))

    def cancel(self):
        if self._server:
            self._server.cancel()


class ActionManager:
    """全局 Action 注册表"""
    _actions: Dict[str, ActionServer] = {}

    @classmethod
    def register(cls, name: str, server: ActionServer):
        cls._actions[name] = server

    @classmethod
    def get(cls, name: str) -> Optional[ActionServer]:
        return cls._actions.get(name)


# ────────────────────────────────────────────────────────────────────────
# 概念 4: Parameter (参数) — 节点的配置变量
# ────────────────────────────────────────────────────────────────────────
# 场景: PID 的 Kp/Ki/Kd 应该做成参数, 运行中可调
# 真实 ROS2: node.declare_parameter("kp", 1.5)
#            node.get_parameter("kp").value

class ParameterManager:
    """全局参数管理器 — 每个 Node 的参数存在这里."""
    _params: Dict[str, Any] = {}

    @classmethod
    def declare(cls, node_name: str, name: str, value: Any, description: str = ""):
        full_name = f"{node_name}.{name}"
        cls._params[full_name] = value
        log_info(f"  [{node_name}] Param /{name} = {value}")

    @classmethod
    def set_param(cls, node_name: str, name: str, value: Any):
        full_name = f"{node_name}.{name}"
        if full_name in cls._params:
            cls._params[full_name] = value
            log_info(f"  [{node_name}] Param /{name} updated -> {value}")

    @classmethod
    def get(cls, node_name: str, name: str) -> Optional[Any]:
        return cls._params.get(f"{node_name}.{name}")


# ────────────────────────────────────────────────────────────────────────
# 概念 5: Node (节点) — ROS2 的最小单元
# ────────────────────────────────────────────────────────────────────────
# Node = 一个独立的 "小程序", 有自己的:
#   - 名字 (唯一标识)
#   - 发布者 (往外发消息)
#   - 订阅者 (收别人消息)
#   - 服务端/客户端 (请求式通信)
#   - Action 端 (长任务)
#   - 日志 (打印信息)
#
# 整个机器人系统 = 一堆 Node 各干各的 + 通过 Topic/Service/Action 通信

class Node:
    """
    ROS2 Node 的教学模拟。

    用法:
        class MyNode(Node):
            def __init__(self):
                super().__init__("my_node")
                self.pub = self.create_publisher("topic_name", "str")
                self.sub = self.create_subscription("topic_name", self.callback)

            def callback(self, msg):
                self.get_logger().info(f"收到: {msg}")

    真实 ROS2 的区别:
        真实: Node 在 spin() 里循环, 靠 rclpy 的事件循环驱动
        Mock: Node 的 tick() 手动调用, 靠主循环驱动

        但核心概念完全一致。
    """

    def __init__(self, node_name: str):
        self.node_name = node_name
        self._publishers: List[tuple] = []     # [(topic_name, msg_type)]
        self._subscriptions: List[tuple] = []  # [(topic_name, callback)]
        self._services: List[ServiceServer] = []
        self._clients: List[ServiceClient] = []
        self._action_servers: List[ActionServer] = []
        self._action_clients: List[ActionClient] = []
        self._timers: List[tuple] = []          # [(interval, callback)]
        self._rate = None                       # Rate 对象
        self._running = True

        log_info(f"Node [{node_name}] created")

    # ── 发布者 ──
    def create_publisher(self, topic_name: str, msg_type: str = "any"):
        """
        创建一个发布者。

        ROS2 里: pub = node.create_publisher(Image, '/camera/image', 10)
        这里:    pub = node.create_publisher('/camera/image', 'Image')

        参数 msg_type 只是标识, mock 里不做严格类型检查。
        真实 ROS2 会检查消息类型是否匹配。
        """
        TopicManager.advertise(topic_name, msg_type)
        self._publishers.append((topic_name, msg_type))
        log_info(f"  [{self.node_name}] Publisher: {topic_name}")
        return self  # 返回 self 方便链式调用

    def publish(self, topic_name: str, msg: Any):
        """发布消息到话题"""
        # 找到这个消息类型
        msg_type = "any"
        for t, mt in self._publishers:
            if t == topic_name:
                msg_type = mt
                break
        TopicManager.publish(topic_name, msg, msg_type)

    # ── 订阅者 ──
    def create_subscription(self, topic_name: str, callback: Callable,
                            msg_type: str = "any"):
        """
        订阅一个话题。

        ROS2 里: sub = node.create_subscription(Image, '/camera/image', cb, 10)
        这里:    sub = node.create_subscription('/camera/image', cb, 'Image')

        你提供的 callback 会在 publish 时被调用。
        """
        TopicManager.subscribe(topic_name, callback, msg_type)
        self._subscriptions.append((topic_name, callback))
        log_info(f"  [{self.node_name}] Subscription: {topic_name}")
        return self

    # ── 服务端 ──
    def create_service(self, service_name: str, handler: Callable):
        """创建一个服务端"""
        svr = ServiceServer(service_name, handler)
        self._services.append(svr)
        return svr

    def create_client(self, service_name: str) -> ServiceClient:
        """创建一个客户端"""
        client = ServiceClient(service_name)
        self._clients.append(client)
        return client

    # ── Action ──
    def create_action_server(self, action_name: str,
                              execute_callback: Callable,
                              cancel_callback: Optional[Callable] = None):
        """创建一个 Action 服务端"""
        server = ActionServer(action_name, execute_callback, cancel_callback)
        self._action_servers.append(server)
        return server

    def create_action_client(self, action_name: str) -> ActionClient:
        """创建一个 Action 客户端"""
        client = ActionClient(action_name)
        self._action_clients.append(client)
        return client

    # ── 参数 (Parameter) ──
    def declare_parameter(self, name: str, value: Any, description: str = ""):
        """声明一个参数, 运行时可通过 ros2 param set 修改."""
        ParameterManager.declare(self.node_name, name, value, description)

    def get_parameter(self, name: str) -> Optional[Any]:
        """读取参数值."""
        return ParameterManager.get(self.node_name, name)

    def set_parameter(self, name: str, value: Any):
        """修改参数值."""
        ParameterManager.set_param(self.node_name, name, value)

    # ── 定时器 ──
    def create_timer(self, interval_sec: float, callback: Callable):
        """
        创建一个定时器。

        ROS2 里: timer = node.create_timer(0.1, callback)
        这里: 一样

        定时器 = "每过 interval 秒, 执行一次 callback"
        这是 ROS2 里最主要的循环驱动方式。
        """
        self._timers.append((interval_sec, callback, 0.0))
        log_info(f"  [{self.node_name}] Timer: every {interval_sec}s")

    # ── 日志 ──
    def get_logger(self):
        """返回一个日志对象, 和 ROS2 的 rclpy.node.Node.get_logger() 一样"""
        return Logger(self.node_name)

    # ── 运行控制 ──
    def tick(self, dt: float):
        """
        处理定时器 (每帧调用)。

        这是 mock 和真实 ROS2 的唯一区别:
            真实: 在 rclpy.spin() 里自动处理
            Mock: 需要你手动调 tick(dt)

        你可以在主循环里:
            while True:
                for node in all_nodes:
                    node.tick(dt)
                time.sleep(dt)
        """
        for i, (interval, callback, elapsed) in enumerate(self._timers):
            new_elapsed = elapsed + dt
            if new_elapsed >= interval:
                try:
                    callback()
                except Exception as e:
                    log_error(f"[{self.node_name}] Timer error: {e}")
                new_elapsed = 0.0
            self._timers[i] = (interval, callback, new_elapsed)

    def destroy(self):
        """销毁节点 (清理资源)"""
        self._running = False
        log_info(f"Node [{self.node_name}] destroyed")


# ────────────────────────────────────────────────────────────────────────
# Logger (日志) — 和 ROS2 的 rclpy.logging 一样
# ────────────────────────────────────────────────────────────────────────

class Logger:
    """模拟 ROS2 的 get_logger()"""

    def __init__(self, name: str):
        self.name = name

    def info(self, msg: str):
        log_info(f"[{self.name}] {msg}")

    def warn(self, msg: str):
        log_warn(f"[{self.name}] {msg}")

    def error(self, msg: str):
        log_error(f"[{self.name}] {msg}")

    def debug(self, msg: str):
        log_info(f"[{self.name}] {msg}")


# ────────────────────────────────────────────────────────────────────────
# Spin — ROS2 的事件循环
# ────────────────────────────────────────────────────────────────────────
# ROS2 的核心是 spin() — 永不返回的事件循环
# 真实 rclpy.spin() 会阻塞, 处理回调
# 这里提供两种模式:

class Rate:
    """频率控制, 和 ROS2 rclpy.Rate 一样"""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.last = time.time()

    def sleep(self):
        """睡到下一帧"""
        now = time.time()
        elapsed = now - self.last
        sleep_time = self.period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last = time.time()


def spin(node: Node, hz: float = 30.0):
    """
    启动一个节点的事件循环。

    在真实 ROS2 里:
        rclpy.spin(node)
        会一直阻塞, 处理订阅回调和定时器

    这里:
        spin(node, hz)
        在一个新线程里跑, 不阻塞主程序
        30Hz = 每 33ms tick 一次
    """

    def _spin():
        rate = Rate(hz)
        while node._running:
            node.tick(1.0 / hz)
            rate.sleep()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    log_info(f"Node [{node.node_name}] spinning at {hz}Hz")
    return thread


def spin_once(nodes: List[Node], dt: float = 0.033):
    """手动 tick 一组节点, 不启动线程"""
    for node in nodes:
        node.tick(dt)


# ────────────────────────────────────────────────────────────────────────
# 消息类型 — ROS2 标准消息的教学简化版
# ────────────────────────────────────────────────────────────────────────
# ROS2 里消息类型 (msg) 定义在 .msg 文件里:
#   geometry_msgs/msg/Twist.msg:
#     Vector3 linear     # 线速度
#     Vector3 angular    # 角速度
#
# 这里用 dataclass 模拟, 结构一样但简单

@dataclass
class Twist:
    """速度指令 (geometry_msgs/Twist)"""
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0

    def __repr__(self):
        return f"Twist(v={self.linear_x:.2f}, ω={self.angular_z:.2f})"


@dataclass
class Pose:
    """位姿 (geometry_msgs/Pose)"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # 朝向 (弧度)

    def __repr__(self):
        return f"Pose({self.x:.2f}, {self.y:.2f}, {self.theta:.2f}rad)"


@dataclass
class Odometry:
    """里程计 (nav_msgs/Odometry)"""
    pose: Pose = field(default_factory=Pose)
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    timestamp: float = 0.0


@dataclass
class LaserScan:
    """激光雷达 (sensor_msgs/LaserScan)"""
    ranges: List[float] = field(default_factory=list)
    angle_min: float = -3.14      # -180°
    angle_max: float = 3.14       # +180°
    angle_increment: float = 0.017  # ~1° 分辨率
    range_max: float = 10.0       # 最大 10m


@dataclass
class Image:
    """图像 (sensor_msgs/Image)"""
    width: int = 640
    height: int = 480
    data: Any = None  # 模拟数据


@dataclass
class OccupancyGrid:
    """占据栅格地图 (nav_msgs/OccupancyGrid)"""
    width: int = 100
    height: int = 100
    resolution: float = 0.05       # 5cm 每格
    origin_x: float = -2.5
    origin_y: float = -2.5
    data: List[int] = field(default_factory=list)  # -1=未知, 0=空, 100=占

    def __post_init__(self):
        if not self.data:
            self.data = [0] * (self.width * self.height)

    def get(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y * self.width + x]
        return -1

    def set(self, x: int, y: int, val: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y * self.width + x] = val


@dataclass
class Path:
    """路径 (nav_msgs/Path) - 一系列带朝向的位姿"""
    poses: List[Pose] = field(default_factory=list)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, i):
        return self.poses[i]


# ────────────────────────────────────────────────────────────────────────
# TF (坐标变换) — ROS2 的坐标系管理
# ────────────────────────────────────────────────────────────────────────
# ROS2 里每个物体都有自己的坐标系:
#   map    — 世界坐标系 (固定)
#   odom   — 里程计坐标系 (以起点为原点)
#   base_link — 机器人本身的中心
#   camera_link — 相机中心
#   laser_link  — 激光雷达中心
#
# TF 负责维护这些坐标系之间的变换关系。
# 比如: 知道 base_link 在 map 里的位置 = 知道机器人在哪

class TFManager:
    """
    简化的 TF 管理器。

    真实 TF2: 广播/监听变换树, 支持任意坐标系
    这里:    只维护 map → odom → base_link 的链
    """

    def __init__(self):
        self._transforms: Dict[str, Dict[str, Pose]] = {}
        # 默认: base_link 在 map 的 (0,0,0)
        self.set_transform("map", "base_link", Pose(0, 0, 0))

    def set_transform(self, parent: str, child: str, pose: Pose):
        """设置父→子的变换"""
        if parent not in self._transforms:
            self._transforms[parent] = {}
        self._transforms[parent][child] = pose

    def get_transform(self, parent: str, child: str) -> Optional[Pose]:
        """获取父→子的变换"""
        return self._transforms.get(parent, {}).get(child)

    def lookup_transform(self, target_frame: str, source_frame: str) -> Optional[Pose]:
        """ROS2 TF2 的 lookupTransform 接口"""
        return self.get_transform(target_frame, source_frame)


# ────────────────────────────────────────────────────────────────────────
# 全局实例
# ────────────────────────────────────────────────────────────────────────

TF_MANAGER = TFManager()


# ────────────────────────────────────────────────────────────────────────
# 日志工具
# ────────────────────────────────────────────────────────────────────────

def log_info(msg: str):
    print(f"  ℹ️  {msg}")

def log_warn(msg: str):
    print(f"  ⚠️  {msg}")

def log_error(msg: str):
    print(f"  ❌ {msg}")

def log_ok(msg: str):
    print(f"  ✅ {msg}")


# ============================================================================
# 教学: ROS2 节点间的数据流
# ============================================================================

def explain_ros2_dataflow():
    print(f"""
    ROS2 导航系统 — 完整数据流 (Topic + Service + Action + Parameter)
    ====================================================================

    ═══ Topic (7 条) — 高频异步数据流 ═══

      Perception          SLAM               Costmap
      (sensor_node)       (rtabmap_node)     (costmap_node)
           │ /odom ──────────> │                   │
           │ /camera/*         │ /map ────────────> │
           │                   │                   │ /costmap ───────┐
           │                   │                   │                 │
           │                   │                   ▼                 ▼
           │                   │              Planner (A*)      Controller
           │                   │                   │                 │
           │                   │                   │ /plan ─────── > │
           │                   │                   │                 │ /cmd_vel
           │                   │                   │                 ▼
           │                   │                   │            Robot wheels

      Decision (decision_node):
          Topic: /goal_point → 发给 Planner
          Topic: /decision_state → 外部监控


    ═══ Service (1 条) — 请求/响应 ═══

      /get_map  (SLAM Node 提供)
          Client: "给我当前地图"
          Server: "这是地图数据"
          用在哪: Planner 初始化或别的 Node 需要地图快照时


    ═══ Action (1 条) — 带反馈的长任务 ═══

      /navigate_to_goal  (Decision Node 提供)
          Goal:    "导航到 (8.0, 8.0)"
          Feedback: 进度 (走到哪了, 还有多远)
          Result:   到达 / 失败 / 取消
          用在哪: 用户或上层系统下发导航指令


    ═══ Parameter — 节点配置, 运行时可调 ═══

      Controller Node 参数:
          /kp   = 1.5    PID 比例增益  (ros2 param set /control_node kp 2.0)
          /ki   = 0.3    PID 积分增益
          /kd   = 0.1    PID 微分增益
          /max_v = 1.0   最大速度 (m/s)
          用在哪: 运行时调 PID, 不需要重新编译/重启


    ═══ 5 个通信原语全用了 ═══

      Topic    ✅ 7 条 — 感知/SLAM/Costmap/决策/规划/控制全走 Topic
      Service  ✅ 1 条 — SLAM 提供 /get_map 给其他 Node 查询
      Action   ✅ 1 条 — Decision 提供 /navigate_to_goal 给上层调用
      Parameter ✅ 4 条 — ControlNode 的 kp/ki/kd/max_v, 运行时在线调
      Node     ✅ 6 个 — 各管一摊, 边界清晰
    """)
