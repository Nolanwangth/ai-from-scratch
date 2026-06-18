# 运行: conda activate ros2_nav && python ros2_nav_course/perception/sensor_node.py
"""
感知节点 — D435 传感器模拟
============================
ROS2 感知层的教学实现。

真实场景:
    订阅 D435 的 /camera/infra1/image_rect_raw, /camera/depth/image_rect_raw
    发布 TF 变换 (camera → base_link → world)

模拟场景 (本文件):
    从世界模型获取真实位姿, 模拟 D435 的观测
    发布: /odom (里程计), /camera/image (RGB 替代), /camera/depth (深度图)

核心教学点:
    1. ROS2 Node 怎么写 — 继承 Node, create_publisher, create_timer
    2. 传感器数据的发布频率 — 30Hz (比控制层50Hz慢, 但够用)
    3. 数据格式 — 和真实 ROS2 消息格式一致

为什么感知独立成节点:
    传感器有自己的驱动、标定参数、失败模式
    和 SLAM/规划/控制解耦 → 哪个坏了只换那个
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import random
from ros2_nav_course.utils.mock_ros2 import Node, Pose, Twist, Odometry, Image
from ros2_nav_course.utils.mock_ros2 import log_info, log_ok


class PerceptionNode(Node):
    """
    感知节点: 模拟 D435 传感器。

    发布的 Topic:
        /odom           — 里程计 (位姿 + 速度)
        /camera/rgb     — RGB 图像 (模拟)
        /camera/depth   — 深度图 (模拟)

    真实 D435 做这些:
        红外投影仪(IR Projector) 发射图案
        左右红外相机(IR Camera) 捕捉反射图案
        双目匹配 → 视差 → 深度

    模拟世界做这些:
        直接用世界的真实位置加噪声 = 模拟 D435 观测
        噪声大小 ~ D435 的真实噪声 (距离越远, 噪声越大)
    """

    def __init__(self, world, robot, hz: float = 30.0):
        super().__init__("perception_node")
        self.world = world
        self.robot = robot
        self.hz = hz

        # 噪声参数 (模拟 D435 的特性)
        self.depth_noise_ratio = 0.01   # 距离的 1% 噪声
        self.angular_noise = 0.02       # ~1° 角度噪声
        self.linear_noise = 0.03        # 3cm 线速度噪声

        # ── 创建发布者 (和真实 ROS2 一样的 API) ──
        self.create_publisher("/odom", "Odometry")
        self.create_publisher("/camera/rgb", "Image")
        self.create_publisher("/camera/depth", "Image")

        # ── 定时器: 每 1/hz 秒发布一次 ──
        self.create_timer(1.0 / hz, self._sensor_callback)

        log_ok(f"[perception] D435 模拟寄存器 @ {hz}Hz")
        log_info(f"  [perception] Publishing: /odom, /camera/rgb, /camera/depth")

    def _sensor_callback(self):
        """
        定时器回调: 每 33ms 执行一次 (30Hz)。

        这个函数模拟 D435 的数据采集流程:
            1. 读真实位姿 + 加噪声 → 里程计
            2. 从位姿生成模拟 RGB + 深度图

        真实 D435 也在 ~30fps 出图 (640x480)。
        """
        x, y, theta = self.robot.get_pose()
        v, omega = self.robot.v, self.robot.omega

        # ── 1. 里程计 ──
        odom = Odometry(
            pose=Pose(
                x=x + random.gauss(0, self.linear_noise),
                y=y + random.gauss(0, self.linear_noise),
                theta=theta + random.gauss(0, self.angular_noise),
            ),
            linear_vel=v + random.gauss(0, self.linear_noise * 0.5),
            angular_vel=omega + random.gauss(0, self.angular_noise * 0.5),
        )
        self.publish("/odom", odom)

        # ── 2. RGB 图像 (模拟) ──
        # 真实: 640x480, 3 通道 uint8
        # 这里: 直接从位姿生成一张模拟图
        img = Image(width=640, height=480, data=f"rgb_frame({x:.2f},{y:.2f})")
        self.publish("/camera/rgb", img)

        # ── 3. 深度图 (模拟) ──
        # D435 的深度噪声: ~1% 在 2m 范围
        # 这里: 附加上噪声的位姿信息
        depth_noise = random.gauss(0, max(0.01, self.depth_noise_ratio * 2.0))
        depth = Image(width=640, height=480,
                      data=f"depth({x+depth_noise:.3f},{y+depth_noise:.3f})")
        self.publish("/camera/depth", depth)


if __name__ == "__main__":
    print("=" * 60)
    print("感知节点 (PerceptionNode) — D435 传感器模拟")
    print("=" * 60)
    print("""
    这个节点模仿真实 D435 的工作:
        每隔 ~33ms:
            读真实位姿
            加 D435 级别的噪声
            发布到 /odom, /camera/rgb, /camera/depth

    在完整系统里:
        SLAM 节点订阅这些话题 → 建图 + 定位
    """)

    # 简单演示
    from ros2_nav_course.simulation.world_2d import World2D
    from ros2_nav_course.simulation.robot_model import DifferentialDriveRobot

    world = World2D()
    robot = DifferentialDriveRobot(1.0, 2.0, 0.5)

    node = PerceptionNode(world, robot, hz=5.0)

    # 机器人走几步
    robot.set_velocity(0.3, 0.1)
    for i in range(15):
        robot.update(0.1)
        node.tick(0.1)
        if i % 5 == 0:
            odom_topic = node.get_logger()
            x, y, t = robot.get_pose()
            print(f"  步 {i:2d}: 真实({x:.2f}, {y:.2f}, {t:.2f})  "
                  f"发布到 /odom")

    node.destroy()
    print("\n✅ 感知节点演示完成")
    print("   下一个节点: SLAM (slam/rtabmap_node.py)")
