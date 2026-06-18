"""
机器人运动模型 — 差速轮底盘
============================
你小车的运动学: 差速驱动 (Differential Drive)

    左轮速度 ωL  ─┐
                   ├─→ 机器人 (v, ω)
    右轮速度 ωR  ─┘

    正向:  左轮和右轮同速 → 直走
          左轮慢右轮快 → 右转
          左轮右轮反向 → 原地转

    这和 MPC 里的自行车模型不同:
        差速轮: 可以原地转 (你这种)
        自行车: 不能原地转 (汽车)

所以横向控制对你来说很简单:
    ω = (v_R - v_L) / 车轮间距
"""

import math
import numpy as np
from typing import Tuple


class DifferentialDriveRobot:
    """
    差速轮机器人模型。

    状态: (x, y, θ)
    控制: (v, ω) — 线速度和角速度

    运动学 (离散时间):
        x' = x + v * cos(θ) * dt
        y' = y + v * sin(θ) * dt
        θ' = θ + ω * dt

    这是 MPC 预测模型的基础。
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0,
                 wheel_base: float = 0.3,  # 轮间距 30cm
                 max_v: float = 1.0,       # 最大速度 1m/s
                 max_omega: float = 2.0):  # 最大角速度 2rad/s
        self.x = x
        self.y = y
        self.theta = theta
        self.wheel_base = wheel_base
        self.max_v = max_v
        self.max_omega = max_omega
        self.v = 0.0
        self.omega = 0.0

    def set_velocity(self, v: float, omega: float):
        """设置速度, 带限幅"""
        self.v = np.clip(v, -self.max_v, self.max_v)
        self.omega = np.clip(omega, -self.max_omega, self.max_omega)

    def update(self, dt: float):
        """走一步: x' = x + f(x, u) * dt"""
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.omega * dt
        # 归一化角度
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def get_pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    def get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x, self.y, self.theta = x, y, theta
        self.v, self.omega = 0.0, 0.0

    @staticmethod
    def forward_model(state: np.ndarray, v: float, omega: float,
                       dt: float) -> np.ndarray:
        """
        正向运动学 (MPC 要用这个来预测未来状态)

        输入:
            state: (x, y, θ)
            v: 线速度
            ω: 角速度
            dt: 时间步

        输出:
            next_state: (x', y', θ')

        这是 MPC 里最核心的函数。
        MPC 靠这个模型来 "想象未来":
            "如果我现在以 v=0.5, ω=0.3 走,
             0.1 秒后我在 (?, ?, ?)"
        """
        x, y, theta = state
        next_x = x + v * math.cos(theta) * dt
        next_y = y + v * math.sin(theta) * dt
        next_theta = theta + omega * dt
        return np.array([next_x, next_y, next_theta])


if __name__ == "__main__":
    robot = DifferentialDriveRobot()
    print("=" * 60)
    print("差速轮机器人运动学")
    print("=" * 60)

    print(f"\n初始位置: ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f})")

    # 直走 2 秒
    robot.set_velocity(0.5, 0.0)
    for _ in range(20):
        robot.update(0.1)
    print(f"直走 2 秒: ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f})")

    # 右转 1 秒
    robot.set_velocity(0.0, 0.5)
    for _ in range(10):
        robot.update(0.1)
    print(f"右转 1 秒: ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f})")

    # 直走 2 秒
    robot.set_velocity(0.5, 0.0)
    for _ in range(20):
        robot.update(0.1)
    print(f"再直走 2 秒: ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f})")

    print(f"\n→ 这就是你的轮式底盘的简化运动模型")
    print(f"→ MPC 控制器的核心: 用 forward_model() 预测未来位置")
