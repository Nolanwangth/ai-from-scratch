# 运行: conda activate ros2_nav && pip install -e .
"""
ROS2 包安装配置。

ROS2 的 setup.py 比普通 Python 包多了 entry_points:
    console_scripts: 注册可执行节点
        装在系统后, ros2 run ros2_nav_course demo_navigation 就能启动

但本项目设计为:
    1. 如果没有 ROS2:  用自带的 mock_ros2 模拟, python 直接跑
    2. 如果有 ROS2:    用这个 setup.py 安装, ros2 run 启动

这种双模式让你在 Mac 上也能学 ROS2 的全部概念。
"""

from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'ros2_nav_course'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib'],
    zip_safe=True,
    author='nolan',
    author_email='nolan@example.com',
    description='ROS2 导航教学: 感知 → RTAB-Map SLAM → Behavior Tree → A* → MPC + PID',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 可执行节点注册 —— 有 ROS2 时用 ros2 run 启动
            'demo_navigation = ros2_nav_course.demo_navigation:main',
            'sensor_node = ros2_nav_course.perception.sensor_node:main',
            'rtabmap_node = ros2_nav_course.slam.rtabmap_node:main',
            'costmap_node = ros2_nav_course.slam.costmap_node:main',
            'decision_node = ros2_nav_course.decision.decision_node:main',
            'planner_node = ros2_nav_course.planning.astar_node:main',
            'control_node = ros2_nav_course.control.control_node:main',
        ],
    },
)
