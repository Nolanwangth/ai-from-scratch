# 运行: conda activate diffusion && python utils/mps_utils.py
"""
MPS (Metal Performance Shaders) 加速工具
=========================================
Mac 没有 NVIDIA GPU (CUDA)，但 Apple Silicon (M1/M2/M3/M4/M5) 有 MPS。
MPS 是 Apple 的 GPU 加速框架，PyTorch 通过 torch.backends.mps 支持。

使用方式:
    from utils.mps_utils import get_device, print_device_info
    device = get_device()
    model.to(device)
"""

import torch


def get_device() -> torch.device:
    """
    自动选择最佳设备: MPS > CPU

    Returns:
        torch.device: 最佳可用设备
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_device_info() -> None:
    """打印当前设备信息，用于调试"""
    device = get_device()
    print(f"🔥 使用设备: {device}")

    if device.type == "mps":
        print(f"   MPS (Metal Performance Shaders) — Apple GPU 加速")
        print(f"   PyTorch 版本: {torch.__version__}")
        # MPS 内存没有直接查询 API，但可以查看当前分配
        print(f"   当前 MPS 内存分配: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")
        print(f"   MPS 驱动内存分配: {torch.mps.driver_allocated_memory() / 1024**2:.1f} MB")
    else:
        print(f"   ⚠️ MPS 不可用，使用 CPU (会慢一些)")


def mps_synchronize() -> None:
    """
    同步 MPS 操作。
    MPS 是异步计算的（像 CUDA），调用此函数确保所有操作完成。
    在 benchmarking 时非常重要。
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark_mps_vs_cpu(func, *args, **kwargs) -> None:
    """
    对比 MPS 和 CPU 的速度差异。

    用法:
        def matmul(size):
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            return a @ b

        benchmark_mps_vs_cpu(matmul, 1000)
    """
    import time

    # MPS benchmark
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # 需要确保 tensor 创建在 MPS 上
        # 这里只测相对速度
        start = time.time()
        torch.mps.synchronize()
        result = func(*args, **kwargs)
        torch.mps.synchronize()
        mps_time = time.time() - start
        print(f"⚡ MPS 耗时: {mps_time:.4f}s")

    # CPU benchmark
    start = time.time()
    result = func(*args, **kwargs)
    cpu_time = time.time() - start
    print(f"🐢 CPU 耗时: {cpu_time:.4f}s")

    if torch.backends.mps.is_available():
        speedup = cpu_time / mps_time
        print(f"🚀 MPS 加速比: {speedup:.1f}x")


if __name__ == "__main__":
    print_device_info()
