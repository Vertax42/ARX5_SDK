#!/usr/bin/env python3
"""
测试/dev/video4的同步和异步读取性能
"""

import sys
import cv2
import time
from pathlib import Path

# 添加cameras目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "cameras"))


def test_video4_performance():
    """测试video4的性能"""
    print("=" * 60)
    print("测试/dev/video4相机性能")
    print("=" * 60)

    try:
        from cameras.opencv import OpenCVCamera
        from cameras.opencv.configuration_opencv import (
            OpenCVCameraConfig,
            ColorMode,
            Cv2Rotation,
        )

        camera_id = "/dev/video4"
        print(f"使用相机: {camera_id}")

        # 创建相机配置
        config = OpenCVCameraConfig(
            index_or_path=camera_id,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        )

        camera = OpenCVCamera(config)
        camera.connect(warmup=True)
        print("✓ 相机连接成功")

        print(f"相机信息:")
        print(f"  分辨率: {camera.width}x{camera.height}")
        print(f"  FPS: {camera.fps}")
        print(f"  颜色模式: {camera.color_mode}")

        print("\n" + "=" * 40)
        print("1. 同步读取性能测试")
        print("=" * 40)

        sync_times = []
        for i in range(20):
            start_time = time.perf_counter()
            frame = camera.read()
            end_time = time.perf_counter()
            read_time = (end_time - start_time) * 1000  # 转换为毫秒
            sync_times.append(read_time)
            if i < 10 or i % 5 == 0:
                print(f"  同步读取 {i+1}: {read_time:.2f}ms")

        avg_sync_time = sum(sync_times) / len(sync_times)
        min_sync_time = min(sync_times)
        max_sync_time = max(sync_times)
        print(f"  平均时间: {avg_sync_time:.2f}ms")
        print(f"  最小时间: {min_sync_time:.2f}ms")
        print(f"  最大时间: {max_sync_time:.2f}ms")

        print("\n" + "=" * 40)
        print("2. 异步读取性能测试")
        print("=" * 40)

        async_times = []
        for i in range(20):
            start_time = time.perf_counter()
            frame = camera.async_read(timeout_ms=1000)
            end_time = time.perf_counter()
            read_time = (end_time - start_time) * 1000  # 转换为毫秒
            async_times.append(read_time)
            if i < 10 or i % 5 == 0:
                print(f"  异步读取 {i+1}: {read_time:.2f}ms")

        avg_async_time = sum(async_times) / len(async_times)
        min_async_time = min(async_times)
        max_async_time = max(async_times)
        print(f"  平均时间: {avg_async_time:.2f}ms")
        print(f"  最小时间: {min_async_time:.2f}ms")
        print(f"  最大时间: {max_async_time:.2f}ms")

        print("\n" + "=" * 40)
        print("3. 连续读取测试")
        print("=" * 40)

        # 同步连续读取
        print("同步连续读取50帧...")
        sync_start = time.time()
        for i in range(50):
            frame = camera.read()
        sync_end = time.time()
        sync_total = sync_end - sync_start
        sync_fps = 50 / sync_total

        # 异步连续读取
        print("异步连续读取50帧...")
        async_start = time.time()
        for i in range(50):
            frame = camera.async_read(timeout_ms=1000)
        async_end = time.time()
        async_total = async_end - async_start
        async_fps = 50 / async_total

        print(f"\n同步读取: {sync_total:.2f}秒, 平均FPS: {sync_fps:.1f}")
        print(f"异步读取: {async_total:.2f}秒, 平均FPS: {async_fps:.1f}")

        print("\n" + "=" * 40)
        print("4. 性能对比总结")
        print("=" * 40)
        print(f"单次读取延迟:")
        print(
            f"  同步: {avg_sync_time:.2f}ms (范围: {min_sync_time:.2f}-{max_sync_time:.2f}ms)"
        )
        print(
            f"  异步: {avg_async_time:.2f}ms (范围: {min_async_time:.2f}-{max_async_time:.2f}ms)"
        )

        print(f"\n连续读取性能:")
        print(f"  同步FPS: {sync_fps:.1f}")
        print(f"  异步FPS: {async_fps:.1f}")

        if avg_sync_time > 0:
            improvement = (avg_sync_time - avg_async_time) / avg_sync_time * 100
            print(f"\n性能提升: {improvement:.1f}%")

        camera.disconnect()
        print("\n✓ 测试完成")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_video4_performance()
