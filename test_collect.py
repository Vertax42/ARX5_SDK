#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_collect.py - 测试数据采集功能
"""

import os
import sys
import time
from pathlib import Path

# 添加当前目录到Python路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

from collect_no_ros import DataCollector, parse_arguments


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试数据采集基本功能")
    print("=" * 60)

    # 创建测试参数
    args = parse_arguments()
    args.datasets = os.path.join(ROOT, "test_datasets")
    args.max_timesteps = 10  # 只采集10帧用于测试
    args.frame_rate = 5  # 低帧率
    args.camera_names = ["camera_1"]  # 只使用一个相机
    args.key_collect = True
    args.task = "test_task"

    print(f"测试参数:")
    print(f"  数据集目录: {args.datasets}")
    print(f"  最大时间步: {args.max_timesteps}")
    print(f"  采集帧率: {args.frame_rate}")
    print(f"  使用相机: {args.camera_names}")
    print(f"  任务名称: {args.task}")

    # 创建数据采集器
    collector = DataCollector(args, {})

    try:
        # 初始化机器人
        print("\n正在初始化机器人系统...")
        collector.initialize_robot()

        # 启动数据采集线程
        print("启动数据采集线程...")
        collector.start_data_collection_thread()

        # 开始一个测试episode
        print("\n开始测试episode...")
        collector.start_episode(0)

        # 等待采集一些数据
        print("等待采集数据...")
        time.sleep(3)  # 等待3秒

        # 停止episode
        print("停止episode...")
        timesteps, actions, actions_eef = collector.stop_episode()

        print(f"\n采集结果:")
        print(f"  时间步数: {len(timesteps)}")
        print(f"  动作数: {len(actions)}")
        print(f"  末端执行器动作数: {len(actions_eef)}")

        if len(timesteps) > 0:
            print(f"  第一个观测的关节位置: {timesteps[0]['qpos']}")
            print(f"  相机图像数量: {len(timesteps[0]['images'])}")

            # 显示相机信息
            for cam_name, img in timesteps[0]["images"].items():
                if img is not None:
                    print(f"  相机 {cam_name}: 图像尺寸 {img.shape}")
                else:
                    print(f"  相机 {cam_name}: 无图像数据")

        print("\n✓ 基本功能测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        collector.cleanup()
        print("资源清理完成")


def test_data_processing():
    """测试数据处理功能"""
    print("\n" + "=" * 60)
    print("测试数据处理功能")
    print("=" * 60)

    import numpy as np
    from collect_no_ros import compress_and_pad_images, create_and_write_hdf5

    # 创建测试数据
    test_data = {
        "/observations/images/camera_1": [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
        ],
        "/observations/images/camera_2": [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
        ],
    }

    print("测试图像压缩和填充...")
    padded_size, _ = compress_and_pad_images(test_data, ["camera_1", "camera_2"])
    print(f"填充后图像大小: {padded_size}")

    # 创建测试HDF5文件
    print("测试HDF5文件创建...")
    test_file = os.path.join(ROOT, "test_data.hdf5")

    # 创建模拟参数
    class TestArgs:
        def __init__(self):
            self.task = "test"
            self.use_depth_image = False
            self.camera_names = ["camera_1", "camera_2"]

    args = TestArgs()

    try:
        create_and_write_hdf5(args, test_data, test_file, 5, padded_size, 0)
        print(f"✓ HDF5文件创建成功: {test_file}")

        # 验证文件
        import h5py

        with h5py.File(test_file + ".hdf5", "r") as f:
            print(f"文件内容: {list(f.keys())}")
            print(f"观测数据: {list(f['observations'].keys())}")
            print(f"图像数据形状: {f['observations/images/camera_1'].shape}")

        # 清理测试文件
        os.remove(test_file + ".hdf5")
        print("✓ 测试文件已清理")

    except Exception as e:
        print(f"❌ HDF5测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    """主测试函数"""
    print("AC_one数据采集系统测试")
    print("=" * 60)

    # 测试基本功能
    test_basic_functionality()

    # 测试数据处理
    test_data_processing()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
