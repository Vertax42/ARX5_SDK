#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_collect_test.py - 简单的数据采集测试
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


def simple_test():
    """简单测试数据采集功能"""
    print("=" * 60)
    print("简单数据采集测试")
    print("=" * 60)

    # 创建测试参数
    args = parse_arguments()
    args.datasets = os.path.join(ROOT, "simple_test_datasets")
    args.max_timesteps = 5  # 只采集5帧
    args.frame_rate = 2  # 低帧率
    args.camera_names = ["camera_1"]  # 只使用一个相机
    args.key_collect = True
    args.task = "simple_test"

    print(f"测试参数:")
    print(f"  数据集目录: {args.datasets}")
    print(f"  最大时间步: {args.max_timesteps}")
    print(f"  采集帧率: {args.frame_rate}")
    print(f"  使用相机: {args.camera_names}")

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
        time.sleep(5)  # 等待5秒

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

        # 测试数据保存
        if len(timesteps) > 0:
            print("\n测试数据保存...")
            from collect_no_ros import save_data

            dataset_path = os.path.join(args.datasets, "test_episode_0")
            os.makedirs(args.datasets, exist_ok=True)

            save_data(args, timesteps, actions, actions_eef, dataset_path)
            print(f"✓ 数据已保存到: {dataset_path}.hdf5")

        print("\n✓ 简单测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        collector.cleanup()
        print("资源清理完成")


if __name__ == "__main__":
    simple_test()
