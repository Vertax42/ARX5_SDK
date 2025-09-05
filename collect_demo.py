#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_demo.py - 数据采集演示脚本
展示如何使用collect_no_ros.py进行数据采集
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

from collect_no_ros import main, parse_arguments


def demo_basic_collection():
    """基本数据采集演示"""
    print("=" * 60)
    print("基本数据采集演示")
    print("=" * 60)

    # 创建基本参数
    args = parse_arguments()

    # 修改默认参数
    args.datasets = os.path.join(ROOT, "demo_datasets")
    args.max_timesteps = 100  # 较短的演示
    args.frame_rate = 10  # 较低的帧率
    args.camera_names = ["camera_1", "camera_2"]  # 只使用两个相机
    args.key_collect = True  # 使用键盘触发
    args.task = "demo_task"

    print(f"数据集目录: {args.datasets}")
    print(f"最大时间步: {args.max_timesteps}")
    print(f"采集帧率: {args.frame_rate}")
    print(f"使用相机: {args.camera_names}")
    print(f"任务名称: {args.task}")

    # 运行数据采集
    main(args)


def demo_advanced_collection():
    """高级数据采集演示"""
    print("=" * 60)
    print("高级数据采集演示")
    print("=" * 60)

    # 创建高级参数
    args = parse_arguments()

    # 修改参数
    args.datasets = os.path.join(ROOT, "advanced_datasets")
    args.max_timesteps = 500
    args.frame_rate = 30
    args.camera_names = ["camera_1", "camera_2", "camera_3"]
    args.use_depth_image = True  # 使用深度图像
    args.key_collect = False  # 不使用键盘触发
    args.task = "advanced_task"
    args.episode_idx = 0  # 从episode 0开始

    print(f"数据集目录: {args.datasets}")
    print(f"最大时间步: {args.max_timesteps}")
    print(f"采集帧率: {args.frame_rate}")
    print(f"使用相机: {args.camera_names}")
    print(f"使用深度图像: {args.use_depth_image}")
    print(f"任务名称: {args.task}")

    # 运行数据采集
    main(args)


def demo_custom_config():
    """使用自定义配置文件的演示"""
    print("=" * 60)
    print("自定义配置演示")
    print("=" * 60)

    # 创建参数
    args = parse_arguments()

    # 使用自定义配置文件
    config_path = os.path.join(ROOT, "data", "config.yaml")
    if os.path.exists(config_path):
        args.config = config_path
        print(f"使用配置文件: {config_path}")
    else:
        print(f"配置文件不存在: {config_path}")
        return

    # 修改其他参数
    args.datasets = os.path.join(ROOT, "custom_datasets")
    args.max_timesteps = 200
    args.frame_rate = 20
    args.task = "custom_task"

    print(f"数据集目录: {args.datasets}")
    print(f"最大时间步: {args.max_timesteps}")
    print(f"采集帧率: {args.frame_rate}")
    print(f"任务名称: {args.task}")

    # 运行数据采集
    main(args)


def show_help():
    """显示帮助信息"""
    print("=" * 60)
    print("AC_one机器人数据采集演示")
    print("=" * 60)
    print()
    print("可用的演示模式:")
    print("1. 基本数据采集 - 使用默认参数进行简单数据采集")
    print("2. 高级数据采集 - 使用更多功能和参数")
    print("3. 自定义配置 - 使用YAML配置文件")
    print("4. 帮助信息 - 显示此帮助")
    print()
    print("使用方法:")
    print("python collect_demo.py [模式]")
    print()
    print("示例:")
    print("python collect_demo.py 1    # 运行基本数据采集")
    print("python collect_demo.py 2    # 运行高级数据采集")
    print("python collect_demo.py 3    # 运行自定义配置")
    print("python collect_demo.py help # 显示帮助")


def main_demo():
    """主演示函数"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "1" or mode == "basic":
            demo_basic_collection()
        elif mode == "2" or mode == "advanced":
            demo_advanced_collection()
        elif mode == "3" or mode == "custom":
            demo_custom_config()
        elif mode == "help" or mode == "h":
            show_help()
        else:
            print(f"未知模式: {mode}")
            show_help()
    else:
        show_help()


if __name__ == "__main__":
    main_demo()
