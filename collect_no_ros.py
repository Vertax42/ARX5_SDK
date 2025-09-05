#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_no_ros.py - 基于AC_one机器人的数据采集脚本
不依赖ROS，直接使用AC_one机器人和OpenCV相机系统
"""

import os
import sys
import time
import h5py
import argparse
import cv2
import yaml
import threading
import pyttsx3
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from collections import deque

# 添加当前目录到Python路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

# 导入AC_one机器人系统
from AC_one import AC_one, RobotConfig, create_default_config

np.set_printoptions(linewidth=200)

# 语音引擎初始化
voice_engine = pyttsx3.init()
voice_engine.setProperty("voice", "en")
voice_engine.setProperty("rate", 120)  # 设置语速
voice_lock = threading.Lock()


class Rate:
    """频率控制类，替代ROS的Rate"""

    def __init__(self, hz):
        self.period = 1.0 / hz
        self.last_time = time.time()

    def sleep(self):
        now = time.time()
        elapsed = now - self.last_time
        sleep_time = self.period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.time()


class DataCollector:
    """数据采集器类"""

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.robot = None
        self.collecting = False
        self.episode_data = []
        self.init_pos = None

        # 数据队列
        self.image_queues = {}
        self.arm_state_queues = {
            "left_arm": deque(maxlen=100),
            "right_arm": deque(maxlen=100),
        }

        # 线程控制
        self.data_thread = None
        self.stop_event = threading.Event()

    def initialize_robot(self):
        """初始化机器人系统"""
        print("正在初始化机器人系统...")

        # 创建机器人配置
        robot_config = create_default_config()

        # 根据参数调整配置
        if hasattr(self.args, "camera_names"):
            # 更新相机配置
            camera_configs = []
            for i, cam_name in enumerate(self.args.camera_names):
                camera_configs.append(
                    {
                        "name": cam_name,
                        "position": f"camera_{i+1}",
                        "device": f"/dev/video{i+2}",  # 假设相机设备路径
                    }
                )
            robot_config.camera_configs = camera_configs

        # 创建机器人实例
        self.robot = AC_one(robot_config)

        # 启动相机流
        self.robot.start_camera_streams()

        print("✓ 机器人系统初始化完成")

    def start_data_collection_thread(self):
        """启动数据采集线程"""
        self.data_thread = threading.Thread(
            target=self._data_collection_loop, daemon=True
        )
        self.data_thread.start()

    def _data_collection_loop(self):
        """数据采集循环"""
        rate = Rate(self.args.frame_rate)

        while not self.stop_event.is_set():
            if self.collecting:
                # 获取当前观测数据
                obs_data = self._get_observation()
                if obs_data is not None:
                    self.episode_data.append(obs_data)
                    print(f"采集数据帧: {len(self.episode_data)}")
                else:
                    print("获取观测数据失败，跳过此帧")

            rate.sleep()

    def _get_observation(self) -> Optional[Dict]:
        """获取观测数据"""
        try:
            # 获取相机图像
            camera_frames = self.robot.get_all_camera_frames()
            if not camera_frames:
                return None

            # 获取机械臂状态（如果可用）
            left_arm_state = self.robot.get_left_arm_state()
            right_arm_state = self.robot.get_right_arm_state()

            # 如果机械臂不可用，使用模拟数据
            if left_arm_state is None:
                left_arm_state = {
                    "joint_positions": np.zeros(7),
                    "joint_velocities": np.zeros(7),
                    "gripper_position": 0.0,
                    "timestamp": time.time(),
                }

            if right_arm_state is None:
                right_arm_state = {
                    "joint_positions": np.zeros(7),
                    "joint_velocities": np.zeros(7),
                    "gripper_position": 0.0,
                    "timestamp": time.time(),
                }

            # 构建观测字典
            obs_dict = {
                "images": camera_frames,
                "qpos": np.concatenate(
                    [
                        left_arm_state["joint_positions"],
                        right_arm_state["joint_positions"],
                    ]
                ),
                "qvel": np.concatenate(
                    [
                        left_arm_state["joint_velocities"],
                        right_arm_state["joint_velocities"],
                    ]
                ),
                "eef": np.concatenate(
                    [
                        left_arm_state["joint_positions"][:6],  # 前6个关节作为末端位置
                        [left_arm_state["gripper_position"]],
                        right_arm_state["joint_positions"][:6],
                        [right_arm_state["gripper_position"]],
                    ]
                ),
                "effort": np.zeros(14),  # 暂时设为0，实际应该从机械臂获取
                "robot_base": np.zeros(6),  # 暂时设为0
                "timestamp": time.time(),
            }

            return obs_dict

        except Exception as e:
            print(f"❌ 获取观测数据失败: {e}")
            return None

    def start_episode(self, episode_idx: int):
        """开始一个episode的数据采集"""
        print(f"准备录制episode {episode_idx}")

        # 倒计时
        for i in range(3, -1, -1):
            print(f"\r等待 {i} 秒开始录制...", end="")
            time.sleep(0.3)

        print(f"\n开始录制程序...")

        # 键盘触发录制
        if self.args.key_collect:
            input("按任意键开始录制: ")
        else:
            # 等待初始位置设置
            print("等待设置初始位置...")
            input("按任意键设置初始位置并开始录制: ")

            # 获取初始位置
            obs_data = self._get_observation()
            if obs_data is not None:
                self.init_pos = obs_data["qpos"].copy()
                print(f"初始位置已设置: {self.init_pos}")

        # 开始采集
        self.collecting = True
        self.episode_data = []

        # 语音提示
        self._voice_process(f"开始录制 {episode_idx % 100}")

    def stop_episode(self) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray]]:
        """停止当前episode的数据采集"""
        self.collecting = False

        # 等待数据采集线程完成
        time.sleep(0.1)

        print(f"\n采集完成，共 {len(self.episode_data)} 帧数据")

        # 处理数据
        timesteps = []
        actions = []
        actions_eef = []

        for i, obs_data in enumerate(self.episode_data):
            timesteps.append(obs_data)

            # 动作就是当前的关节位置
            action = obs_data["qpos"].copy()
            action_eef = obs_data["eef"].copy()

            # 夹爪动作处理
            gripper_idx = [6, 13]  # 左右臂夹爪索引
            gripper_close = -2.1

            for idx in gripper_idx:
                if idx < len(action):
                    action[idx] = 0 if action[idx] > gripper_close else action[idx]

            if len(action_eef) > 6:
                action_eef[6] = 0 if action_eef[6] > gripper_close else action_eef[6]
            if len(action_eef) > 13:
                action_eef[13] = 0 if action_eef[13] > gripper_close else action_eef[13]

            actions.append(action)
            actions_eef.append(action_eef)

            # 检查是否应该停止（回到初始位置）
            if i > 100 and self.init_pos is not None:
                if all(
                    abs(val - init) <= 0.1 for val, init in zip(action, self.init_pos)
                ):
                    print(f"检测到回到初始位置，在第 {i} 帧停止")
                    break

        return timesteps, actions, actions_eef

    def _voice_process(self, text: str):
        """语音处理"""
        with voice_lock:
            voice_engine.say(text)
            voice_engine.runAndWait()
            print(text)

    def cleanup(self):
        """清理资源"""
        self.stop_event.set()
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)

        if self.robot:
            self.robot.stop_system()


def compress_and_pad_images(
    data_dict: Dict, camera_names: List[str], use_depth: bool = False, quality: int = 50
):
    """压缩和填充图像数据"""

    def compress_and_pad(key_prefix: str):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        all_encoded = []

        for cam in camera_names:
            key = f"/observations/{key_prefix}/{cam}"
            encoded_list = []
            for img in data_dict[key]:
                _, enc = cv2.imencode(".jpg", img, encode_param)
                encoded_list.append(enc)
                all_encoded.append(len(enc))
            data_dict[key] = encoded_list

        padded_size = max(all_encoded) if all_encoded else 0

        for cam in camera_names:
            key = f"/observations/{key_prefix}/{cam}"
            padded = [
                np.pad(enc, (0, padded_size - len(enc)), constant_values=0)
                for enc in data_dict[key]
            ]
            data_dict[key] = padded

        return padded_size

    # RGB图像
    padded_size = compress_and_pad("images")

    # 深度图像（如果使用）
    padded_size_depth = compress_and_pad("images_depth") if use_depth else 0

    return padded_size, padded_size_depth


def create_and_write_hdf5(
    args,
    data_dict: Dict,
    dataset_path: str,
    data_size: int,
    padded_size: int,
    padded_size_depth: int,
):
    """创建并写入HDF5文件"""
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["task"] = str(args.task)

        obs_dict = root.create_group("observations")
        image = obs_dict.create_group("images")
        if args.use_depth_image:
            depth = obs_dict.create_group("images_depth")

        for cam_name in args.camera_names:
            img_shape = (data_size, padded_size)
            img_chunk = (1, padded_size)
            if args.use_depth_image:
                depth_shape = (data_size, padded_size_depth)
                depth_chunk = (1, padded_size_depth)

            image.create_dataset(cam_name, img_shape, "uint8", chunks=img_chunk)
            if args.use_depth_image:
                depth.create_dataset(cam_name, depth_shape, "uint8", chunks=depth_chunk)

        # 创建观测和动作数据集
        state_dim = 14  # 7+7 关节
        eef_dim = 14  # 6+1+6+1 末端执行器
        obs_specs = {
            "qpos": state_dim,
            "eef": eef_dim,
            "qvel": state_dim,
            "effort": state_dim,
            "robot_base": 6,
        }
        act_specs = {
            "action": state_dim,
            "action_eef": eef_dim,
        }

        for name, dim in obs_specs.items():
            obs_dict.create_dataset(name, (data_size, dim))
        for name, dim in act_specs.items():
            root.create_dataset(name, (data_size, dim))

        for name, arr in data_dict.items():
            root[name][...] = arr


def save_data(
    args,
    timesteps: List[Dict],
    actions: List[np.ndarray],
    actions_eef: List[np.ndarray],
    dataset_path: str,
):
    """保存数据到HDF5文件"""
    data_size = len(actions)

    # 数据字典
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/observations/eef": [],
        "/observations/robot_base": [],
        "/action": [],
        "/action_eef": [],
    }

    # 初始化相机字典
    for cam_name in args.camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []
        if args.use_depth_image:
            data_dict[f"/observations/images_depth/{cam_name}"] = []

    # 遍历并收集数据
    for i, (ts, action, action_eef) in enumerate(zip(timesteps, actions, actions_eef)):
        # 填充数据
        data_dict["/observations/qpos"].append(ts["qpos"])
        data_dict["/observations/qvel"].append(ts["qvel"])
        data_dict["/observations/eef"].append(ts["eef"])
        data_dict["/observations/effort"].append(ts["effort"])
        data_dict["/observations/robot_base"].append(ts["robot_base"])
        data_dict["/action"].append(action)
        data_dict["/action_eef"].append(action_eef)

        # 相机数据
        for cam_name in args.camera_names:
            if cam_name in ts["images"]:
                data_dict[f"/observations/images/{cam_name}"].append(
                    ts["images"][cam_name]
                )
            if args.use_depth_image and f"{cam_name}_depth" in ts.get(
                "images_depth", {}
            ):
                data_dict[f"/observations/images_depth/{cam_name}"].append(
                    ts["images_depth"][f"{cam_name}_depth"]
                )

    # 压缩图像数据
    padded_size, padded_size_depth = compress_and_pad_images(
        data_dict, args.camera_names, args.use_depth_image
    )

    # 写入HDF5文件
    t0 = time.time()
    create_and_write_hdf5(
        args, data_dict, dataset_path, data_size, padded_size, padded_size_depth
    )

    print(f"\033[32m\n保存完成，耗时 {time.time() - t0:.1f}s: {dataset_path}\033[0m\n")


def load_yaml(yaml_file: str) -> Optional[Dict]:
    """加载YAML配置文件"""
    try:
        with open(yaml_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {yaml_file}")
        return None
    except yaml.YAMLError as e:
        print(f"错误: YAML文件解析失败 - {e}")
        return None


def main(args):
    """主函数"""
    print("=" * 60)
    print("AC_one机器人数据采集系统")
    print("=" * 60)

    # 加载配置文件
    config = load_yaml(args.config) if args.config else {}

    # 创建数据采集器
    collector = DataCollector(args, config)

    try:
        # 初始化机器人
        collector.initialize_robot()

        # 启动数据采集线程
        collector.start_data_collection_thread()

        # 数据集目录
        datasets_dir = (
            args.datasets
            if os.path.isabs(args.datasets)
            else os.path.join(ROOT, args.datasets)
        )
        os.makedirs(datasets_dir, exist_ok=True)

        # 查找最大episode序号
        max_episode = -1
        if os.path.exists(datasets_dir):
            for filename in os.listdir(datasets_dir):
                if filename.startswith("episode_") and filename.endswith(".hdf5"):
                    try:
                        episode_num = int(filename.split("_")[1].split(".")[0])
                        max_episode = max(max_episode, episode_num)
                    except ValueError:
                        continue

        # 确定起始episode
        current_episode = max_episode + 1 if max_episode >= 0 else 0
        num_episodes = 1000 if args.episode_idx == -1 else 1

        if args.episode_idx != -1:
            current_episode = args.episode_idx

        # 开始数据采集循环
        episode_num = 0
        while episode_num < num_episodes:
            print(f"\nEpisode {episode_num + 1}/{num_episodes}")

            # 开始episode
            collector.start_episode(current_episode)

            # 等待用户停止或达到最大时间步
            print(f"开始录制episode {current_episode}")
            print("按Ctrl+C停止当前episode")

            try:
                # 等待用户中断或达到最大时间步
                start_time = time.time()
                while True:
                    if len(collector.episode_data) >= args.max_timesteps:
                        print(f"达到最大时间步 {args.max_timesteps}，停止录制")
                        break

                    time.sleep(0.1)

            except KeyboardInterrupt:
                print("\n用户中断，停止当前episode")

            # 停止episode并保存数据
            timesteps, actions, actions_eef = collector.stop_episode()

            if len(actions) > 0:
                dataset_path = os.path.join(datasets_dir, f"episode_{current_episode}")

                # 在后台线程中保存数据
                save_thread = threading.Thread(
                    target=save_data,
                    args=(args, timesteps, actions, actions_eef, dataset_path),
                    daemon=True,
                )
                save_thread.start()

                print(f"Episode {current_episode} 数据保存中...")
            else:
                print("❌ 没有采集到数据，跳过保存")

            episode_num += 1
            current_episode += 1

            # 询问是否继续
            if episode_num < num_episodes:
                try:
                    continue_collect = input(
                        "按Enter继续下一个episode，或输入'q'退出: "
                    ).strip()
                    if continue_collect.lower() == "q":
                        break
                except KeyboardInterrupt:
                    break

    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        collector.cleanup()
        print("数据采集系统已停止")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AC_one机器人数据采集系统")

    # 数据集配置
    parser.add_argument(
        "--datasets",
        type=str,
        default=os.path.join(ROOT, "datasets"),
        help="数据集目录",
    )
    parser.add_argument(
        "--episode_idx", type=int, default=-1, help="episode索引，-1表示自动递增"
    )
    parser.add_argument("--max_timesteps", type=int, default=800, help="最大时间步数")
    parser.add_argument("--frame_rate", type=int, default=30, help="采集帧率")

    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT, "data", "config.yaml"),
        help="配置文件路径",
    )

    # 图像处理选项
    parser.add_argument(
        "--camera_names",
        nargs="+",
        type=str,
        choices=["camera_1", "camera_2", "camera_3"],
        default=["camera_1", "camera_2", "camera_3"],
        help="相机名称列表",
    )
    parser.add_argument("--use_depth_image", action="store_true", help="使用深度图像")

    # 数据采集选项
    parser.add_argument("--key_collect", action="store_true", help="使用键盘触发采集")
    parser.add_argument("--task", type=str, default="", help="任务名称")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
