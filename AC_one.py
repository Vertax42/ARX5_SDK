#!/usr/bin/env python3
"""
AC_one.py - 双臂机器人系统
包含两个X5机械臂和3个Intel RealSense D405相机
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
import os

# 添加cameras目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "cameras"))

# 添加arx5-sdk到Python路径
arx5_sdk_path = Path(__file__).parent / "arx5-sdk" / "python"
sys.path.insert(0, str(arx5_sdk_path))
os.chdir(str(arx5_sdk_path))

try:
    import arx5_interface as arx5

    ARX5_AVAILABLE = True
except ImportError:
    print("⚠ ARX5 SDK not available, mechanical arm control will be simulated")
    ARX5_AVAILABLE = False


@dataclass
class RobotConfig:
    """机器人配置类"""

    # 机械臂配置
    left_arm_model: str = "X5"  # ARX arm model: X5 or L5
    left_arm_interface: str = "can0"  # CAN bus interface
    right_arm_model: str = "X5"  # ARX arm model: X5 or L5
    right_arm_interface: str = "can1"  # CAN bus interface

    # 相机配置 (OpenCV相机)
    camera_configs: List[Dict] = None

    # 系统配置
    fps: int = 30
    resolution: Tuple[int, int] = (640, 480)
    use_multithreading: bool = True  # 使用多线程控制


class AC_one:
    """AC_one双臂机器人类

    功能:
    - 控制左右两个X5机械臂
    - 管理3个OpenCV相机 (支持USB相机)
    - 提供统一的机器人控制接口
    """

    def __init__(self, config: RobotConfig):
        """初始化AC_one机器人

        Args:
            config: 机器人配置
        """
        self.config = config
        self.is_initialized = False

        # 机械臂实例
        self.left_arm = None
        self.right_arm = None

        # 相机实例
        self.cameras = {}
        self.camera_threads = {}
        self.camera_frames = {}
        self.camera_locks = {}

        # 系统状态
        self.system_running = False
        self.arm_connected = False
        self.cameras_connected = False

        print("AC_one机器人系统初始化...")
        self._initialize_arms()
        self._initialize_cameras()

    def _initialize_arms(self):
        """初始化机械臂"""
        print("正在初始化机械臂...")

        if not ARX5_AVAILABLE:
            print("⚠ ARX5 SDK不可用，使用模拟模式")
            self.arm_connected = True
            return

        try:
            print(
                f"  - 初始化左臂 ({self.config.left_arm_model} on {self.config.left_arm_interface})..."
            )

            # 创建左臂控制器
            self.left_arm = arx5.Arx5JointController(
                self.config.left_arm_model, self.config.left_arm_interface
            )

            # 设置日志级别
            self.left_arm.set_log_level(arx5.LogLevel.INFO)

            # 获取配置
            self.robot_config = self.left_arm.get_robot_config()
            self.controller_config = self.left_arm.get_controller_config()

            # 配置多线程
            if self.config.use_multithreading:
                self.controller_config.background_send_recv = True
            else:
                self.controller_config.background_send_recv = False

            print(
                f"  - 初始化右臂 ({self.config.right_arm_model} on {self.config.right_arm_interface})..."
            )

            # 创建右臂控制器
            self.right_arm = arx5.Arx5JointController(
                self.config.right_arm_model, self.config.right_arm_interface
            )

            # 设置日志级别
            self.right_arm.set_log_level(arx5.LogLevel.INFO)

            # 重置到初始位置
            print("  - 重置机械臂到初始位置...")
            self.left_arm.reset_to_home()
            self.right_arm.reset_to_home()

            self.arm_connected = True
            print("✓ 机械臂初始化完成")

        except Exception as e:
            print(f"❌ 机械臂初始化失败: {e}")
            self.arm_connected = False
            # 清理已创建的控制器
            if hasattr(self, "left_arm"):
                del self.left_arm
            if hasattr(self, "right_arm"):
                del self.right_arm

    def _initialize_cameras(self):
        """初始化相机系统"""
        print("正在初始化相机系统...")

        try:
            from cameras.opencv import OpenCVCamera
            from cameras.opencv.configuration_opencv import (
                OpenCVCameraConfig,
                ColorMode,
                Cv2Rotation,
            )

            # 查找可用的OpenCV相机
            available_cameras = OpenCVCamera.find_cameras()
            print(f"找到 {len(available_cameras)} 个OpenCV相机")

            # 初始化相机
            for i, camera_info in enumerate(available_cameras[:3]):  # 最多使用3个相机
                camera_name = f"camera_{i+1}"
                camera_id = camera_info["id"]

                print(f"  - 初始化相机 {camera_name} (ID: {camera_id})")

                # 创建相机配置
                config = OpenCVCameraConfig(
                    index_or_path=camera_id,
                    fps=self.config.fps,
                    width=self.config.resolution[0],
                    height=self.config.resolution[1],
                    color_mode=ColorMode.RGB,
                    rotation=Cv2Rotation.NO_ROTATION,
                )

                # 创建相机实例
                camera = OpenCVCamera(config)
                camera.connect(warmup=True)

                # 存储相机实例
                self.cameras[camera_name] = camera
                self.camera_frames[camera_name] = {
                    "color": None,
                    "timestamp": 0,
                }
                self.camera_locks[camera_name] = threading.Lock()

                print(f"    ✓ 相机 {camera_name} 连接成功")

            self.cameras_connected = True
            print("✓ 相机系统初始化完成")

        except Exception as e:
            print(f"❌ 相机系统初始化失败: {e}")
            self.cameras_connected = False

    def start_camera_streams(self):
        """启动相机流"""
        if not self.cameras_connected:
            print("❌ 相机未连接，无法启动流")
            return False

        print("启动相机流...")

        for camera_name, camera in self.cameras.items():
            # 为每个相机创建独立的线程
            thread = threading.Thread(
                target=self._camera_stream_loop,
                args=(camera_name, camera),
                name=f"Camera_{camera_name}_Thread",
            )
            thread.daemon = True
            thread.start()

            self.camera_threads[camera_name] = thread
            print(f"  ✓ 相机 {camera_name} 流已启动")

        self.system_running = True
        print("✓ 所有相机流已启动")
        return True

    def _camera_stream_loop(self, camera_name: str, camera):
        """相机流循环线程"""
        while self.system_running:
            try:
                # 读取彩色帧
                color_frame = camera.read()

                # 更新帧数据
                with self.camera_locks[camera_name]:
                    self.camera_frames[camera_name]["color"] = color_frame
                    self.camera_frames[camera_name]["timestamp"] = time.time()

                time.sleep(1.0 / self.config.fps)  # 控制帧率

            except Exception as e:
                print(f"相机 {camera_name} 流错误: {e}")
                time.sleep(0.1)

    def get_camera_frame(
        self, camera_name: str, frame_type: str = "color"
    ) -> Optional[np.ndarray]:
        """获取相机帧

        Args:
            camera_name: 相机名称 (camera_1, camera_2, camera_3)
            frame_type: 帧类型 (OpenCV相机只支持'color')

        Returns:
            相机帧数组，如果不可用则返回None
        """
        if camera_name not in self.camera_frames:
            print(f"❌ 相机 {camera_name} 不存在")
            return None

        if frame_type != "color":
            print(f"⚠ OpenCV相机只支持彩色帧，请求的'{frame_type}'将被忽略")
            frame_type = "color"

        with self.camera_locks[camera_name]:
            return self.camera_frames[camera_name][frame_type]

    def get_all_camera_frames(self, frame_type: str = "color") -> Dict[str, np.ndarray]:
        """获取所有相机的帧

        Args:
            frame_type: 帧类型 (OpenCV相机只支持'color')

        Returns:
            包含所有相机帧的字典
        """
        frames = {}
        for camera_name in self.cameras.keys():
            frame = self.get_camera_frame(camera_name, frame_type)
            if frame is not None:
                frames[camera_name] = frame
        return frames

    def move_left_arm_joint(
        self, joint_positions: List[float], gripper_pos: float = 0.0
    ):
        """移动左臂关节位置

        Args:
            joint_positions: 关节位置 [j1, j2, j3, j4, j5, j6]
            gripper_pos: 夹爪位置 (0.0-1.0)

        Returns:
            bool: 是否成功
        """
        if not self.arm_connected or not ARX5_AVAILABLE:
            print("❌ 机械臂未连接或ARX5 SDK不可用")
            return False

        try:
            # 创建关节状态命令
            cmd = arx5.JointState(self.robot_config.joint_dof)
            cmd.pos()[:] = np.array(joint_positions)
            cmd.gripper_pos = gripper_pos

            # 发送命令
            self.left_arm.set_joint_cmd(cmd)

            if not self.config.use_multithreading:
                self.left_arm.send_recv_once()

            return True

        except Exception as e:
            print(f"❌ 左臂移动失败: {e}")
            return False

    def move_right_arm_joint(
        self, joint_positions: List[float], gripper_pos: float = 0.0
    ):
        """移动右臂关节位置

        Args:
            joint_positions: 关节位置 [j1, j2, j3, j4, j5, j6]
            gripper_pos: 夹爪位置 (0.0-1.0)

        Returns:
            bool: 是否成功
        """
        if not self.arm_connected or not ARX5_AVAILABLE:
            print("❌ 机械臂未连接或ARX5 SDK不可用")
            return False

        try:
            # 创建关节状态命令
            cmd = arx5.JointState(self.robot_config.joint_dof)
            cmd.pos()[:] = np.array(joint_positions)
            cmd.gripper_pos = gripper_pos

            # 发送命令
            self.right_arm.set_joint_cmd(cmd)

            if not self.config.use_multithreading:
                self.right_arm.send_recv_once()

            return True

        except Exception as e:
            print(f"❌ 右臂移动失败: {e}")
            return False

    def move_both_arms_joint(
        self,
        left_joint_positions: List[float],
        right_joint_positions: List[float],
        left_gripper_pos: float = 0.0,
        right_gripper_pos: float = 0.0,
    ):
        """同时移动两个机械臂关节位置

        Args:
            left_joint_positions: 左臂关节位置
            right_joint_positions: 右臂关节位置
            left_gripper_pos: 左臂夹爪位置
            right_gripper_pos: 右臂夹爪位置

        Returns:
            bool: 是否成功
        """
        if not self.arm_connected or not ARX5_AVAILABLE:
            print("❌ 机械臂未连接或ARX5 SDK不可用")
            return False

        try:
            # 创建左臂命令
            left_cmd = arx5.JointState(self.robot_config.joint_dof)
            left_cmd.pos()[:] = np.array(left_joint_positions)
            left_cmd.gripper_pos = left_gripper_pos

            # 创建右臂命令
            right_cmd = arx5.JointState(self.robot_config.joint_dof)
            right_cmd.pos()[:] = np.array(right_joint_positions)
            right_cmd.gripper_pos = right_gripper_pos

            # 发送命令
            self.left_arm.set_joint_cmd(left_cmd)
            self.right_arm.set_joint_cmd(right_cmd)

            if not self.config.use_multithreading:
                self.left_arm.send_recv_once()
                self.right_arm.send_recv_once()

            return True

        except Exception as e:
            print(f"❌ 双臂移动失败: {e}")
            return False

    def get_left_arm_state(self) -> Optional[Dict]:
        """获取左臂状态

        Returns:
            左臂状态字典，如果不可用则返回None
        """
        if not self.arm_connected or not ARX5_AVAILABLE:
            return None

        try:
            joint_state = self.left_arm.get_joint_state()
            return {
                "joint_positions": joint_state.pos().copy(),
                "joint_velocities": joint_state.vel().copy(),
                "gripper_position": joint_state.gripper_pos,
                "timestamp": joint_state.timestamp,
            }
        except Exception as e:
            print(f"❌ 获取左臂状态失败: {e}")
            return None

    def get_right_arm_state(self) -> Optional[Dict]:
        """获取右臂状态

        Returns:
            右臂状态字典，如果不可用则返回None
        """
        if not self.arm_connected or not ARX5_AVAILABLE:
            return None

        try:
            joint_state = self.right_arm.get_joint_state()
            return {
                "joint_positions": joint_state.pos().copy(),
                "joint_velocities": joint_state.vel().copy(),
                "gripper_position": joint_state.gripper_pos,
                "timestamp": joint_state.timestamp,
            }
        except Exception as e:
            print(f"❌ 获取右臂状态失败: {e}")
            return None

    def reset_arms_to_home(self):
        """重置双臂到初始位置

        Returns:
            bool: 是否成功
        """
        if not self.arm_connected or not ARX5_AVAILABLE:
            print("❌ 机械臂未连接或ARX5 SDK不可用")
            return False

        try:
            print("重置双臂到初始位置...")
            self.left_arm.reset_to_home()
            self.right_arm.reset_to_home()
            print("✓ 双臂已重置到初始位置")
            return True
        except Exception as e:
            print(f"❌ 重置双臂失败: {e}")
            return False

    def get_arm_status(self) -> Dict[str, Dict]:
        """获取机械臂状态

        Returns:
            包含左右臂状态的字典
        """
        status = {
            "left_arm": {
                "connected": self.arm_connected,
                "position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 实际应该从机械臂获取
                "status": "ready" if self.arm_connected else "disconnected",
            },
            "right_arm": {
                "connected": self.arm_connected,
                "position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 实际应该从机械臂获取
                "status": "ready" if self.arm_connected else "disconnected",
            },
        }
        return status

    def get_camera_status(self) -> Dict[str, Dict]:
        """获取相机状态

        Returns:
            包含所有相机状态的字典
        """
        status = {}
        for camera_name, camera in self.cameras.items():
            status[camera_name] = {
                "connected": camera.is_connected,
                "resolution": f"{camera.width}x{camera.height}",
                "fps": camera.fps,
                "last_frame_time": self.camera_frames[camera_name]["timestamp"],
            }
        return status

    def get_system_status(self) -> Dict:
        """获取系统状态

        Returns:
            包含整个系统状态的字典
        """
        return {
            "system_running": self.system_running,
            "arm_connected": self.arm_connected,
            "cameras_connected": self.cameras_connected,
            "camera_count": len(self.cameras),
            "arms": self.get_arm_status(),
            "cameras": self.get_camera_status(),
        }

    def stop_system(self):
        """停止系统"""
        print("正在停止AC_one系统...")

        # 停止相机流
        self.system_running = False

        # 等待相机线程结束
        for camera_name, thread in self.camera_threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
                print(f"  ✓ 相机 {camera_name} 线程已停止")

        # 断开相机连接
        for camera_name, camera in self.cameras.items():
            try:
                camera.disconnect()
                print(f"  ✓ 相机 {camera_name} 已断开")
            except Exception as e:
                print(f"❌ 相机 {camera_name} 断开失败: {e}")
                raise e

        # 断开机械臂连接
        if self.arm_connected:
            print("  ✓ 机械臂已断开")

        print("✓ AC_one系统已停止")

    def __del__(self):
        """析构函数"""
        try:
            self.stop_system()
        except:
            pass  # 忽略析构函数中的错误


def create_default_config() -> RobotConfig:
    """创建默认配置"""
    return RobotConfig(
        left_arm_model="X5",
        left_arm_interface="can0",
        right_arm_model="X5",
        right_arm_interface="can1",
        camera_configs=[
            {"name": "camera_1", "position": "front", "device": "/dev/video2"},
            {"name": "camera_2", "position": "left", "device": "/dev/video4"},
            {"name": "camera_3", "position": "right", "device": "/dev/video6"},
        ],
        fps=30,
        resolution=(640, 480),
        use_multithreading=True,
    )


def main():
    """主函数 - 演示AC_one机器人使用"""
    print("=" * 60)
    print("AC_one双臂机器人系统演示")
    print("=" * 60)

    # 创建配置
    config = create_default_config()

    # 创建机器人实例
    robot = AC_one(config)

    # 检查系统状态
    status = robot.get_system_status()
    print("\n系统状态:")
    print(f"  机械臂连接: {'✓' if status['arm_connected'] else '❌'}")
    print(f"  相机连接: {'✓' if status['cameras_connected'] else '❌'}")
    print(f"  相机数量: {status['camera_count']}")

    if not status["cameras_connected"]:
        print("❌ 相机未连接，退出演示")
        return

    # 启动相机流
    if robot.start_camera_streams():
        print("\n相机流已启动，按Ctrl+C停止...")

        try:
            frame_count = 0
            while True:
                # 获取所有相机的彩色帧
                color_frames = robot.get_all_camera_frames("color")

                if color_frames:
                    # 显示第一个相机的帧
                    first_camera = list(color_frames.keys())[0]
                    frame = color_frames[first_camera]

                    # 转换颜色格式用于显示
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 添加信息
                    info_text = (
                        f"AC_one Robot - Camera: {first_camera} - Frame: {frame_count}"
                    )
                    cv2.putText(
                        frame_bgr,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # 显示图像
                    cv2.imshow("AC_one Robot System", frame_bgr)

                    frame_count += 1

                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        # 保存当前帧
                        filename = f"ac_one_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame_bgr)
                        print(f"✓ 保存帧到: {filename}")

                time.sleep(0.033)  # 约30FPS

        except KeyboardInterrupt:
            print("\n用户中断...")

        finally:
            # 清理
            cv2.destroyAllWindows()
            robot.stop_system()

    print("AC_one演示结束")


if __name__ == "__main__":
    main()
