#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_arx5_sdk_only.py - 创建纯ARX5 SDK包
只包含arx5-sdk下的原生代码，不包含相机系统和AC_one机器人
"""

import os
import shutil
from pathlib import Path


def create_arx5_sdk_only():
    """创建纯ARX5 SDK包"""

    # 源目录和目标目录
    source_dir = Path(__file__).parent
    package_dir = source_dir / "arx5_sdk_only"

    print("创建纯ARX5 SDK包...")
    print(f"源目录: {source_dir}")
    print(f"包目录: {package_dir}")

    # 创建包目录结构
    package_dir.mkdir(exist_ok=True)

    # 1. 复制ARX5 SDK核心文件
    print("\n1. 复制ARX5 SDK核心文件...")
    sdk_source = source_dir / "arx5-sdk"
    sdk_dest = package_dir / "arx5_sdk"

    if sdk_source.exists():
        if sdk_dest.exists():
            shutil.rmtree(sdk_dest)
        shutil.copytree(sdk_source, sdk_dest)
        print(f"✓ ARX5 SDK 复制到: {sdk_dest}")
    else:
        print(f"❌ ARX5 SDK 源目录不存在: {sdk_source}")
        return

    # 2. 创建Python包结构
    print("\n2. 创建Python包结构...")

    # 创建__init__.py
    init_file = package_dir / "__init__.py"
    with open(init_file, "w", encoding="utf-8") as f:
        f.write(
            '''"""
ARX5 SDK Python Package

Pure ARX5 SDK without camera systems or robot controllers.
Only contains the native ARX5 SDK functionality.
"""

import os
import sys
from pathlib import Path

# 添加SDK路径
sdk_path = Path(__file__).parent / "arx5_sdk" / "python"
if str(sdk_path) not in sys.path:
    sys.path.insert(0, str(sdk_path))

try:
    import arx5_interface as arx5
    ARX5_AVAILABLE = True
except ImportError:
    print("⚠ ARX5 SDK not available")
    ARX5_AVAILABLE = False
    arx5 = None

# 导出主要类和函数
__all__ = [
    "arx5",
    "ARX5_AVAILABLE",
]

__version__ = "1.0.0"
'''
        )

    print(f"✓ __init__.py 创建: {init_file}")

    # 3. 创建setup.py
    print("\n3. 创建setup.py...")

    setup_file = package_dir / "setup.py"
    with open(setup_file, "w", encoding="utf-8") as f:
        f.write(
            '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5 SDK Setup
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# 获取当前目录
current_dir = Path(__file__).parent

setup(
    name="arx5-sdk",
    version="1.0.0",
    description="ARX5 Robot SDK - Pure SDK without additional dependencies",
    author="ARX5 Team",
    author_email="contact@arx5.com",
    url="https://github.com/arx5/arx5-sdk",
    packages=find_packages(),
    package_data={
        "": [
            "arx5_sdk/python/*.so",
            "arx5_sdk/python/*.pyi",
            "arx5_sdk/lib/x86_64/*.so",
            "arx5_sdk/lib/aarch64/*.so",
            "arx5_sdk/models/*.urdf",
            "arx5_sdk/models/meshes/*",
        ]
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
'''
        )

    print(f"✓ setup.py 创建: {setup_file}")

    # 4. 创建使用示例
    print("\n4. 创建使用示例...")

    example_file = package_dir / "example_usage.py"
    with open(example_file, "w", encoding="utf-8") as f:
        f.write(
            '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5 SDK使用示例
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import arx5_sdk
    from arx5_sdk import arx5, ARX5_AVAILABLE
except ImportError:
    print("❌ 无法导入arx5_sdk")
    sys.exit(1)

def test_basic_functionality():
    """测试基本功能"""
    print("ARX5 SDK基本功能测试")
    print("=" * 40)
    
    if not ARX5_AVAILABLE:
        print("❌ ARX5 SDK不可用")
        return False
    
    print("✓ ARX5 SDK可用")
    
    # 测试创建控制器
    try:
        print("\\n测试创建左臂控制器...")
        left_arm = arx5.Arx5JointController("X5", "can0")
        print("✓ 左臂控制器创建成功")
        
        # 获取配置
        robot_config = left_arm.get_robot_config()
        controller_config = left_arm.get_controller_config()
        
        print(f"机器人配置:")
        print(f"  关节自由度: {robot_config.joint_dof}")
        print(f"  关节位置范围: {robot_config.joint_pos_min} ~ {robot_config.joint_pos_max}")
        print(f"  关节速度范围: {robot_config.joint_vel_max}")
        print(f"  夹爪宽度: {robot_config.gripper_width}")
        
        print(f"\\n控制器配置:")
        print(f"  控制频率: {1.0/controller_config.controller_dt:.1f} Hz")
        print(f"  默认Kp: {controller_config.default_kp}")
        print(f"  默认Kd: {controller_config.default_kd}")
        
        # 测试关节状态
        print("\\n测试获取关节状态...")
        joint_state = left_arm.get_joint_state()
        print(f"当前关节位置: {joint_state.pos()}")
        print(f"当前关节速度: {joint_state.vel()}")
        print(f"当前夹爪位置: {joint_state.gripper_pos}")
        
        # 测试关节控制
        print("\\n测试关节控制...")
        cmd = arx5.JointState(robot_config.joint_dof)
        cmd.pos()[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        cmd.gripper_pos = 0.5
        
        left_arm.set_joint_cmd(cmd)
        print("✓ 关节命令发送成功")
        
        # 清理
        del left_arm
        print("✓ 控制器清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bimanual_control():
    """测试双臂控制"""
    print("\\n" + "=" * 40)
    print("ARX5 SDK双臂控制测试")
    print("=" * 40)
    
    if not ARX5_AVAILABLE:
        print("❌ ARX5 SDK不可用")
        return False
    
    try:
        print("\\n测试创建双臂控制器...")
        left_arm = arx5.Arx5JointController("X5", "can0")
        right_arm = arx5.Arx5JointController("X5", "can1")
        
        print("✓ 双臂控制器创建成功")
        
        # 获取配置
        robot_config = left_arm.get_robot_config()
        
        # 创建关节命令
        left_cmd = arx5.JointState(robot_config.joint_dof)
        right_cmd = arx5.JointState(robot_config.joint_dof)
        
        # 设置不同的目标位置
        left_cmd.pos()[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        left_cmd.gripper_pos = 0.5
        
        right_cmd.pos()[:] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
        right_cmd.gripper_pos = 0.3
        
        # 发送命令
        left_arm.set_joint_cmd(left_cmd)
        right_arm.set_joint_cmd(right_cmd)
        
        print("✓ 双臂命令发送成功")
        
        # 获取状态
        left_state = left_arm.get_joint_state()
        right_state = right_arm.get_joint_state()
        
        print(f"左臂状态: {left_state.pos()}")
        print(f"右臂状态: {right_state.pos()}")
        
        # 清理
        del left_arm
        del right_arm
        print("✓ 双臂控制器清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 双臂控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("ARX5 SDK测试程序")
    print("=" * 50)
    
    # 测试基本功能
    basic_ok = test_basic_functionality()
    
    # 测试双臂控制
    bimanual_ok = test_bimanual_control()
    
    print("\\n" + "=" * 50)
    print("测试结果:")
    print(f"基本功能: {'✓ 通过' if basic_ok else '❌ 失败'}")
    print(f"双臂控制: {'✓ 通过' if bimanual_ok else '❌ 失败'}")
    
    if basic_ok and bimanual_ok:
        print("\\n🎉 所有测试通过！ARX5 SDK工作正常")
    else:
        print("\\n⚠ 部分测试失败，请检查硬件连接和配置")

if __name__ == "__main__":
    main()
'''
        )

    print(f"✓ 使用示例创建: {example_file}")

    # 5. 创建README
    print("\n5. 创建README...")

    readme_file = package_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(
            """# ARX5 SDK Python Package

纯ARX5 SDK Python包，不包含相机系统和机器人控制器，只提供ARX5机械臂的原生SDK功能。

## 功能特性

- ✅ **ARX5机械臂控制**: 支持X5和L5机械臂的关节控制
- ✅ **双臂支持**: 支持同时控制左右两个机械臂
- ✅ **原生SDK**: 直接使用ARX5 C++ SDK的Python绑定
- ✅ **轻量级**: 不包含额外的相机和机器人控制功能

## 安装

### 方法1: 从源码安装

```bash
# 进入包目录
cd arx5_sdk_only

# 安装包
pip install -e .
```

### 方法2: 直接使用

```bash
# 将arx5_sdk_only目录复制到你的项目中
# 然后在Python中导入
import sys
sys.path.append('/path/to/arx5_sdk_only')
import arx5_sdk
```

## 快速开始

### 1. 基本使用

```python
import arx5_sdk
from arx5_sdk import arx5, ARX5_AVAILABLE

# 检查SDK是否可用
if not ARX5_AVAILABLE:
    print("ARX5 SDK不可用")
    exit(1)

# 创建机械臂控制器
left_arm = arx5.Arx5JointController("X5", "can0")

# 获取配置
robot_config = left_arm.get_robot_config()
controller_config = left_arm.get_controller_config()

# 获取当前状态
joint_state = left_arm.get_joint_state()
print(f"当前关节位置: {joint_state.pos()}")
print(f"当前夹爪位置: {joint_state.gripper_pos}")

# 控制机械臂
cmd = arx5.JointState(robot_config.joint_dof)
cmd.pos()[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
cmd.gripper_pos = 0.5

left_arm.set_joint_cmd(cmd)

# 清理
del left_arm
```

### 2. 双臂控制

```python
import arx5_sdk
from arx5_sdk import arx5

# 创建双臂控制器
left_arm = arx5.Arx5JointController("X5", "can0")
right_arm = arx5.Arx5JointController("X5", "can1")

# 获取配置
robot_config = left_arm.get_robot_config()

# 创建命令
left_cmd = arx5.JointState(robot_config.joint_dof)
right_cmd = arx5.JointState(robot_config.joint_dof)

# 设置目标位置
left_cmd.pos()[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
right_cmd.pos()[:] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]

# 发送命令
left_arm.set_joint_cmd(left_cmd)
right_arm.set_joint_cmd(right_cmd)

# 清理
del left_arm
del right_arm
```

## API参考

### 主要类

- `arx5.Arx5JointController(model, interface)`: 关节控制器
- `arx5.JointState(dof)`: 关节状态
- `arx5.RobotConfig`: 机器人配置
- `arx5.ControllerConfig`: 控制器配置

### 主要方法

#### Arx5JointController

- `get_robot_config()`: 获取机器人配置
- `get_controller_config()`: 获取控制器配置
- `get_joint_state()`: 获取关节状态
- `set_joint_cmd(cmd)`: 设置关节命令
- `reset_to_home()`: 重置到初始位置
- `set_log_level(level)`: 设置日志级别

#### JointState

- `pos()`: 获取/设置关节位置
- `vel()`: 获取/设置关节速度
- `gripper_pos`: 获取/设置夹爪位置

## 系统要求

### 硬件要求

- ARX5机械臂 (X5或L5)
- CAN接口支持
- Linux系统

### 软件要求

- Python 3.8+
- numpy
- CAN接口驱动

## 故障排除

### 1. CAN接口问题

```bash
# 检查CAN接口
ip link show can0
ip link show can1

# 启动CAN接口
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

### 2. 库文件问题

确保以下库文件在正确位置：
- `arx5_interface.cpython-*.so`
- `libhardware.so`
- `libsolver.so`

### 3. 权限问题

```bash
# 确保用户有CAN接口权限
sudo usermod -a -G dialout $USER
# 重新登录后生效
```

## 示例项目

### 简单轨迹控制

```python
import arx5_sdk
from arx5_sdk import arx5
import time
import numpy as np

# 创建控制器
arm = arx5.Arx5JointController("X5", "can0")
robot_config = arm.get_robot_config()

# 定义轨迹点
trajectory = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

# 执行轨迹
for point in trajectory:
    cmd = arx5.JointState(robot_config.joint_dof)
    cmd.pos()[:] = point
    arm.set_joint_cmd(cmd)
    time.sleep(1.0)

# 清理
del arm
```

### 实时控制循环

```python
import arx5_sdk
from arx5_sdk import arx5
import time

# 创建控制器
arm = arx5.Arx5JointController("X5", "can0")
robot_config = arm.get_robot_config()
controller_config = arm.get_controller_config()

# 控制循环
dt = controller_config.controller_dt
for i in range(1000):
    # 获取当前状态
    state = arm.get_joint_state()
    current_pos = state.pos()
    
    # 计算目标位置 (简单的正弦波轨迹)
    target_pos = 0.1 * np.sin(2 * np.pi * i * dt)
    cmd = arx5.JointState(robot_config.joint_dof)
    cmd.pos()[:] = [target_pos] * robot_config.joint_dof
    
    # 发送命令
    arm.set_joint_cmd(cmd)
    
    # 等待下一个控制周期
    time.sleep(dt)

# 清理
del arm
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个SDK包！
"""
        )

    print(f"✓ README.md 创建: {readme_file}")

    # 6. 创建requirements.txt
    print("\n6. 创建requirements.txt...")

    requirements_file = package_dir / "requirements.txt"
    with open(requirements_file, "w", encoding="utf-8") as f:
        f.write(
            """# ARX5 SDK依赖
numpy>=1.19.0
"""
        )

    print(f"✓ requirements.txt 创建: {requirements_file}")

    # 7. 创建安装脚本
    print("\n7. 创建安装脚本...")

    install_file = package_dir / "install.sh"
    with open(install_file, "w", encoding="utf-8") as f:
        f.write(
            """#!/bin/bash
# ARX5 SDK安装脚本

echo "安装ARX5 SDK依赖..."

# 安装Python依赖
pip install numpy

echo "安装ARX5 SDK..."
pip install -e .

echo "✓ 安装完成"
echo ""
echo "使用方法:"
echo "  python example_usage.py"
echo "  python -c \"import arx5_sdk; print('ARX5 SDK可用')\""
"""
        )

    # 设置执行权限
    os.chmod(install_file, 0o755)
    print(f"✓ 安装脚本创建: {install_file}")

    # 8. 创建.gitignore
    print("\n8. 创建.gitignore...")

    gitignore_file = package_dir / ".gitignore"
    with open(gitignore_file, "w", encoding="utf-8") as f:
        f.write(
            """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db
"""
        )

    print(f"✓ .gitignore 创建: {gitignore_file}")

    print("\n" + "=" * 60)
    print("纯ARX5 SDK包创建完成!")
    print("=" * 60)
    print(f"包目录: {package_dir}")
    print("\n包含的文件:")
    print("- arx5_sdk/ (ARX5 SDK核心文件)")
    print("- __init__.py (Python包初始化)")
    print("- setup.py (安装脚本)")
    print("- example_usage.py (使用示例)")
    print("- README.md (使用说明)")
    print("- requirements.txt (依赖列表)")
    print("- install.sh (安装脚本)")
    print("\n使用方法:")
    print(f"  cd {package_dir}")
    print("  pip install -e .")
    print("  python example_usage.py")


if __name__ == "__main__":
    create_arx5_sdk_only()
