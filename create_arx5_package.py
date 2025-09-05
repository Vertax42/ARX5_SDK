#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_arx5_package.py - 创建ARX5独立包
"""

import os
import shutil
from pathlib import Path


def create_arx5_package():
    """创建ARX5独立包"""

    # 源目录和目标目录
    source_dir = Path(__file__).parent
    package_dir = source_dir / "arx5_package"

    print("创建ARX5独立包...")
    print(f"源目录: {source_dir}")
    print(f"包目录: {package_dir}")

    # 创建包目录结构
    package_dir.mkdir(exist_ok=True)

    # 1. 复制ARX5 SDK
    print("\n1. 复制ARX5 SDK...")
    sdk_source = source_dir / "arx5-sdk"
    sdk_dest = package_dir / "arx5" / "sdk"

    if sdk_source.exists():
        if sdk_dest.exists():
            shutil.rmtree(sdk_dest)
        shutil.copytree(sdk_source, sdk_dest)
        print(f"✓ ARX5 SDK 复制到: {sdk_dest}")
    else:
        print(f"❌ ARX5 SDK 源目录不存在: {sdk_source}")

    # 2. 复制相机系统
    print("\n2. 复制相机系统...")
    cameras_source = source_dir / "cameras"
    cameras_dest = package_dir / "arx5" / "cameras"

    if cameras_source.exists():
        if cameras_dest.exists():
            shutil.rmtree(cameras_dest)
        shutil.copytree(cameras_source, cameras_dest)
        print(f"✓ 相机系统 复制到: {cameras_dest}")
    else:
        print(f"❌ 相机系统 源目录不存在: {cameras_source}")

    # 3. 复制机器人控制代码
    print("\n3. 复制机器人控制代码...")

    # 复制AC_one.py作为参考
    ac_one_source = source_dir / "AC_one.py"
    if ac_one_source.exists():
        shutil.copy2(ac_one_source, package_dir / "AC_one_reference.py")
        print(f"✓ AC_one.py 复制为参考文件")

    # 4. 复制数据采集代码
    print("\n4. 复制数据采集代码...")

    collect_files = [
        "collect_no_ros.py",
        "collect_demo.py",
        "test_collect.py",
        "simple_collect_test.py",
    ]

    for file_name in collect_files:
        file_source = source_dir / file_name
        if file_source.exists():
            shutil.copy2(file_source, package_dir / file_name)
            print(f"✓ {file_name} 复制到包目录")

    # 5. 复制配置文件
    print("\n5. 复制配置文件...")

    config_source = source_dir / "data" / "config.yaml"
    if config_source.exists():
        config_dest = package_dir / "data"
        config_dest.mkdir(exist_ok=True)
        shutil.copy2(config_source, config_dest / "config.yaml")
        print(f"✓ 配置文件 复制到: {config_dest}")

    # 6. 创建示例脚本
    print("\n6. 创建示例脚本...")

    example_script = package_dir / "example_usage.py"
    with open(example_script, "w", encoding="utf-8") as f:
        f.write(
            '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5包使用示例
"""

from arx5 import ARX5Robot, create_default_config
import time

def main():
    """主函数"""
    print("ARX5机器人使用示例")
    print("=" * 40)
    
    # 创建默认配置
    config = create_default_config()
    
    # 创建机器人实例
    robot = ARX5Robot(config)
    
    try:
        # 获取机械臂状态
        print("\\n获取机械臂状态...")
        left_state = robot.get_left_arm_state()
        right_state = robot.get_right_arm_state()
        
        if left_state:
            print(f"左臂关节位置: {left_state['joint_positions']}")
        else:
            print("左臂未连接")
            
        if right_state:
            print(f"右臂关节位置: {right_state['joint_positions']}")
        else:
            print("右臂未连接")
        
        # 获取相机图像
        print("\\n获取相机图像...")
        camera_frames = robot.get_camera_frames()
        print(f"获取到 {len(camera_frames)} 个相机图像")
        
        for cam_name, img in camera_frames.items():
            if img is not None:
                print(f"  {cam_name}: {img.shape}")
            else:
                print(f"  {cam_name}: 无图像")
        
        # 简单的机械臂控制示例
        print("\\n机械臂控制示例...")
        if left_state:
            # 获取当前位置
            current_pos = left_state['joint_positions'].copy()
            print(f"当前左臂位置: {current_pos}")
            
            # 稍微移动一下
            new_pos = current_pos + 0.01
            print(f"移动到新位置: {new_pos}")
            robot.move_left_arm_joint(new_pos.tolist())
            
            time.sleep(1)
            
            # 回到原位置
            print("回到原位置...")
            robot.move_left_arm_joint(current_pos.tolist())
        
        print("\\n✓ 示例完成")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 停止系统
        robot.stop_system()

if __name__ == "__main__":
    main()
'''
        )

    print(f"✓ 示例脚本创建: {example_script}")

    # 7. 创建安装脚本
    print("\n7. 创建安装脚本...")

    install_script = package_dir / "install.sh"
    with open(install_script, "w", encoding="utf-8") as f:
        f.write(
            """#!/bin/bash
# ARX5包安装脚本

echo "安装ARX5包依赖..."

# 安装Python依赖
pip install numpy opencv-python h5py pyyaml

# 可选依赖
pip install pyrealsense2 pyttsx3

echo "安装ARX5包..."
pip install -e .

echo "✓ 安装完成"
echo ""
echo "使用方法:"
echo "  python example_usage.py"
echo "  python -c \"from arx5 import ARX5Robot; robot = ARX5Robot(); print('ARX5包可用')\""
"""
        )

    # 设置执行权限
    os.chmod(install_script, 0o755)
    print(f"✓ 安装脚本创建: {install_script}")

    # 8. 创建requirements.txt
    print("\n8. 创建requirements.txt...")

    requirements_file = package_dir / "requirements.txt"
    with open(requirements_file, "w", encoding="utf-8") as f:
        f.write(
            """# ARX5包依赖
numpy>=1.19.0
opencv-python>=4.5.0
h5py>=3.0.0
pyyaml>=6.0

# 可选依赖
pyrealsense2>=2.50.0  # RealSense相机支持
pyttsx3>=2.90         # 语音提示支持
"""
        )

    print(f"✓ requirements.txt 创建: {requirements_file}")

    # 9. 创建.gitignore
    print("\n9. 创建.gitignore...")

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

# 数据文件
datasets/
*.hdf5
*.h5

# 日志文件
*.log

# 临时文件
*.tmp
*.temp

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
    print("ARX5独立包创建完成!")
    print("=" * 60)
    print(f"包目录: {package_dir}")
    print("\n包含的文件:")
    print("- arx5/ (主要包)")
    print("- arx5/sdk/ (ARX5 SDK)")
    print("- arx5/cameras/ (相机系统)")
    print("- setup.py (安装脚本)")
    print("- example_usage.py (使用示例)")
    print("- install.sh (安装脚本)")
    print("- requirements.txt (依赖列表)")
    print("- README.md (使用说明)")
    print("\n使用方法:")
    print(f"  cd {package_dir}")
    print("  pip install -e .")
    print("  python example_usage.py")


if __name__ == "__main__":
    create_arx5_package()
