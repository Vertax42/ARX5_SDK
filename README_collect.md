# AC_one机器人数据采集系统

基于AC_one机器人的非ROS数据采集系统，支持多相机和双臂机械臂的数据采集。

## 功能特性

- ✅ **非ROS架构**: 不依赖ROS，直接使用AC_one机器人系统
- ✅ **多相机支持**: 支持多个OpenCV相机同时采集
- ✅ **双臂机械臂**: 支持左右两个X5机械臂的数据采集
- ✅ **实时数据采集**: 多线程实时数据采集和处理
- ✅ **HDF5数据格式**: 使用HDF5格式保存数据，支持压缩
- ✅ **语音提示**: 支持语音反馈和提示
- ✅ **灵活配置**: 支持YAML配置文件和命令行参数

## 系统架构

```
collect_no_ros.py
├── DataCollector          # 数据采集器主类
├── Rate                   # 频率控制类
├── 数据采集线程            # 后台数据采集
├── 图像压缩和保存          # HDF5数据保存
└── AC_one集成             # 机器人系统集成
```

## 安装依赖

```bash
pip install h5py pyttsx3 pyyaml opencv-python numpy
```

## 快速开始

### 1. 基本使用

```bash
# 使用默认参数进行数据采集
python collect_no_ros.py

# 指定数据集目录和参数
python collect_no_ros.py --datasets ./my_datasets --max_timesteps 500 --frame_rate 30
```

### 2. 使用演示脚本

```bash
# 基本数据采集演示
python collect_demo.py 1

# 高级数据采集演示
python collect_demo.py 2

# 自定义配置演示
python collect_demo.py 3
```

### 3. 命令行参数

```bash
python collect_no_ros.py --help
```

主要参数：
- `--datasets`: 数据集保存目录
- `--max_timesteps`: 最大时间步数
- `--frame_rate`: 采集帧率
- `--camera_names`: 使用的相机名称列表
- `--use_depth_image`: 是否使用深度图像
- `--key_collect`: 是否使用键盘触发
- `--task`: 任务名称
- `--config`: 配置文件路径

## 配置文件

创建 `data/config.yaml` 文件：

```yaml
# 相机配置
camera_config:
  devices:
    camera_1: "/dev/video2"
    camera_2: "/dev/video4" 
    camera_3: "/dev/video6"
  resolution:
    width: 640
    height: 480
  fps: 30

# 机械臂配置
arm_config:
  left_arm:
    model: "X5"
    interface: "can0"
  right_arm:
    model: "X5"
    interface: "can1"

# 数据采集配置
data_collection:
  max_timesteps: 800
  frame_rate: 30
  image_compression_quality: 50
```

## 数据格式

采集的数据以HDF5格式保存，包含以下数据：

### 观测数据 (observations)
- `qpos`: 关节位置 (14维: 左臂7 + 右臂7)
- `qvel`: 关节速度 (14维)
- `eef`: 末端执行器位置 (14维: 左臂6+夹爪1 + 右臂6+夹爪1)
- `effort`: 关节力矩 (14维)
- `robot_base`: 机器人基座状态 (6维)
- `images/{camera_name}`: 相机图像数据

### 动作数据 (actions)
- `action`: 关节动作 (14维)
- `action_eef`: 末端执行器动作 (14维)

## 使用流程

1. **初始化系统**
   ```python
   collector = DataCollector(args, config)
   collector.initialize_robot()
   ```

2. **开始数据采集**
   ```python
   collector.start_episode(episode_idx)
   # 系统开始采集数据
   ```

3. **停止并保存数据**
   ```python
   timesteps, actions, actions_eef = collector.stop_episode()
   save_data(args, timesteps, actions, actions_eef, dataset_path)
   ```

## 高级功能

### 1. 自定义数据采集

```python
from collect_no_ros import DataCollector

# 创建自定义采集器
collector = DataCollector(args, config)
collector.initialize_robot()

# 开始采集
collector.start_episode(0)
# ... 执行机器人动作 ...
timesteps, actions, actions_eef = collector.stop_episode()
```

### 2. 实时数据监控

```python
# 在数据采集过程中可以实时获取数据
obs_data = collector._get_observation()
if obs_data:
    print(f"当前关节位置: {obs_data['qpos']}")
    print(f"相机图像数量: {len(obs_data['images'])}")
```

### 3. 数据预处理

```python
# 自定义数据预处理
def preprocess_data(obs_data):
    # 图像预处理
    for cam_name, img in obs_data['images'].items():
        # 调整图像大小
        img = cv2.resize(img, (320, 240))
        obs_data['images'][cam_name] = img
    
    # 关节位置归一化
    obs_data['qpos'] = (obs_data['qpos'] - mean_pos) / std_pos
    
    return obs_data
```

## 故障排除

### 1. 相机连接问题
```bash
# 检查相机设备
ls /dev/video*

# 测试相机
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### 2. 机械臂连接问题
```bash
# 检查CAN接口
ip link show can0
ip link show can1

# 测试机械臂连接
python AC_one.py
```

### 3. 数据保存问题
```bash
# 检查磁盘空间
df -h

# 检查HDF5文件
python -c "import h5py; f = h5py.File('dataset.hdf5', 'r'); print(list(f.keys()))"
```

## 性能优化

1. **调整帧率**: 根据任务需求调整 `frame_rate` 参数
2. **图像压缩**: 调整 `image_compression_quality` 参数
3. **内存管理**: 使用 `max_timesteps` 限制单次采集的数据量
4. **多线程**: 确保 `use_multithreading=True` 以获得最佳性能

## 扩展功能

### 1. 添加新的传感器
```python
def add_custom_sensor(self, sensor_name, sensor_data):
    """添加自定义传感器数据"""
    self.custom_sensors[sensor_name] = sensor_data
```

### 2. 自定义数据格式
```python
def custom_save_format(self, data, filepath):
    """自定义数据保存格式"""
    # 实现自定义保存逻辑
    pass
```

### 3. 实时数据可视化
```python
def visualize_data(self, obs_data):
    """实时数据可视化"""
    # 显示相机图像
    for cam_name, img in obs_data['images'].items():
        cv2.imshow(f"Camera {cam_name}", img)
    
    # 显示关节状态
    print(f"Joint positions: {obs_data['qpos']}")
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个数据采集系统！
