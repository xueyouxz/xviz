# NuScenes到XVIZ协议转换器

将NuScenes数据集转换为XVIZ协议格式,用于自动驾驶数据可视化。

## 功能特性

- ✅ **激光雷达点云**: 转换LIDAR_TOP点云数据
- ✅ **6路相机**: 转换所有相机图像(前/后/左/右)
- ✅ **毫米波雷达**: 转换5个雷达传感器数据
- ✅ **3D标注**: 转换物体的3D边界框
- ✅ **历史轨迹**: 显示对象过去3秒的运动轨迹
- ✅ **未来预测**: 预测对象未来3秒(6步)的位置

## 项目结构

```
nuscenes_converter/
├── README.md                    # 说明文档
├── main.py                      # 主程序入口
├── config.yaml                  # 配置文件
├── utils.py                     # 工具函数
├── nuscenes_converter.py        # 转换器主类
└── converters/                  # 各类数据转换器
    ├── lidar_converter.py       # 激光雷达转换器
    ├── camera_converter.py      # 相机转换器
    ├── radar_converter.py       # 毫米波雷达转换器
    ├── annotation_converter.py  # 标注对象转换器
    └── future_object_converter.py  # 未来轨迹转换器
```

## 依赖安装

```bash
pip install nuscenes-devkit
pip install xviz-avs
pip install numpy
pip install pillow
pip install pyquaternion
pip install pyyaml
```

## 配置说明

编辑 `config.yaml`:

```yaml
version: 'v1.0-mini'              # NuScenes版本
scenes: 'mini_val'                # 场景集合
data_root: "/path/to/nuscenes"    # 数据集根目录
sample_limit: 40                  # 每个场景转换的最大帧数
output_dir: "./xviz_output"       # 输出目录
image_max_width: 400              # 图像最大宽度(压缩)
image_max_height: 300             # 图像最大高度(压缩)
```

## 使用方法

### 1. 转换数据

```bash
python main.py
```

转换后的数据将保存在 `xviz_output` 目录下,每个场景一个子目录。

### 2. 数据格式

输出数据采用XVIZ GLB格式:
- `0-frame.json`: 元数据文件
- `1-frame.glb`, `2-frame.glb`, ...: 各帧数据

## XVIZ流定义

### 传感器数据流

| 流名称 | 类型 | 说明 |
|--------|------|------|
| `/lidar/points` | point | 激光雷达点云 |
| `/camera/cam_front` | image | 前向相机 |
| `/camera/cam_front_left` | image | 左前相机 |
| `/camera/cam_front_right` | image | 右前相机 |
| `/camera/cam_back` | image | 后向相机 |
| `/camera/cam_back_left` | image | 左后相机 |
| `/camera/cam_back_right` | image | 右后相机 |
| `/radar/radar_front` | point | 前向雷达 |
| `/radar/radar_front_left` | point | 左前雷达 |
| `/radar/radar_front_right` | point | 右前雷达 |
| `/radar/radar_back_left` | point | 左后雷达 |
| `/radar/radar_back_right` | point | 右后雷达 |

### 标注数据流

| 流名称 | 类型 | 说明 |
|--------|------|------|
| `/object/boxes` | polygon | 3D边界框 |
| `/object/trajectory` | polyline | 历史轨迹(过去3秒) |
| `/object/future_trajectory` | polyline | 未来轨迹(未来3秒) |
| `/object/future_boxes` | polygon | 未来位置边界框 |

## 坐标系说明

- **全局坐标系**: NuScenes原始数据使用全局坐标系
- **车辆坐标系**: XVIZ使用车辆相对坐标系
  - X轴: 车辆前进方向
  - Y轴: 车辆左侧方向
  - Z轴: 车辆上方方向

## 代码架构

### NuscenesConverter (主转换器)
负责协调各个子转换器,管理转换流程:
1. 加载场景和帧数据
2. 初始化各个子转换器
3. 生成XVIZ元数据
4. 逐帧调用各转换器

### 子转换器
每个转换器负责特定类型的数据:
- 实现 `load()` 方法: 加载数据
- 实现 `convert_message()` 方法: 转换单帧数据
- 实现 `get_metadata()` 方法: 定义流元数据

## 性能优化建议

1. **图像压缩**: 调整 `image_max_width/height` 减小数据量
2. **限制帧数**: 设置 `sample_limit` 只转换部分帧
3. **选择性转换**: 在代码中注释掉不需要的转换器

## 故障排查

### 文件未找到
检查 `config.yaml` 中的 `data_root` 路径是否正确

### 内存不足
- 减小 `image_max_width/height`
- 减小 `sample_limit`
- 逐个场景转换

### 转换错误
查看控制台输出的详细错误信息,通常是数据格式或路径问题

## 扩展开发

### 添加新的数据流

1. 在 `converters/` 下创建新的转换器类
2. 实现 `load()`, `convert_message()`, `get_metadata()` 方法
3. 在 `nuscenes_converter.py` 中注册新转换器

### 修改可视化样式

在各转换器的 `get_metadata()` 方法中修改 `stream_style()` 参数

## 参考资料

- [NuScenes数据集](https://www.nuscenes.org/)
- [XVIZ协议文档](https://avs.auto/#/xviz/overview/introduction)
- [streetscape.gl](https://avs.auto/#/streetscape.gl/overview/introduction)

## 许可证

本项目仅用于演示和学习目的。使用NuScenes数据需遵守其[使用条款](https://www.nuscenes.org/terms-of-use)。