# GKD_Vision 🚀

感谢天津大学北洋机甲战队开源的自瞄方案， 本自瞄方案仅北洋机甲的方案上进行了适配

## ✨ 特性

- 🎯 **精准目标检测** - 基于 OpenVINO 的 YOLOv7 模型实现高性能装甲板识别
- 🎯 **多目标识别** - 支持英雄、步兵、哨兵等多类目标的精确分类
- 🏃 **动态跟踪** - 使用卡尔曼滤波器实现平滑的目标跟踪
- 🎯 **PnP 解算** - 实现精确的 3D-2D 坐标转换和弹道补偿
- 🤖 **多兵种支持** - 支持英雄、步兵、哨兵等不同机器人类型
- 📸 **相机支持** - 集成海康威视相机驱动（HIK driver）
- ⚡ **高性能** - 优化的多线程处理架构

## 🚀 快速开始

### 构建与运行

```bash
# 进入项目目录
cd /path/to/GKD_Vision

# 构建并运行（指定机器人类型）
sudo ./run.sh infantry -s      # 步兵模式，带图像显示
sudo ./run.sh hero -v          # 英雄模式，详细输出
sudo ./run.sh sentry_l         # 左哨兵模式
sudo ./run.sh sentry_r         # 右哨兵模式
```
## ⚙️ 配置说明

配置文件位于 `config/[robot_type]/config.yaml`，主要参数包括：

- `model_path_xml` / `model_path_bin` - OpenVINO 模型路径
- `conf_threshold` - 检测置信度阈值
- `cam_gain` - 相机增益
- `cam_exptime` - 相机曝光时间
- `framerate` - 相机帧率
- `shoot_speed` - 弹丸初速度
- `armor_small_h/w` - 小装甲板尺寸
- `armor_large_h/w` - 大装甲板尺寸
