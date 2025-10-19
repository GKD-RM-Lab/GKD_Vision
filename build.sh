#!/bin/bash
# ==============================================================
# 🚀 GKD_Vision 一键构建 & 部署 & 运行脚本（通用增强版）
# ==============================================================

yellow="\033[1;33m"
blue="\033[1;34m"
red="\033[1;31m"
reset="\033[0m"

# ==============================================================
# 0. 参数检查
# ==============================================================
TARGET="$1"
if [ -z "$TARGET" ]; then
    echo -e "${red}用法: ./run.sh [infantry|hero|sentry_l|sentry_r|balance|dronse] [-v|-s]${reset}"
    exit 1
fi

imshow=0
verbose=1
shift
if [ "$1" = "-v" ]; then
    verbose=1
elif [ "$1" = "-s" ]; then
    imshow=1
fi

# ==============================================================
# 1. 环境准备
# ==============================================================
export LD_LIBRARY_PATH=$(pwd)/lib/64:$LD_LIBRARY_PATH
SYS_DIR="/etc/visionlib"

sudo install -d -m755 "$SYS_DIR"
sudo install -d -m755 "$SYS_DIR/models"
sudo install -d -m755 "$SYS_DIR/forward_config"

echo -e "${yellow}环境已准备：$SYS_DIR${reset}"

# ==============================================================
# 2. 构建
# ==============================================================
echo -e "${yellow}<--- Start CMake (${TARGET}) --->${reset}"
rm -rf build && mkdir build && cd build
cmake -DARMY=${TARGET} ..
make -j$(nproc)
cd ..

# ==============================================================
# 3. 模型文件安装
# ==============================================================
echo -e "${yellow}<--- 安装模型文件 --->${reset}"
sudo cp -f include/model/yolov7_quantized/rmyolov7-latest-int8.* "$SYS_DIR/models/" 2>/dev/null || true

# ==============================================================
# 4. 兵种配置安装
# ==============================================================
FORWARD_CONFIG_DIR="$SYS_DIR/forward_config/${TARGET}"
sudo install -d -m755 "$FORWARD_CONFIG_DIR"
sudo cp -rf config/${TARGET}/* "$FORWARD_CONFIG_DIR/" 2>/dev/null || true
sudo cp -rf config/forward_config/* "$FORWARD_CONFIG_DIR/" 2>/dev/null || true

# Copy the main Config.json file
sudo cp -f data/uniconfig/Config.json "$SYS_DIR/" 2>/dev/null || true

echo -e "${yellow}配置文件已更新到：$FORWARD_CONFIG_DIR${reset}"