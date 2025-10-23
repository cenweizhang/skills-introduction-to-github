#!/bin/bash

# SDXL Inpainting 微调训练快速启动脚本

set -e

echo "🚀 SDXL Inpainting 微调训练快速启动"
echo "=================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python版本要求: >= 3.8，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU检测到:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU训练（速度较慢）"
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建Python虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "📈 升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 安装依赖包..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ 依赖包安装完成"
else
    echo "❌ requirements.txt文件不存在"
    exit 1
fi

# 创建必要目录
echo "📁 创建目录结构..."
mkdir -p data/train/images
mkdir -p data/train/masks
mkdir -p output
mkdir -p models
mkdir -p hf_cache

echo "✅ 目录结构创建完成"

# 创建默认配置文件
if [ ! -f "config.yaml" ]; then
    echo "⚙️  创建默认配置文件..."
    python run_training.py --create-config
    echo "✅ 默认配置文件创建完成"
else
    echo "✅ 配置文件已存在"
fi

# 创建配置模板
echo "📋 创建配置模板..."
python train_config.py --action create_templates

# 创建示例数据
echo "🎨 准备示例数据..."
python data_utils.py --data_root ./data/train --action prepare

echo ""
echo "🎉 快速启动完成！"
echo ""
echo "接下来的步骤："
echo "1. 将您的训练图像放入 ./data/train/images/ 目录"
echo "2. (可选) 将对应的遮罩放入 ./data/train/masks/ 目录"
echo "3. (可选) 编辑 config.yaml 调整训练参数"
echo "4. 运行训练: python run_training.py"
echo ""
echo "快速测试训练:"
echo "python run_training.py --config config_quick_test.yaml"
echo ""
echo "高质量训练:"
echo "python run_training.py --config config_high_quality.yaml"
echo ""
echo "检查环境和配置:"
echo "python run_training.py --check-only"
echo ""
echo "更多帮助请查看 README.md 文件"
echo "=================================="