#!/bin/bash
# SDXL图像修复模型微调项目安装脚本

echo "SDXL图像修复模型微调项目安装脚本"
echo "=================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

echo "Python版本检查通过: $python_version"

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建项目目录..."
mkdir -p data/train
mkdir -p data/val
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

# 设置权限
echo "设置脚本权限..."
chmod +x train_sdxl_inpaint.py
chmod +x inference.py
chmod +x example_usage.py

# 创建示例配置文件
echo "创建示例配置文件..."
python3 -c "
from config import create_config_file
create_config_file('medium', 'config_medium.yaml')
create_config_file('small', 'config_small.yaml')
create_config_file('debug', 'config_debug.yaml')
print('示例配置文件已创建')
"

echo ""
echo "安装完成！"
echo "=========="
echo ""
echo "使用方法:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 准备训练数据到 data/train/ 目录"
echo "3. 开始训练: python train_sdxl_inpaint.py --config config.yaml"
echo "4. 运行推理: python inference.py --model_path ./sdxl-inpaint-finetuned --image input.jpg --prompt '描述'"
echo "5. 运行示例: python example_usage.py"
echo ""
echo "更多信息请查看 README.md"