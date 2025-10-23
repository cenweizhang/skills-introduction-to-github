#!/usr/bin/env python3
"""
训练启动脚本

提供简化的训练启动接口，支持配置文件和命令行参数
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from train_config import ConfigManager
from train_sdxl_inpainting import main as train_main


def setup_environment():
    """设置训练环境"""
    # 设置CUDA相关环境变量
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    
    # 设置HuggingFace缓存目录
    os.environ.setdefault("HF_HOME", "./hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")
    os.environ.setdefault("DIFFUSERS_CACHE", "./hf_cache/diffusers")
    
    # 设置其他优化选项
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        "torch", "torchvision", "diffusers", "transformers", 
        "accelerate", "PIL", "datasets", "yaml"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True


def create_directories(config):
    """创建必要的目录"""
    directories = [
        config.logging.output_dir,
        config.model.cache_dir,
        "./hf_cache",
        "./hf_cache/transformers", 
        "./hf_cache/diffusers",
    ]
    
    if config.data.train_data_dir:
        directories.append(config.data.train_data_dir)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 创建目录: {directory}")


def print_training_info(config):
    """打印训练信息"""
    print("\n" + "="*60)
    print("🚀 SDXL Inpainting 微调训练")
    print("="*60)
    print(f"📊 模型: {config.model.pretrained_model_name_or_path}")
    print(f"📁 数据目录: {config.data.train_data_dir}")
    print(f"🖼️  图像分辨率: {config.image.resolution}")
    print(f"📦 批次大小: {config.training.train_batch_size}")
    print(f"🔄 训练轮数: {config.training.num_train_epochs}")
    print(f"📈 学习率: {config.training.learning_rate}")
    print(f"💾 输出目录: {config.logging.output_dir}")
    print(f"🎯 混合精度: {config.training.mixed_precision}")
    print(f"🌱 随机种子: {config.seed}")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SDXL Inpainting 微调训练启动脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查环境和配置，不开始训练"
    )
    parser.add_argument(
        "--create-config",
        action="store_true", 
        help="创建默认配置文件"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="训练数据目录（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批次大小（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="学习率（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="训练轮数（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="从检查点恢复训练"
    )
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 创建默认配置文件
    if args.create_config:
        ConfigManager.create_default_config(args.config)
        print(f"✅ 默认配置文件已创建: {args.config}")
        return
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print("使用 --create-config 创建默认配置文件")
        return
    
    try:
        config = ConfigManager.load_from_yaml(args.config)
        print(f"✅ 配置文件加载成功: {args.config}")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return
    
    # 应用命令行参数覆盖
    if args.data_dir:
        config.data.train_data_dir = args.data_dir
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    if args.batch_size:
        config.training.train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.resume:
        config.logging.resume_from_checkpoint = args.resume
    
    # 验证配置
    if not ConfigManager.validate_config(config):
        return
    
    # 创建必要目录
    create_directories(config)
    
    # 打印训练信息
    print_training_info(config)
    
    # 仅检查模式
    if args.check_only:
        print("✅ 环境检查完成，配置有效")
        return
    
    # 确认开始训练
    try:
        response = input("\n是否开始训练? (y/N): ").strip().lower()
        if response not in ['y', 'yes', '是']:
            print("训练已取消")
            return
    except KeyboardInterrupt:
        print("\n训练已取消")
        return
    
    # 转换配置为训练参数
    train_args = ConfigManager.to_args(config)
    
    # 开始训练
    try:
        print("\n🚀 开始训练...")
        
        # 临时修改sys.argv以传递参数给训练脚本
        original_argv = sys.argv.copy()
        sys.argv = ['train_sdxl_inpainting.py']
        
        # 将配置转换为命令行参数
        for key, value in train_args.__dict__.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f'--{key}')
                elif isinstance(value, list):
                    sys.argv.append(f'--{key}')
                    sys.argv.extend([str(v) for v in value])
                else:
                    sys.argv.extend([f'--{key}', str(value)])
        
        # 调用训练主函数
        train_main()
        
        # 恢复原始argv
        sys.argv = original_argv
        
        print("\n🎉 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()