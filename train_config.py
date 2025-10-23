#!/usr/bin/env python3
"""
训练配置管理

用于加载和管理SDXL inpainting训练配置
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型配置"""
    pretrained_model_name_or_path: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    revision: Optional[str] = None
    variant: Optional[str] = None
    cache_dir: str = "./models"


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_data_dir: str = "./data/train"
    image_column: str = "image"
    caption_column: str = "text"
    max_train_samples: Optional[int] = None
    caption_file: Optional[str] = "./data/captions.json"


@dataclass
class ImageConfig:
    """图像处理配置"""
    resolution: int = 1024
    center_crop: bool = False
    random_flip: bool = True
    mask_type: str = "random"  # random, provided, mixed
    mask_ratio_range: list = field(default_factory=lambda: [0.1, 0.3])


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    train_batch_size: int = 1
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    
    # 学习率配置
    learning_rate: float = 1e-5
    scale_lr: bool = False
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    
    # 优化器配置
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 高级训练选项
    prediction_type: Optional[str] = None
    snr_gamma: Optional[float] = None
    noise_offset: float = 0.1
    input_perturbation: float = 0.1
    
    # 内存和性能优化
    mixed_precision: str = "fp16"
    enable_xformers_memory_efficient_attention: bool = True
    allow_tf32: bool = True
    dataloader_num_workers: int = 4
    
    # EMA配置
    use_ema: bool = True


@dataclass
class ValidationConfig:
    """验证配置"""
    validation_prompts: list = field(default_factory=lambda: [
        "修复图像中的缺失部分",
        "填补图像中的空白区域", 
        "恢复图像的完整性",
        "修复损坏的图像区域",
        "高质量的图像修复"
    ])
    validation_steps: int = 500
    num_validation_images: int = 4


@dataclass
class LoggingConfig:
    """日志和保存配置"""
    output_dir: str = "./output"
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    tracker_project_name: str = "sdxl-inpainting-finetune"
    
    # 检查点配置
    checkpointing_steps: int = 1000
    checkpoints_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Hub配置
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None


@dataclass
class Config:
    """完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_from_yaml(config_path: str) -> Config:
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigManager.from_dict(config_dict)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> Config:
        """从字典创建配置"""
        config = Config()
        
        # 更新各个配置部分
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        if 'image' in config_dict:
            config.image = ImageConfig(**config_dict['image'])
        
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        if 'validation' in config_dict:
            config.validation = ValidationConfig(**config_dict['validation'])
        
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        if 'seed' in config_dict:
            config.seed = config_dict['seed']
        
        return config
    
    @staticmethod
    def save_to_yaml(config: Config, output_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            'model': config.model.__dict__,
            'data': config.data.__dict__,
            'image': config.image.__dict__,
            'training': config.training.__dict__,
            'validation': config.validation.__dict__,
            'logging': config.logging.__dict__,
            'seed': config.seed,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def to_args(config: Config) -> argparse.Namespace:
        """将配置转换为argparse.Namespace对象"""
        args_dict = {}
        
        # 展平配置
        for section_name, section in [
            ('model', config.model),
            ('data', config.data), 
            ('image', config.image),
            ('training', config.training),
            ('validation', config.validation),
            ('logging', config.logging),
        ]:
            for key, value in section.__dict__.items():
                args_dict[key] = value
        
        args_dict['seed'] = config.seed
        
        return argparse.Namespace(**args_dict)
    
    @staticmethod
    def create_default_config(output_path: str):
        """创建默认配置文件"""
        config = Config()
        ConfigManager.save_to_yaml(config, output_path)
        print(f"默认配置文件已创建: {output_path}")
    
    @staticmethod
    def validate_config(config: Config) -> bool:
        """验证配置的有效性"""
        errors = []
        
        # 验证模型配置
        if not config.model.pretrained_model_name_or_path:
            errors.append("模型路径不能为空")
        
        # 验证数据配置
        if not config.data.dataset_name and not config.data.train_data_dir:
            errors.append("必须指定数据集名称或训练数据目录")
        
        if config.data.train_data_dir and not os.path.exists(config.data.train_data_dir):
            errors.append(f"训练数据目录不存在: {config.data.train_data_dir}")
        
        # 验证图像配置
        if config.image.resolution <= 0:
            errors.append("图像分辨率必须大于0")
        
        if config.image.mask_type not in ['random', 'provided', 'mixed']:
            errors.append("遮罩类型必须是 'random', 'provided', 或 'mixed'")
        
        # 验证训练配置
        if config.training.train_batch_size <= 0:
            errors.append("批次大小必须大于0")
        
        if config.training.learning_rate <= 0:
            errors.append("学习率必须大于0")
        
        if config.training.mixed_precision not in ['no', 'fp16', 'bf16']:
            errors.append("混合精度必须是 'no', 'fp16', 或 'bf16'")
        
        # 验证日志配置
        if not config.logging.output_dir:
            errors.append("输出目录不能为空")
        
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("配置验证通过")
        return True


def create_training_configs():
    """创建不同场景的训练配置模板"""
    
    # 快速测试配置
    quick_config = Config()
    quick_config.training.num_train_epochs = 5
    quick_config.training.train_batch_size = 2
    quick_config.training.checkpointing_steps = 100
    quick_config.validation.validation_steps = 50
    ConfigManager.save_to_yaml(quick_config, "config_quick_test.yaml")
    
    # 高质量训练配置
    quality_config = Config()
    quality_config.training.num_train_epochs = 200
    quality_config.training.train_batch_size = 1
    quality_config.training.gradient_accumulation_steps = 8
    quality_config.training.learning_rate = 5e-6
    quality_config.training.lr_scheduler = "cosine_with_restarts"
    quality_config.training.use_ema = True
    ConfigManager.save_to_yaml(quality_config, "config_high_quality.yaml")
    
    # 大数据集配置
    large_dataset_config = Config()
    large_dataset_config.training.train_batch_size = 4
    large_dataset_config.training.gradient_accumulation_steps = 2
    large_dataset_config.training.dataloader_num_workers = 8
    large_dataset_config.training.checkpointing_steps = 2000
    large_dataset_config.validation.validation_steps = 1000
    ConfigManager.save_to_yaml(large_dataset_config, "config_large_dataset.yaml")
    
    print("训练配置模板已创建:")
    print("  - config_quick_test.yaml: 快速测试配置")
    print("  - config_high_quality.yaml: 高质量训练配置")  
    print("  - config_large_dataset.yaml: 大数据集配置")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="配置管理工具")
    parser.add_argument("--action", type=str, choices=['create_default', 'create_templates', 'validate'],
                       default='create_default', help="执行的操作")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    if args.action == 'create_default':
        ConfigManager.create_default_config(args.config)
    elif args.action == 'create_templates':
        create_training_configs()
    elif args.action == 'validate':
        if os.path.exists(args.config):
            config = ConfigManager.load_from_yaml(args.config)
            ConfigManager.validate_config(config)
        else:
            print(f"配置文件不存在: {args.config}")