#!/usr/bin/env python3
"""
训练配置文件
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 模型配置
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir: str = "./sdxl-inpaint-finetuned"
    
    # 数据配置
    train_data_dir: str = "./data/train"
    resolution: int = 1024
    center_crop: bool = False
    
    # 训练配置
    train_batch_size: int = 1
    max_train_steps: int = 10000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 混合精度训练
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    
    # 数据加载配置
    dataloader_num_workers: int = 4
    
    # 日志和保存配置
    logging_steps: int = 100
    save_steps: int = 1000
    seed: int = 42
    
    # Wandb配置
    use_wandb: bool = False
    wandb_project: str = "sdxl-inpaint-finetune"
    wandb_run_name: Optional[str] = None
    
    # 数据增强配置
    mask_generation_strategy: str = "random"  # "random", "irregular", "center"
    mask_probability: float = 0.5
    min_mask_ratio: float = 0.1
    max_mask_ratio: float = 0.5
    
    # 验证配置
    validation_steps: int = 500
    num_validation_images: int = 4
    
    # 其他配置
    gradient_checkpointing: bool = True
    use_ema: bool = False
    ema_decay: float = 0.999
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """从YAML文件加载配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """保存配置到YAML文件"""
        config_dict = self.__dict__.copy()
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def validate(self):
        """验证配置"""
        # 检查必要的目录
        if not os.path.exists(self.train_data_dir):
            raise ValueError(f"训练数据目录不存在: {self.train_data_dir}")
        
        # 检查模型路径
        if not os.path.exists(self.pretrained_model_name_or_path) and not self.pretrained_model_name_or_path.startswith("stabilityai/"):
            raise ValueError(f"预训练模型路径无效: {self.pretrained_model_name_or_path}")
        
        # 检查参数范围
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于0")
        
        if self.train_batch_size <= 0:
            raise ValueError("批次大小必须大于0")
        
        if self.max_train_steps <= 0:
            raise ValueError("最大训练步数必须大于0")
        
        if not 0 <= self.mask_probability <= 1:
            raise ValueError("掩码概率必须在0-1之间")
        
        if not 0 <= self.min_mask_ratio <= self.max_mask_ratio <= 1:
            raise ValueError("掩码比例范围无效")


# 默认配置
DEFAULT_CONFIG = TrainingConfig()

# 预定义配置
CONFIGS = {
    "small": TrainingConfig(
        train_batch_size=1,
        max_train_steps=5000,
        learning_rate=5e-6,
        resolution=512,
    ),
    
    "medium": TrainingConfig(
        train_batch_size=2,
        max_train_steps=10000,
        learning_rate=1e-5,
        resolution=1024,
    ),
    
    "large": TrainingConfig(
        train_batch_size=4,
        max_train_steps=20000,
        learning_rate=2e-5,
        resolution=1024,
        gradient_accumulation_steps=2,
    ),
    
    "debug": TrainingConfig(
        train_batch_size=1,
        max_train_steps=100,
        learning_rate=1e-5,
        resolution=512,
        logging_steps=10,
        save_steps=50,
    ),
}


def get_config(config_name: str = "medium") -> TrainingConfig:
    """获取预定义配置"""
    if config_name not in CONFIGS:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]


def create_config_file(config_name: str = "medium", output_path: str = "config.yaml"):
    """创建配置文件"""
    config = get_config(config_name)
    config.to_yaml(output_path)
    print(f"配置文件已创建: {output_path}")


if __name__ == "__main__":
    # 创建默认配置文件
    create_config_file("medium", "config.yaml")
    print("默认配置文件已创建")