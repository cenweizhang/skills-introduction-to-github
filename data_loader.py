#!/usr/bin/env python3
"""
图像修复数据集加载器
支持多种数据格式和预处理方法
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from transformers import CLIPTokenizer

logger = logging.getLogger(__name__)


class InpaintDataset(Dataset):
    """图像修复数据集"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        size: int = 1024,
        center_crop: bool = False,
        max_length: int = 77,
        mask_generation_strategy: str = "random",
        mask_probability: float = 0.5,
        min_mask_ratio: float = 0.1,
        max_mask_ratio: float = 0.5,
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            tokenizer: 主文本编码器
            tokenizer_2: 第二个文本编码器
            size: 图像尺寸
            center_crop: 是否中心裁剪
            max_length: 文本最大长度
            mask_generation_strategy: 掩码生成策略
            mask_probability: 掩码概率
            min_mask_ratio: 最小掩码比例
            max_mask_ratio: 最大掩码比例
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.max_length = max_length
        self.mask_generation_strategy = mask_generation_strategy
        self.mask_probability = mask_probability
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        
        # 加载数据
        self.data_samples = self._load_data()
        logger.info(f"加载了 {len(self.data_samples)} 个数据样本")
        
    def _load_data(self) -> List[Dict]:
        """加载数据样本"""
        data_samples = []
        
        # 支持的数据格式
        if (self.data_dir / "metadata.json").exists():
            # JSON格式元数据
            data_samples = self._load_from_json()
        elif (self.data_dir / "images").exists():
            # 文件夹格式
            data_samples = self._load_from_folders()
        else:
            # 单个文件夹格式
            data_samples = self._load_from_single_folder()
            
        return data_samples
    
    def _load_from_json(self) -> List[Dict]:
        """从JSON文件加载数据"""
        with open(self.data_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        data_samples = []
        for item in metadata:
            if self._validate_sample(item):
                data_samples.append(item)
        
        return data_samples
    
    def _load_from_folders(self) -> List[Dict]:
        """从文件夹结构加载数据"""
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        captions_dir = self.data_dir / "captions"
        
        data_samples = []
        for img_path in images_dir.glob("*.jpg"):
            sample = {
                "image_path": str(img_path),
                "mask_path": str(masks_dir / img_path.name) if masks_dir.exists() else None,
                "caption_path": str(captions_dir / f"{img_path.stem}.txt") if captions_dir.exists() else None,
            }
            
            if self._validate_sample(sample):
                data_samples.append(sample)
        
        return data_samples
    
    def _load_from_single_folder(self) -> List[Dict]:
        """从单个文件夹加载数据"""
        data_samples = []
        for img_path in self.data_dir.glob("*.jpg"):
            sample = {
                "image_path": str(img_path),
                "mask_path": None,
                "caption_path": None,
            }
            
            if self._validate_sample(sample):
                data_samples.append(sample)
        
        return data_samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证数据样本"""
        # 检查图像文件是否存在
        if not os.path.exists(sample["image_path"]):
            return False
        
        # 检查掩码文件（如果存在）
        if sample["mask_path"] and not os.path.exists(sample["mask_path"]):
            return False
        
        # 检查描述文件（如果存在）
        if sample["caption_path"] and not os.path.exists(sample["caption_path"]):
            return False
        
        return True
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据样本"""
        sample = self.data_samples[idx]
        
        # 加载图像
        image = self._load_image(sample["image_path"])
        
        # 生成或加载掩码
        mask = self._generate_mask(image, sample)
        
        # 生成掩码图像
        masked_image = self._apply_mask(image, mask)
        
        # 加载或生成描述
        caption = self._load_caption(sample)
        
        # 编码文本
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_inputs_2 = self.tokenizer_2(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "masked_images": masked_image,
            "masks": mask,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "input_ids_2": text_inputs_2.input_ids.squeeze(0),
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载和预处理图像"""
        image = Image.open(image_path).convert("RGB")
        
        # 调整尺寸
        if self.center_crop:
            image = self._center_crop(image)
        else:
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # 转换为张量
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # 归一化到[-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def _center_crop(self, image: Image.Image) -> Image.Image:
        """中心裁剪图像"""
        width, height = image.size
        min_size = min(width, height)
        
        left = (width - min_size) // 2
        top = (height - min_size) // 2
        right = left + min_size
        bottom = top + min_size
        
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        return image
    
    def _generate_mask(self, image: torch.Tensor, sample: Dict) -> torch.Tensor:
        """生成掩码"""
        # 如果存在预定义掩码，加载它
        if sample["mask_path"] and os.path.exists(sample["mask_path"]):
            mask = self._load_mask(sample["mask_path"])
        else:
            # 生成随机掩码
            mask = self._generate_random_mask(image.shape[1:])
        
        return mask
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """加载预定义掩码"""
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return mask
    
    def _generate_random_mask(self, image_shape: Tuple[int, int]) -> torch.Tensor:
        """生成随机掩码"""
        if random.random() > self.mask_probability:
            # 不生成掩码
            return torch.zeros((1, *image_shape))
        
        mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        
        if self.mask_generation_strategy == "random":
            mask = self._generate_random_rectangular_mask(image_shape, mask_ratio)
        elif self.mask_generation_strategy == "irregular":
            mask = self._generate_irregular_mask(image_shape, mask_ratio)
        elif self.mask_generation_strategy == "center":
            mask = self._generate_center_mask(image_shape, mask_ratio)
        else:
            mask = self._generate_random_rectangular_mask(image_shape, mask_ratio)
        
        return mask
    
    def _generate_random_rectangular_mask(self, image_shape: Tuple[int, int], mask_ratio: float) -> torch.Tensor:
        """生成随机矩形掩码"""
        h, w = image_shape
        mask_area = int(h * w * mask_ratio)
        
        # 随机选择矩形区域
        mask_h = random.randint(int(mask_area / w), h)
        mask_w = int(mask_area / mask_h)
        
        # 随机位置
        start_h = random.randint(0, h - mask_h)
        start_w = random.randint(0, w - mask_w)
        
        mask = torch.zeros((1, h, w))
        mask[:, start_h:start_h + mask_h, start_w:start_w + mask_w] = 1.0
        
        return mask
    
    def _generate_irregular_mask(self, image_shape: Tuple[int, int], mask_ratio: float) -> torch.Tensor:
        """生成不规则掩码"""
        h, w = image_shape
        mask = torch.zeros((1, h, w))
        
        # 生成多个随机形状
        num_shapes = random.randint(1, 5)
        for _ in range(num_shapes):
            # 随机椭圆
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius_x = random.randint(10, min(w, h) // 4)
            radius_y = random.randint(10, min(w, h) // 4)
            
            # 创建椭圆掩码
            y, x = np.ogrid[:h, :w]
            ellipse_mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            mask[0] = torch.logical_or(mask[0], torch.from_numpy(ellipse_mask).float())
        
        return mask
    
    def _generate_center_mask(self, image_shape: Tuple[int, int], mask_ratio: float) -> torch.Tensor:
        """生成中心掩码"""
        h, w = image_shape
        mask_size = int(min(h, w) * np.sqrt(mask_ratio))
        
        mask = torch.zeros((1, h, w))
        start_h = (h - mask_size) // 2
        start_w = (w - mask_size) // 2
        
        mask[:, start_h:start_h + mask_size, start_w:start_w + mask_size] = 1.0
        
        return mask
    
    def _apply_mask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用掩码到图像"""
        # 将掩码区域设置为黑色
        masked_image = image.clone()
        masked_image = masked_image * (1 - mask)
        
        return masked_image
    
    def _load_caption(self, sample: Dict) -> str:
        """加载或生成描述"""
        if sample["caption_path"] and os.path.exists(sample["caption_path"]):
            with open(sample["caption_path"], "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            # 生成默认描述
            caption = "a high quality image"
        
        return caption


class InpaintDataCollator:
    """数据整理器"""
    
    def __init__(self, tokenizer, tokenizer_2, max_length: int = 77):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        # 堆叠张量
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        masked_images = torch.stack([item["masked_images"] for item in batch])
        masks = torch.stack([item["masks"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        input_ids_2 = torch.stack([item["input_ids_2"] for item in batch])
        
        return {
            "pixel_values": pixel_values,
            "masked_images": masked_images,
            "masks": masks,
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
        }