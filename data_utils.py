#!/usr/bin/env python3
"""
数据处理工具

用于SDXL inpainting训练的数据预处理和增强工具
"""

import os
import random
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json


class InpaintingDataset(Dataset):
    """
    图像修复数据集类
    
    支持多种数据格式：
    1. 图像+遮罩对
    2. 完整图像（自动生成遮罩）
    3. 带标注的图像修复数据
    """
    
    def __init__(
        self,
        data_root: str,
        resolution: int = 1024,
        center_crop: bool = False,
        random_flip: bool = True,
        mask_type: str = "random",  # random, provided, mixed
        mask_ratio_range: Tuple[float, float] = (0.1, 0.3),
        caption_file: Optional[str] = None,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    ):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            resolution: 图像分辨率
            center_crop: 是否中心裁剪
            random_flip: 是否随机翻转
            mask_type: 遮罩类型 (random/provided/mixed)
            mask_ratio_range: 遮罩面积比例范围
            caption_file: 标题文件路径
            image_extensions: 支持的图像格式
        """
        self.data_root = data_root
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.mask_type = mask_type
        self.mask_ratio_range = mask_ratio_range
        self.image_extensions = image_extensions
        
        # 加载图像文件列表
        self.image_files = self._load_image_files()
        
        # 加载标题信息
        self.captions = self._load_captions(caption_file)
        
        # 设置图像变换
        self.image_transforms = self._setup_transforms()
        
        print(f"加载了 {len(self.image_files)} 张图像用于训练")
    
    def _load_image_files(self) -> List[str]:
        """加载图像文件列表"""
        image_files = []
        
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def _load_captions(self, caption_file: Optional[str]) -> Dict[str, str]:
        """加载图像标题"""
        captions = {}
        
        if caption_file and os.path.exists(caption_file):
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = json.load(f)
        
        # 如果没有标题文件，使用默认标题
        default_captions = [
            "高质量的图像修复",
            "填补图像中的缺失部分",
            "恢复图像的完整性",
            "修复损坏的图像区域",
            "图像修复和重建",
        ]
        
        for image_file in self.image_files:
            if image_file not in captions:
                captions[image_file] = random.choice(default_captions)
        
        return captions
    
    def _setup_transforms(self):
        """设置图像变换"""
        transforms_list = []
        
        # 调整大小
        transforms_list.append(transforms.Resize(
            self.resolution, 
            interpolation=transforms.InterpolationMode.BILINEAR
        ))
        
        # 裁剪
        if self.center_crop:
            transforms_list.append(transforms.CenterCrop(self.resolution))
        else:
            transforms_list.append(transforms.RandomCrop(self.resolution))
        
        # 随机翻转
        if self.random_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # 转换为张量并归一化
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return transforms.Compose(transforms_list)
    
    def _generate_random_mask(self, size: Tuple[int, int]) -> Image.Image:
        """生成随机遮罩"""
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 计算遮罩面积
        total_area = size[0] * size[1]
        mask_ratio = random.uniform(*self.mask_ratio_range)
        target_area = total_area * mask_ratio
        
        # 生成多个随机形状
        shapes_count = random.randint(1, 5)
        current_area = 0
        
        for _ in range(shapes_count):
            if current_area >= target_area:
                break
            
            # 随机选择形状类型
            shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
            
            if shape_type == 'rectangle':
                # 矩形
                w = random.randint(20, size[0] // 4)
                h = random.randint(20, size[1] // 4)
                x = random.randint(0, size[0] - w)
                y = random.randint(0, size[1] - h)
                draw.rectangle([x, y, x + w, y + h], fill=255)
                current_area += w * h
                
            elif shape_type == 'ellipse':
                # 椭圆
                w = random.randint(20, size[0] // 4)
                h = random.randint(20, size[1] // 4)
                x = random.randint(0, size[0] - w)
                y = random.randint(0, size[1] - h)
                draw.ellipse([x, y, x + w, y + h], fill=255)
                current_area += (w * h * 3.14159) / 4
                
            else:
                # 多边形
                points_count = random.randint(3, 8)
                center_x = random.randint(50, size[0] - 50)
                center_y = random.randint(50, size[1] - 50)
                radius = random.randint(20, min(size[0], size[1]) // 8)
                
                points = []
                for i in range(points_count):
                    angle = (2 * 3.14159 * i) / points_count
                    r = radius * random.uniform(0.5, 1.5)
                    x = center_x + r * np.cos(angle)
                    y = center_y + r * np.sin(angle)
                    points.append((x, y))
                
                draw.polygon(points, fill=255)
                current_area += radius * radius * 3.14159
        
        # 应用高斯模糊使边缘更自然
        if random.random() < 0.3:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
        
        return mask
    
    def _load_provided_mask(self, image_path: str) -> Optional[Image.Image]:
        """加载提供的遮罩文件"""
        # 尝试不同的遮罩文件命名方式
        base_name = os.path.splitext(image_path)[0]
        mask_extensions = ['_mask.png', '_mask.jpg', '.mask.png', '.mask.jpg']
        
        for ext in mask_extensions:
            mask_path = base_name + ext
            if os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    return mask
                except Exception as e:
                    print(f"加载遮罩失败 {mask_path}: {e}")
                    continue
        
        return None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """获取数据项"""
        image_path = self.image_files[idx]
        
        # 加载原始图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            # 返回一个随机索引的图像
            return self.__getitem__(random.randint(0, len(self.image_files) - 1))
        
        original_size = image.size
        
        # 获取遮罩
        mask = None
        if self.mask_type == "provided":
            mask = self._load_provided_mask(image_path)
        elif self.mask_type == "mixed":
            if random.random() < 0.5:
                mask = self._load_provided_mask(image_path)
        
        # 如果没有提供的遮罩，生成随机遮罩
        if mask is None:
            mask = self._generate_random_mask(image.size)
        else:
            # 确保遮罩尺寸匹配
            mask = mask.resize(image.size, Image.LANCZOS)
        
        # 创建被遮罩的图像
        masked_image = image.copy()
        masked_image.paste((128, 128, 128), mask=mask)  # 用灰色填充遮罩区域
        
        # 应用变换
        # 注意：需要对图像和遮罩应用相同的变换
        seed = random.randint(0, 2**32 - 1)
        
        # 变换原始图像
        random.seed(seed)
        torch.manual_seed(seed)
        image_tensor = self.image_transforms(image)
        
        # 变换被遮罩的图像
        random.seed(seed)
        torch.manual_seed(seed)
        masked_image_tensor = self.image_transforms(masked_image)
        
        # 变换遮罩（不需要归一化）
        mask_transforms = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            transforms.RandomHorizontalFlip(p=0.5) if self.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        
        random.seed(seed)
        torch.manual_seed(seed)
        mask_tensor = mask_transforms(mask)
        
        # 获取标题
        caption = self.captions.get(image_path, "图像修复")
        
        return {
            'pixel_values': image_tensor,
            'masked_images': masked_image_tensor,
            'masks': mask_tensor,
            'caption': caption,
            'original_size': original_size,
            'image_path': image_path,
        }


class MaskGenerator:
    """遮罩生成器"""
    
    @staticmethod
    def generate_brush_mask(size: Tuple[int, int], brush_width_range: Tuple[int, int] = (10, 50)) -> Image.Image:
        """生成画笔风格的遮罩"""
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 生成随机路径
        num_strokes = random.randint(1, 5)
        
        for _ in range(num_strokes):
            # 随机起点
            start_x = random.randint(0, size[0])
            start_y = random.randint(0, size[1])
            
            # 生成路径点
            points = [(start_x, start_y)]
            current_x, current_y = start_x, start_y
            
            num_points = random.randint(5, 20)
            for _ in range(num_points):
                # 随机移动
                dx = random.randint(-50, 50)
                dy = random.randint(-50, 50)
                current_x = max(0, min(size[0], current_x + dx))
                current_y = max(0, min(size[1], current_y + dy))
                points.append((current_x, current_y))
            
            # 绘制路径
            brush_width = random.randint(*brush_width_range)
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=255, width=brush_width)
        
        return mask
    
    @staticmethod
    def generate_object_mask(size: Tuple[int, int]) -> Image.Image:
        """生成类似物体形状的遮罩"""
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 生成不规则形状
        center_x = size[0] // 2 + random.randint(-size[0]//4, size[0]//4)
        center_y = size[1] // 2 + random.randint(-size[1]//4, size[1]//4)
        
        # 生成多边形点
        num_points = random.randint(6, 12)
        points = []
        
        base_radius = min(size[0], size[1]) // 4
        for i in range(num_points):
            angle = (2 * 3.14159 * i) / num_points
            radius = base_radius * random.uniform(0.5, 1.5)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            x = max(0, min(size[0], x))
            y = max(0, min(size[1], y))
            points.append((x, y))
        
        draw.polygon(points, fill=255)
        
        # 添加一些小的洞
        num_holes = random.randint(0, 3)
        for _ in range(num_holes):
            hole_x = random.randint(0, size[0])
            hole_y = random.randint(0, size[1])
            hole_size = random.randint(10, 30)
            draw.ellipse([
                hole_x - hole_size, hole_y - hole_size,
                hole_x + hole_size, hole_y + hole_size
            ], fill=0)
        
        return mask
    
    @staticmethod
    def generate_edge_mask(size: Tuple[int, int]) -> Image.Image:
        """生成边缘遮罩"""
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 随机选择边缘
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        mask_size = random.randint(size[0] // 8, size[0] // 4)
        
        if edge == 'top':
            draw.rectangle([0, 0, size[0], mask_size], fill=255)
        elif edge == 'bottom':
            draw.rectangle([0, size[1] - mask_size, size[0], size[1]], fill=255)
        elif edge == 'left':
            draw.rectangle([0, 0, mask_size, size[1]], fill=255)
        else:  # right
            draw.rectangle([size[0] - mask_size, 0, size[0], size[1]], fill=255)
        
        return mask


def create_caption_file(data_root: str, output_file: str, default_captions: Optional[List[str]] = None):
    """
    创建标题文件
    
    Args:
        data_root: 数据根目录
        output_file: 输出的标题文件路径
        default_captions: 默认标题列表
    """
    if default_captions is None:
        default_captions = [
            "高质量的图像修复",
            "填补图像中的缺失部分", 
            "恢复图像的完整性",
            "修复损坏的图像区域",
            "图像修复和重建",
            "完善图像细节",
            "修复图像瑕疵",
            "图像内容补全",
        ]
    
    captions = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                captions[image_path] = random.choice(default_captions)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    
    print(f"创建标题文件: {output_file}，包含 {len(captions)} 个条目")


def prepare_dataset_structure(data_root: str):
    """
    准备数据集目录结构
    
    Args:
        data_root: 数据根目录
    """
    # 创建必要的目录
    os.makedirs(os.path.join(data_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'masks'), exist_ok=True)
    
    # 创建示例说明文件
    readme_content = """
# 数据集目录结构

## 目录说明
- `images/`: 存放训练图像
- `masks/`: 存放对应的遮罩图像（可选）
- `captions.json`: 图像标题文件（可选）

## 数据格式要求

### 图像文件
- 支持格式: JPG, JPEG, PNG, BMP, TIFF
- 建议分辨率: 1024x1024 或更高
- 图像质量: 清晰，无明显噪声

### 遮罩文件（可选）
- 格式: PNG（推荐）或JPG
- 命名规则: 
  - `image_name_mask.png`
  - `image_name.mask.png`
- 遮罩说明:
  - 白色区域 (255): 需要修复的区域
  - 黑色区域 (0): 保持不变的区域

### 标题文件（可选）
- 格式: JSON
- 结构: {"图像路径": "描述文本"}
- 如果不提供，将使用默认标题

## 使用示例

```python
from data_utils import InpaintingDataset

# 创建数据集
dataset = InpaintingDataset(
    data_root="path/to/your/data",
    resolution=1024,
    mask_type="random",  # 或 "provided", "mixed"
    caption_file="captions.json"
)
```
"""
    
    with open(os.path.join(data_root, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"数据集目录结构已准备完成: {data_root}")


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="数据处理工具")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--action", type=str, choices=['prepare', 'create_captions'], 
                       default='prepare', help="执行的操作")
    parser.add_argument("--caption_file", type=str, default="captions.json", 
                       help="标题文件名")
    
    args = parser.parse_args()
    
    if args.action == 'prepare':
        prepare_dataset_structure(args.data_root)
    elif args.action == 'create_captions':
        create_caption_file(args.data_root, 
                          os.path.join(args.data_root, args.caption_file))