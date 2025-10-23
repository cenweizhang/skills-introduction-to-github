#!/usr/bin/env python3
"""
数据准备脚本
帮助您将原始数据整理成训练所需的格式

支持的输入格式：
1. 自动生成mask（从完整图像生成训练对）
2. 已有的图像-mask对
"""

import argparse
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm


def create_random_mask(image_size, min_size=50, max_size=300, num_shapes=3):
    """
    创建随机mask
    
    Args:
        image_size: (width, height)
        min_size: 最小形状大小
        max_size: 最大形状大小  
        num_shapes: 形状数量
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'ellipse'])
        
        # 随机位置和大小
        x1 = random.randint(0, image_size[0] - max_size)
        y1 = random.randint(0, image_size[1] - max_size)
        size = random.randint(min_size, max_size)
        x2 = min(x1 + size, image_size[0])
        y2 = min(y1 + size, image_size[1])
        
        if shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=255)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=255)
    
    return mask


def create_brush_stroke_mask(image_size, num_strokes=5):
    """创建笔刷风格的mask（更自然）"""
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for _ in range(num_strokes):
        # 生成随机路径点
        num_points = random.randint(3, 8)
        points = []
        start_x = random.randint(0, image_size[0])
        start_y = random.randint(0, image_size[1])
        
        for i in range(num_points):
            if i == 0:
                points.append((start_x, start_y))
            else:
                # 在上一个点附近生成新点
                prev_x, prev_y = points[-1]
                new_x = prev_x + random.randint(-100, 100)
                new_y = prev_y + random.randint(-100, 100)
                new_x = max(0, min(new_x, image_size[0]))
                new_y = max(0, min(new_y, image_size[1]))
                points.append((new_x, new_y))
        
        # 绘制线条
        width = random.randint(20, 60)
        if len(points) > 1:
            draw.line(points, fill=255, width=width, joint='curve')
    
    return mask


def prepare_dataset_auto(input_dir, output_dir, mask_type='random', num_variants=1):
    """
    自动准备数据集（生成mask）
    
    Args:
        input_dir: 包含原始图像的目录
        output_dir: 输出目录
        mask_type: mask类型 ('random' 或 'brush')
        num_variants: 每张图像生成多少个变体
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + \
                  list(input_path.glob("*.png")) + list(input_path.glob("*.bmp"))
    
    print(f"找到 {len(image_files)} 张图像")
    
    prompts = []
    
    for img_file in tqdm(image_files, desc="处理图像"):
        try:
            image = Image.open(img_file).convert('RGB')
            
            for variant in range(num_variants):
                # 生成唯一的文件名
                if num_variants > 1:
                    output_name = f"{img_file.stem}_v{variant}{img_file.suffix}"
                else:
                    output_name = img_file.name
                
                # 保存原始图像
                image.save(images_dir / output_name)
                
                # 生成mask
                if mask_type == 'brush':
                    mask = create_brush_stroke_mask(image.size)
                else:
                    mask = create_random_mask(image.size)
                
                mask.save(masks_dir / output_name)
                
                # 添加默认提示词（您需要手动修改）
                prompts.append(f"A photo of {img_file.stem}")
        
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")
            continue
    
    # 保存prompts
    with open(output_path / "prompts.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(prompts))
    
    print(f"\n完成！处理了 {len(prompts)} 个样本")
    print(f"输出目录: {output_path}")
    print(f"\n请手动编辑 {output_path / 'prompts.txt'} 以添加合适的描述文本")


def prepare_dataset_from_pairs(image_dir, mask_dir, output_dir, prompts_file=None):
    """
    从已有的图像-mask对准备数据集
    
    Args:
        image_dir: 图像目录
        mask_dir: mask目录
        output_dir: 输出目录
        prompts_file: 可选的prompts文件路径
    """
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg")) + \
                  list(image_path.glob("*.png")) + list(image_path.glob("*.bmp"))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 加载prompts（如果提供）
    prompts_dict = {}
    if prompts_file and Path(prompts_file).exists():
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_list = [line.strip() for line in f.readlines()]
            # 假设prompts的顺序与文件名排序后的顺序一致
            for idx, img_file in enumerate(sorted(image_files)):
                if idx < len(prompts_list):
                    prompts_dict[img_file.name] = prompts_list[idx]
    
    valid_count = 0
    prompts = []
    
    for img_file in tqdm(image_files, desc="复制文件"):
        mask_file = mask_path / img_file.name
        
        if not mask_file.exists():
            print(f"警告: 未找到对应的mask文件 {mask_file}")
            continue
        
        try:
            # 验证图像可以打开
            image = Image.open(img_file)
            mask = Image.open(mask_file)
            
            # 复制到输出目录
            image.save(images_dir / img_file.name)
            mask.save(masks_dir / img_file.name)
            
            # 添加prompt
            prompt = prompts_dict.get(img_file.name, f"A photo of {img_file.stem}")
            prompts.append(prompt)
            
            valid_count += 1
        
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")
            continue
    
    # 保存prompts
    with open(output_path / "prompts.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(prompts))
    
    print(f"\n完成！处理了 {valid_count} 个有效样本")
    print(f"输出目录: {output_path}")


def validate_dataset(data_dir):
    """验证数据集格式是否正确"""
    data_path = Path(data_dir)
    
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    prompts_file = data_path / "prompts.txt"
    
    if not images_dir.exists():
        print(f"❌ 错误: 未找到images目录")
        return False
    
    if not masks_dir.exists():
        print(f"❌ 错误: 未找到masks目录")
        return False
    
    if not prompts_file.exists():
        print(f"❌ 错误: 未找到prompts.txt文件")
        return False
    
    # 检查图像和mask数量
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + \
                  list(images_dir.glob("*.png")) + list(images_dir.glob("*.bmp"))
    
    mask_files = list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.jpeg")) + \
                 list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.bmp"))
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines()]
    
    print(f"✓ 图像数量: {len(image_files)}")
    print(f"✓ Mask数量: {len(mask_files)}")
    print(f"✓ 提示词数量: {len(prompts)}")
    
    if len(image_files) == 0:
        print(f"❌ 错误: 没有找到图像文件")
        return False
    
    if len(image_files) != len(mask_files):
        print(f"⚠️  警告: 图像和mask数量不匹配")
    
    if len(image_files) != len(prompts):
        print(f"⚠️  警告: 图像和提示词数量不匹配")
    
    # 检查每个图像是否有对应的mask
    missing_masks = []
    for img_file in image_files:
        mask_file = masks_dir / img_file.name
        if not mask_file.exists():
            missing_masks.append(img_file.name)
    
    if missing_masks:
        print(f"⚠️  警告: {len(missing_masks)} 个图像缺少对应的mask")
        if len(missing_masks) <= 5:
            for name in missing_masks:
                print(f"   - {name}")
    
    print(f"\n✓ 数据集格式验证完成")
    return True


def main():
    parser = argparse.ArgumentParser(description="准备SDXL Inpainting训练数据集")
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 自动生成模式
    auto_parser = subparsers.add_parser('auto', help='自动生成mask')
    auto_parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    auto_parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    auto_parser.add_argument('--mask_type', type=str, default='random', choices=['random', 'brush'],
                            help='Mask类型')
    auto_parser.add_argument('--num_variants', type=int, default=1,
                            help='每张图像生成多少个变体')
    
    # 从已有对准备
    pair_parser = subparsers.add_parser('from-pairs', help='从已有的图像-mask对准备')
    pair_parser.add_argument('--image_dir', type=str, required=True, help='图像目录')
    pair_parser.add_argument('--mask_dir', type=str, required=True, help='Mask目录')
    pair_parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    pair_parser.add_argument('--prompts_file', type=str, default=None, help='Prompts文件（可选）')
    
    # 验证数据集
    validate_parser = subparsers.add_parser('validate', help='验证数据集格式')
    validate_parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    
    args = parser.parse_args()
    
    if args.command == 'auto':
        prepare_dataset_auto(
            args.input_dir,
            args.output_dir,
            args.mask_type,
            args.num_variants
        )
    elif args.command == 'from-pairs':
        prepare_dataset_from_pairs(
            args.image_dir,
            args.mask_dir,
            args.output_dir,
            args.prompts_file
        )
    elif args.command == 'validate':
        validate_dataset(args.data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
