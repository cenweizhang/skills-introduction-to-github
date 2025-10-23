#!/usr/bin/env python3
"""
SDXL Inpainting 推理示例

展示如何使用微调后的模型进行图像修复推理
"""

import os
import argparse
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionXLInpaintPipeline
import numpy as np


def create_sample_mask(image_size, mask_type="center"):
    """创建示例遮罩"""
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "center":
        # 中心矩形遮罩
        w, h = image_size
        mask_w, mask_h = w // 3, h // 3
        x1 = (w - mask_w) // 2
        y1 = (h - mask_h) // 2
        draw.rectangle([x1, y1, x1 + mask_w, y1 + mask_h], fill=255)
    
    elif mask_type == "random":
        # 随机形状遮罩
        w, h = image_size
        num_shapes = np.random.randint(2, 6)
        
        for _ in range(num_shapes):
            shape_type = np.random.choice(['rectangle', 'ellipse'])
            
            if shape_type == 'rectangle':
                x1 = np.random.randint(0, w // 2)
                y1 = np.random.randint(0, h // 2)
                x2 = np.random.randint(x1 + 50, min(w, x1 + w // 3))
                y2 = np.random.randint(y1 + 50, min(h, y1 + h // 3))
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                x1 = np.random.randint(0, w // 2)
                y1 = np.random.randint(0, h // 2)
                x2 = np.random.randint(x1 + 50, min(w, x1 + w // 3))
                y2 = np.random.randint(y1 + 50, min(h, y1 + h // 3))
                draw.ellipse([x1, y1, x2, y2], fill=255)
    
    elif mask_type == "edge":
        # 边缘遮罩
        w, h = image_size
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        mask_size = min(w, h) // 4
        
        if edge == 'top':
            draw.rectangle([0, 0, w, mask_size], fill=255)
        elif edge == 'bottom':
            draw.rectangle([0, h - mask_size, w, h], fill=255)
        elif edge == 'left':
            draw.rectangle([0, 0, mask_size, h], fill=255)
        else:  # right
            draw.rectangle([w - mask_size, 0, w, h], fill=255)
    
    return mask


def load_pipeline(model_path, device="cuda", dtype=torch.float16):
    """加载推理管道"""
    print(f"正在加载模型: {model_path}")
    
    try:
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None
        )
        pipeline.to(device)
        
        # 启用内存优化
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ xFormers内存优化已启用")
        except:
            print("⚠️  xFormers不可用，使用默认注意力机制")
        
        # 启用CPU卸载以节省显存
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
            print("✅ CPU卸载已启用")
        
        print("✅ 模型加载完成")
        return pipeline
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def inpaint_image(pipeline, image, mask, prompt, negative_prompt="", **kwargs):
    """执行图像修复"""
    default_params = {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "strength": 0.8,
        "num_images_per_prompt": 1,
    }
    default_params.update(kwargs)
    
    print(f"开始推理...")
    print(f"提示词: {prompt}")
    print(f"推理步数: {default_params['num_inference_steps']}")
    print(f"引导强度: {default_params['guidance_scale']}")
    
    try:
        with torch.autocast("cuda"):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                **default_params
            )
        
        return result.images[0]
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return None


def create_comparison_image(original, mask, result):
    """创建对比图像"""
    # 创建被遮罩的图像用于对比
    masked_image = original.copy()
    mask_array = np.array(mask)
    original_array = np.array(original)
    
    # 将遮罩区域设为灰色
    masked_array = original_array.copy()
    masked_array[mask_array > 128] = [128, 128, 128]
    masked_image = Image.fromarray(masked_array)
    
    # 拼接图像：原图 | 遮罩图 | 修复结果
    width, height = original.size
    comparison = Image.new('RGB', (width * 3, height))
    
    comparison.paste(original, (0, 0))
    comparison.paste(masked_image, (width, 0))
    comparison.paste(result, (width * 2, 0))
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="SDXL Inpainting 推理示例")
    parser.add_argument("--model", type=str, required=True, help="微调后的模型路径")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--mask", type=str, help="遮罩图像路径")
    parser.add_argument("--prompt", type=str, default="高质量的图像修复", help="修复提示词")
    parser.add_argument("--negative_prompt", type=str, default="低质量, 模糊, 噪声", help="负面提示词")
    parser.add_argument("--output", type=str, default="output.jpg", help="输出图像路径")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance", type=float, default=7.5, help="引导强度")
    parser.add_argument("--strength", type=float, default=0.8, help="修复强度")
    parser.add_argument("--device", type=str, default="cuda", help="设备类型")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="数据类型")
    parser.add_argument("--create_mask", type=str, choices=["center", "random", "edge"], help="自动创建遮罩类型")
    parser.add_argument("--comparison", action="store_true", help="创建对比图像")
    
    args = parser.parse_args()
    
    # 设置数据类型
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    # 加载模型
    pipeline = load_pipeline(args.model, args.device, dtype)
    if pipeline is None:
        return
    
    # 加载或创建图像
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 图像文件不存在: {args.image}")
            return
        image = Image.open(args.image).convert('RGB')
        print(f"✅ 图像加载成功: {args.image}")
    else:
        # 创建示例图像
        image = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
        print("✅ 使用默认白色图像")
    
    # 加载或创建遮罩
    if args.mask:
        if not os.path.exists(args.mask):
            print(f"❌ 遮罩文件不存在: {args.mask}")
            return
        mask = Image.open(args.mask).convert('L')
        print(f"✅ 遮罩加载成功: {args.mask}")
    elif args.create_mask:
        mask = create_sample_mask(image.size, args.create_mask)
        print(f"✅ 自动创建遮罩: {args.create_mask}")
    else:
        # 创建默认中心遮罩
        mask = create_sample_mask(image.size, "center")
        print("✅ 使用默认中心遮罩")
    
    # 确保图像和遮罩尺寸匹配
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.LANCZOS)
        print("✅ 遮罩尺寸已调整")
    
    # 执行推理
    result = inpaint_image(
        pipeline=pipeline,
        image=image,
        mask=mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        strength=args.strength,
    )
    
    if result is None:
        return
    
    # 保存结果
    if args.comparison:
        # 创建对比图像
        comparison = create_comparison_image(image, mask, result)
        comparison.save(args.output)
        print(f"✅ 对比图像已保存: {args.output}")
    else:
        # 仅保存修复结果
        result.save(args.output)
        print(f"✅ 修复结果已保存: {args.output}")
    
    print("🎉 推理完成!")


if __name__ == "__main__":
    main()