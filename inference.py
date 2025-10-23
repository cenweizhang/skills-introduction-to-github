#!/usr/bin/env python3
"""
SDXL Inpainting 推理脚本
使用微调后的模型进行图像修复
"""

import argparse
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline


def main():
    parser = argparse.ArgumentParser(description="使用SDXL Inpainting模型进行推理")
    
    # 必需参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="微调后的模型路径"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图像路径"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Mask图像路径（白色=需要修复的区域）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文本提示词"
    )
    
    # 可选参数
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="负面提示词"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.jpg",
        help="输出图像路径"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数（越多越慢但质量可能更好）"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="引导比例（控制提示词的影响强度）"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="修复强度（0.0-1.0，越高修改越大）"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="生成图像数量"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (cuda 或 cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="数据类型"
    )
    
    args = parser.parse_args()
    
    # 设置数据类型
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    print(f"加载模型: {args.model_path}")
    
    # 加载pipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    pipe = pipe.to(args.device)
    
    # 优化（如果是CUDA）
    if args.device == "cuda":
        # 启用注意力切片以节省显存
        pipe.enable_attention_slicing()
        
        # 如果安装了xformers，启用它
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ 已启用xformers加速")
        except:
            print("⚠ xformers未安装，使用默认注意力机制")
    
    print(f"加载图像: {args.image}")
    print(f"加载mask: {args.mask}")
    
    # 加载图像和mask
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    
    # 确保mask和图像大小一致
    if mask.size != image.size:
        print(f"调整mask大小从 {mask.size} 到 {image.size}")
        mask = mask.resize(image.size, Image.LANCZOS)
    
    # 设置随机种子
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    print(f"\n开始生成...")
    print(f"提示词: {args.prompt}")
    if args.negative_prompt:
        print(f"负面提示词: {args.negative_prompt}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"引导比例: {args.guidance_scale}")
    
    # 生成
    results = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        image=image,
        mask_image=mask,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        num_images_per_prompt=args.num_images,
        generator=generator,
    ).images
    
    # 保存结果
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(results) == 1:
        results[0].save(args.output)
        print(f"\n✓ 结果已保存到: {args.output}")
    else:
        for idx, result in enumerate(results):
            output_file = output_dir / f"{output_path.stem}_{idx}{output_path.suffix}"
            result.save(output_file)
            print(f"✓ 结果 {idx+1} 已保存到: {output_file}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
