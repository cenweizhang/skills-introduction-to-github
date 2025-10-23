#!/usr/bin/env python3
"""
SDXL图像修复模型使用示例
演示如何使用微调后的模型进行图像修复
"""

import os
import torch
from PIL import Image, ImageDraw
import numpy as np
from inference import SDXLInpaintInference
from config import TrainingConfig


def create_sample_data():
    """创建示例数据"""
    print("创建示例数据...")
    
    # 创建数据目录
    os.makedirs("data/train", exist_ok=True)
    
    # 创建示例图像
    image = Image.new("RGB", (1024, 1024), color="lightblue")
    draw = ImageDraw.Draw(image)
    
    # 绘制一些内容
    draw.rectangle([100, 100, 500, 400], fill="red", outline="black", width=3)
    draw.ellipse([600, 200, 900, 500], fill="yellow", outline="black", width=3)
    draw.text((200, 600), "Hello World!", fill="black")
    
    # 保存图像
    image.save("data/train/sample_image.jpg")
    
    # 创建掩码
    mask = Image.new("L", (1024, 1024), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([100, 100, 500, 400], fill=255)  # 红色区域需要修复
    
    # 保存掩码
    mask.save("data/train/sample_mask.png")
    
    # 创建描述文件
    with open("data/train/sample_caption.txt", "w", encoding="utf-8") as f:
        f.write("a colorful image with geometric shapes and text")
    
    print("示例数据创建完成!")


def example_training():
    """训练示例"""
    print("开始训练示例...")
    
    # 创建示例数据
    create_sample_data()
    
    # 创建配置
    config = TrainingConfig(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        train_data_dir="./data/train",
        output_dir="./sdxl-inpaint-finetuned",
        resolution=512,  # 使用较小分辨率进行快速训练
        train_batch_size=1,
        max_train_steps=100,  # 少量步数用于演示
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=50,
    )
    
    # 保存配置
    config.to_yaml("example_config.yaml")
    
    print("配置已保存到 example_config.yaml")
    print("运行以下命令开始训练:")
    print("python train_sdxl_inpaint.py --config example_config.yaml")


def example_inference():
    """推理示例"""
    print("推理示例...")
    
    # 检查模型是否存在
    model_path = "./sdxl-inpaint-finetuned"
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        print("请先运行训练示例或使用预训练模型")
        return
    
    try:
        # 创建推理器
        inference = SDXLInpaintInference(model_path)
        
        # 加载示例图像
        image = Image.open("data/train/sample_image.jpg")
        mask = Image.open("data/train/sample_mask.png")
        
        # 执行修复
        result = inference.inpaint(
            image=image,
            mask=mask,
            prompt="a beautiful landscape with mountains and trees",
            num_inference_steps=20,  # 较少步数用于快速演示
            guidance_scale=7.5,
        )
        
        # 保存结果
        result.save("example_result.png")
        print("修复结果已保存到 example_result.png")
        
        # 可视化结果
        inference.visualize_results(
            image, mask, result, "example_comparison.png"
        )
        print("对比图已保存到 example_comparison.png")
        
    except Exception as e:
        print(f"推理失败: {e}")


def example_batch_processing():
    """批量处理示例"""
    print("批量处理示例...")
    
    # 创建多个示例图像
    for i in range(3):
        # 创建图像
        image = Image.new("RGB", (512, 512), color=f"hsl({i*120}, 50%, 50%)")
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 200, 200], fill="white")
        
        # 创建掩码
        mask = Image.new("L", (512, 512), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([50, 50, 200, 200], fill=255)
        
        # 保存文件
        image.save(f"data/train/batch_image_{i}.jpg")
        mask.save(f"data/train/batch_mask_{i}.png")
    
    # 批量处理
    try:
        inference = SDXLInpaintInference("./sdxl-inpaint-finetuned")
        
        results = inference.batch_inpaint(
            image_paths=[
                "data/train/batch_image_0.jpg",
                "data/train/batch_image_1.jpg",
                "data/train/batch_image_2.jpg",
            ],
            mask_paths=[
                "data/train/batch_mask_0.png",
                "data/train/batch_mask_1.png",
                "data/train/batch_mask_2.png",
            ],
            prompts=[
                "a beautiful sunset landscape",
                "a modern city skyline",
                "a peaceful forest scene",
            ],
            output_dir="./batch_results",
            num_inference_steps=20,
        )
        
        print(f"批量处理完成，结果保存在 ./batch_results/")
        
    except Exception as e:
        print(f"批量处理失败: {e}")


def example_interactive_mask():
    """交互式掩码创建示例"""
    print("交互式掩码创建示例...")
    
    try:
        # 创建推理器
        inference = SDXLInpaintInference("./sdxl-inpaint-finetuned")
        
        # 加载图像
        image = Image.open("data/train/sample_image.jpg")
        
        # 创建不同类型的掩码
        mask_types = ["rectangle", "circle"]
        
        for mask_type in mask_types:
            mask = inference.create_mask_interactive(image, mask_type)
            
            # 执行修复
            result = inference.inpaint(
                image=image,
                mask=mask,
                prompt=f"a beautiful image with {mask_type} mask",
                num_inference_steps=20,
            )
            
            # 保存结果
            result.save(f"example_{mask_type}_result.png")
            print(f"{mask_type} 掩码结果已保存到 example_{mask_type}_result.png")
            
    except Exception as e:
        print(f"交互式掩码创建失败: {e}")


def main():
    """主函数"""
    print("SDXL图像修复模型使用示例")
    print("=" * 50)
    
    while True:
        print("\n请选择要运行的示例:")
        print("1. 创建示例数据")
        print("2. 训练示例")
        print("3. 推理示例")
        print("4. 批量处理示例")
        print("5. 交互式掩码创建示例")
        print("6. 运行所有示例")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-6): ").strip()
        
        if choice == "0":
            print("退出程序")
            break
        elif choice == "1":
            create_sample_data()
        elif choice == "2":
            example_training()
        elif choice == "3":
            example_inference()
        elif choice == "4":
            example_batch_processing()
        elif choice == "5":
            example_interactive_mask()
        elif choice == "6":
            print("运行所有示例...")
            create_sample_data()
            example_training()
            example_inference()
            example_batch_processing()
            example_interactive_mask()
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()