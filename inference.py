#!/usr/bin/env python3
"""
SDXL图像修复模型推理脚本
支持微调后的模型进行图像修复
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple
import warnings

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from diffusers import StableDiffusionXLInpaintPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class SDXLInpaintInference:
    """SDXL图像修复推理类"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        use_safetensors: bool = True,
    ):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径（可以是Hugging Face模型ID或本地路径）
            device: 设备类型
            torch_dtype: 数据类型
            use_safetensors: 是否使用safetensors格式
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.torch_dtype = torch_dtype
        
        # 加载模型
        self._load_pipeline()
        
        logger.info(f"模型已加载到设备: {self.device}")
        
    def _get_device(self, device: str) -> str:
        """获取设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_pipeline(self):
        """加载推理管道"""
        try:
            # 尝试加载微调后的模型
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None,
            )
        except Exception as e:
            logger.warning(f"无法加载微调模型: {e}")
            logger.info("尝试加载原始模型...")
            
            # 回退到原始模型
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None,
            )
        
        # 移动到设备
        self.pipeline = self.pipeline.to(self.device)
        
        # 优化内存使用
        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_model_cpu_offload()
        
        # 设置调度器
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
    def inpaint(
        self,
        image: Union[str, Image.Image, np.ndarray],
        mask: Union[str, Image.Image, np.ndarray],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 1.0,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        """
        执行图像修复
        
        Args:
            image: 输入图像
            mask: 掩码图像
            prompt: 文本提示
            negative_prompt: 负面提示
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            strength: 修复强度
            seed: 随机种子
            width: 输出宽度
            height: 输出高度
            
        Returns:
            修复后的图像
        """
        # 预处理输入
        image = self._preprocess_image(image, width, height)
        mask = self._preprocess_mask(mask, width, height)
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 执行推理
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                width=width,
                height=height,
            ).images[0]
        
        return result
    
    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray], width: int, height: int) -> Image.Image:
        """预处理输入图像"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("不支持的图像类型")
        
        # 调整尺寸
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        return image
    
    def _preprocess_mask(self, mask: Union[str, Image.Image, np.ndarray], width: int, height: int) -> Image.Image:
        """预处理掩码图像"""
        if isinstance(mask, str):
            mask = Image.open(mask).convert("L")
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask).convert("L")
        elif not isinstance(mask, Image.Image):
            raise ValueError("不支持的掩码类型")
        
        # 调整尺寸
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        
        return mask
    
    def batch_inpaint(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        prompts: List[str],
        output_dir: str,
        **kwargs
    ) -> List[Image.Image]:
        """
        批量图像修复
        
        Args:
            image_paths: 图像路径列表
            mask_paths: 掩码路径列表
            prompts: 提示列表
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            修复后的图像列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, (img_path, mask_path, prompt) in enumerate(zip(image_paths, mask_paths, prompts)):
            logger.info(f"处理图像 {i+1}/{len(image_paths)}: {img_path}")
            
            try:
                # 执行修复
                result = self.inpaint(img_path, mask_path, prompt, **kwargs)
                
                # 保存结果
                output_path = output_dir / f"result_{i:04d}.png"
                result.save(output_path)
                
                results.append(result)
                logger.info(f"结果已保存: {output_path}")
                
            except Exception as e:
                logger.error(f"处理图像失败 {img_path}: {e}")
                results.append(None)
        
        return results
    
    def create_mask_interactive(self, image: Image.Image, mask_type: str = "rectangle") -> Image.Image:
        """
        交互式创建掩码
        
        Args:
            image: 输入图像
            mask_type: 掩码类型 ("rectangle", "circle", "brush")
            
        Returns:
            掩码图像
        """
        # 创建掩码图像
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        if mask_type == "rectangle":
            # 矩形掩码
            width, height = image.size
            x1, y1 = width // 4, height // 4
            x2, y2 = 3 * width // 4, 3 * height // 4
            draw.rectangle([x1, y1, x2, y2], fill=255)
            
        elif mask_type == "circle":
            # 圆形掩码
            width, height = image.size
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], fill=255)
        
        return mask
    
    def visualize_results(
        self,
        original: Image.Image,
        mask: Image.Image,
        result: Image.Image,
        save_path: Optional[str] = None
    ) -> None:
        """
        可视化结果
        
        Args:
            original: 原始图像
            mask: 掩码图像
            result: 修复结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(original)
        axes[0].set_title("原始图像")
        axes[0].axis("off")
        
        # 掩码
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("掩码")
        axes[1].axis("off")
        
        # 修复结果
        axes[2].imshow(result)
        axes[2].set_title("修复结果")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"可视化结果已保存: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SDXL图像修复推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--mask", type=str, help="掩码图像路径")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示")
    parser.add_argument("--negative_prompt", type=str, default="", help="负面提示")
    parser.add_argument("--output", type=str, default="output.png", help="输出路径")
    parser.add_argument("--width", type=int, default=1024, help="输出宽度")
    parser.add_argument("--height", type=int, default=1024, help="输出高度")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导强度")
    parser.add_argument("--strength", type=float, default=1.0, help="修复强度")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--device", type=str, default="auto", help="设备类型")
    parser.add_argument("--create_mask", action="store_true", help="交互式创建掩码")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = SDXLInpaintInference(
        model_path=args.model_path,
        device=args.device,
    )
    
    # 加载图像
    image = Image.open(args.image).convert("RGB")
    
    # 创建或加载掩码
    if args.create_mask:
        mask = inference.create_mask_interactive(image)
    elif args.mask:
        mask = Image.open(args.mask).convert("L")
    else:
        # 创建默认掩码
        mask = inference.create_mask_interactive(image)
    
    # 执行修复
    result = inference.inpaint(
        image=image,
        mask=mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    
    # 保存结果
    result.save(args.output)
    logger.info(f"修复结果已保存: {args.output}")
    
    # 可视化结果
    inference.visualize_results(image, mask, result, f"{args.output}_comparison.png")


if __name__ == "__main__":
    main()