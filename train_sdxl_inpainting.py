#!/usr/bin/env python3
"""
SDXL Inpainting 模型微调训练脚本
支持自定义数据集训练，优化内存使用，支持多GPU训练
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available

# 检查版本
check_min_version("0.24.0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 SDXL Inpainting 模型")
    
    # 数据相关参数
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="训练数据目录，应包含images, masks, prompts子目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-inpainting-finetuned",
        help="模型保存目录"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="预训练模型缓存目录"
    )
    
    # 模型相关参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="预训练模型名称或路径"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="预训练模型的版本"
    )
    
    # 训练参数
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="输入图像分辨率（SDXL默认1024）"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="训练批次大小"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="最大训练步数（如果设置，将覆盖num_train_epochs）"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="是否使用梯度检查点以节省内存"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="初始学习率"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="是否根据批次大小缩放学习率"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="学习率调度器类型"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="学习率预热步数"
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam优化器的beta1参数"
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam优化器的beta2参数"
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam优化器的权重衰减"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Adam优化器的epsilon值"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="最大梯度范数"
    )
    
    # 数据增强
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="是否随机水平翻转"
    )
    
    # 日志和保存
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard日志目录"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="混合精度训练"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="上报到的平台（tensorboard, wandb等）"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="每隔多少步保存检查点"
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="最多保存的检查点数量"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    
    # 其他
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="数据加载器工作线程数"
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="是否使用8bit Adam优化器以节省内存"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="是否使用EMA（指数移动平均）"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="是否推送到Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub上的模型ID"
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hugging Face Hub token"
    )
    
    args = parser.parse_args()
    
    # 环境变量
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args


class InpaintingDataset(Dataset):
    """
    Inpainting训练数据集
    目录结构：
    data_dir/
        images/          # 原始图像
            img1.jpg
            img2.png
        masks/           # 对应的mask图像（白色表示需要修复的区域）
            img1.jpg
            img2.png
        prompts.txt      # 文本提示词，每行对应一张图像
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer_one,
        tokenizer_two,
        size: int = 1024,
        random_flip: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.size = size
        self.random_flip = random_flip
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        
        # 加载图像列表
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        
        if not self.image_dir.exists():
            raise ValueError(f"图像目录不存在: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask目录不存在: {self.mask_dir}")
        
        # 获取所有图像文件
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        # 加载提示词
        prompts_file = self.data_dir / "prompts.txt"
        if prompts_file.exists():
            with open(prompts_file, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f.readlines()]
            
            if len(self.prompts) != len(self.image_files):
                logger.warning(f"提示词数量({len(self.prompts)})与图像数量({len(self.image_files)})不匹配！")
                # 填充或截断
                if len(self.prompts) < len(self.image_files):
                    self.prompts.extend([""] * (len(self.image_files) - len(self.prompts)))
                else:
                    self.prompts = self.prompts[:len(self.image_files)]
        else:
            logger.warning(f"未找到prompts.txt，将使用空提示词")
            self.prompts = [""] * len(self.image_files)
        
        logger.info(f"加载了 {len(self.image_files)} 个训练样本")
        
        # 图像变换
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def tokenize_prompt(self, prompt):
        """使用两个tokenizer编码提示词（SDXL需要）"""
        tokens_one = self.tokenizer_one(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        
        tokens_two = self.tokenizer_two(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        
        return tokens_one, tokens_two
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = self.mask_dir / image_file.name
        prompt = self.prompts[idx]
        
        # 加载图像
        image = Image.open(image_file).convert("RGB")
        
        # 加载mask
        if mask_file.exists():
            mask = Image.open(mask_file).convert("L")  # 灰度图
        else:
            logger.warning(f"未找到mask文件: {mask_file}，使用空mask")
            mask = Image.new("L", image.size, 0)
        
        # 随机翻转
        if self.random_flip and torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # 应用变换
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)
        
        # Tokenize提示词
        tokens_one, tokens_two = self.tokenize_prompt(prompt)
        
        return {
            "pixel_values": image,
            "mask_values": mask,
            "input_ids_one": tokens_one,
            "input_ids_two": tokens_two,
        }


def collate_fn(examples):
    """数据批次整理函数"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    mask_values = torch.stack([example["mask_values"] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "mask_values": mask_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
    }


def main():
    args = parse_args()
    
    # 初始化accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    
    # 加载模型
    logger.info("加载预训练模型...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    text_encoder_one = diffusers.models.transformers.CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    
    text_encoder_two = diffusers.models.transformers.CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    
    # 冻结vae和text_encoder
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # 缩放学习率
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps 
            * args.train_batch_size * accelerator.num_processes
        )
    
    # 初始化优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("请安装bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = InpaintingDataset(
        data_dir=args.data_dir,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        size=args.resolution,
        random_flip=args.random_flip,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    # 计算训练步数
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # 使用accelerator准备
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # 移动模型到设备
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    
    # 创建EMA
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        ema_unet.to(accelerator.device)
    
    # 重新计算总训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # 训练信息
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  单设备批次大小 = {args.train_batch_size}")
    logger.info(f"  总批次大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # 从检查点恢复
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # 获取最新的检查点
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            logger.info(f"未找到检查点，从头开始训练")
            initial_global_step = 0
        else:
            logger.info(f"从检查点恢复训练: {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="训练步数",
        disable=not accelerator.is_local_main_process,
    )
    
    # 训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 将图像转换为latent空间
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 处理mask
                mask = batch["mask_values"]
                
                # 将mask下采样到latent空间的大小
                mask = F.interpolate(mask, size=(latents.shape[2], latents.shape[3]))
                
                # 创建masked图像的latent
                masked_image = batch["pixel_values"] * (1 - batch["mask_values"])
                masked_latents = vae.encode(masked_image.to(dtype=vae.dtype)).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                
                # 采样噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # 采样随机时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()
                
                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 获取文本嵌入
                encoder_hidden_states_one = text_encoder_one(
                    batch["input_ids_one"],
                    output_hidden_states=True,
                ).hidden_states[-2]
                
                encoder_hidden_states_two = text_encoder_two(
                    batch["input_ids_two"],
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = encoder_hidden_states_two[0]
                encoder_hidden_states_two = encoder_hidden_states_two.hidden_states[-2]
                
                # 连接文本嵌入
                encoder_hidden_states = torch.cat([encoder_hidden_states_one, encoder_hidden_states_two], dim=-1)
                
                # 准备added_cond_kwargs (SDXL特有)
                add_time_ids = torch.cat([
                    torch.tensor([args.resolution, args.resolution, 0, 0, args.resolution, args.resolution], device=latents.device).unsqueeze(0)
                ] * bsz)
                
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }
                
                # 将mask和masked_latents连接到noisy_latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
                
                # 预测噪声
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                
                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"未知的预测类型 {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新EMA
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # 保存检查点
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"保存检查点到 {save_path}")
                        
                        # 删除旧的检查点
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                
                                logger.info(f"删除 {len(removing_checkpoints)} 个旧检查点")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
            
            if global_step >= args.max_train_steps:
                break
    
    # 训练完成，保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        
        # 创建pipeline并保存
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)
        
        logger.info(f"模型已保存到 {args.output_dir}")
        
        # 推送到Hub
        if args.push_to_hub:
            upload_folder(
                repo_id=args.hub_model_id,
                folder_path=args.output_dir,
                commit_message="训练完成",
                token=args.hub_token,
            )
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
