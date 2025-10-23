#!/usr/bin/env python3
"""
SDXL图像修复模型微调训练脚本
基于Hugging Face Diffusers库
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusers import (
    StableDiffusionXLInpaintPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb

from data_loader import InpaintDataset
from config import TrainingConfig

# 设置日志
logger = get_logger(__name__)


class SDXLInpaintTrainer:
    """SDXL图像修复模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None,
        )
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 初始化模型组件
        self._load_models()
        self._setup_optimizer()
        self._setup_scheduler()
        self._prepare_training()
        
    def _load_models(self):
        """加载预训练模型"""
        logger.info("正在加载预训练模型...")
        
        # 加载文本编码器
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder",
        )
        
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
        )
        
        # 加载VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="vae",
        )
        
        # 加载UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="unet",
        )
        
        # 加载噪声调度器
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        
        # 冻结VAE和文本编码器（只训练UNet）
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        
        # 只训练UNet
        self.unet.requires_grad_(True)
        
        logger.info("模型加载完成")
        
    def _setup_optimizer(self):
        """设置优化器"""
        # 只优化UNet参数
        params_to_optimize = self.unet.parameters()
        
        self.optimizer = AdamW(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )
        
    def _setup_scheduler(self):
        """设置学习率调度器"""
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )
        
    def _prepare_training(self):
        """准备训练"""
        # 创建数据集
        self.train_dataset = InpaintDataset(
            data_dir=self.config.train_data_dir,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            size=self.config.resolution,
            center_crop=self.config.center_crop,
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
        )
        
        # 准备模型、优化器和调度器
        (
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        
        # 初始化wandb
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__,
            )
            
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        # 计算总步数
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        num_train_epochs = self.config.max_train_steps // num_update_steps_per_epoch
        
        # 训练循环
        global_step = 0
        
        for epoch in range(num_train_epochs):
            self.unet.train()
            
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch}",
                disable=not self.accelerator.is_local_main_process,
            )
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # 前向传播
                    loss = self._compute_loss(batch)
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.unet.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
                
                # 记录日志
                if self.accelerator.is_main_process:
                    if global_step % self.config.logging_steps == 0:
                        logs = {
                            "step_loss": loss.item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                        }
                        if self.config.use_wandb:
                            wandb.log(logs, step=global_step)
                        logger.info(f"Step {global_step}: {logs}")
                    
                    # 保存检查点
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step)
                
                global_step += 1
                
                if global_step >= self.config.max_train_steps:
                    break
            
            progress_bar.close()
            
            if global_step >= self.config.max_train_steps:
                break
        
        # 保存最终模型
        self._save_final_model()
        logger.info("训练完成!")
        
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算训练损失"""
        # 获取输入数据
        images = batch["pixel_values"]
        masks = batch["masked_images"]
        input_ids = batch["input_ids"]
        input_ids_2 = batch["input_ids_2"]
        
        # 编码图像
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # 编码掩码
        masked_latents = self.vae.encode(masks).latent_dist.sample()
        masked_latents = masked_latents * self.vae.config.scaling_factor
        
        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 编码文本
        encoder_hidden_states = self.text_encoder(input_ids)[0]
        encoder_hidden_states_2 = self.text_encoder_2(input_ids_2)[0]
        
        # 准备输入
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)
        
        # 前向传播
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        
        # 计算损失
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        return loss
        
    def _save_checkpoint(self, step: int):
        """保存检查点"""
        if self.accelerator.is_main_process:
            checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存UNet
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(checkpoint_dir / "unet")
            
            # 保存调度器
            self.noise_scheduler.save_pretrained(checkpoint_dir / "scheduler")
            
            logger.info(f"检查点已保存到 {checkpoint_dir}")
            
    def _save_final_model(self):
        """保存最终模型"""
        if self.accelerator.is_main_process:
            # 创建完整的pipeline
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.config.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                vae=self.vae,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                scheduler=self.noise_scheduler,
            )
            
            # 保存完整pipeline
            pipeline.save_pretrained(self.config.output_dir)
            
            logger.info(f"最终模型已保存到 {self.config.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SDXL图像修复模型微调")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    # 加载配置
    config = TrainingConfig.from_yaml(args.config)
    
    # 创建训练器
    trainer = SDXLInpaintTrainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()