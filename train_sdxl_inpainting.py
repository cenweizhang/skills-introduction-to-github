#!/usr/bin/env python3
"""
SDXL Inpainting模型微调训练脚本

基于Diffusers库实现SDXL inpainting模型的微调训练
支持自定义数据集和训练参数配置
"""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
import transformers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available


# 检查最小版本要求
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def log_validation(vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler, args, accelerator, weight_dtype, step):
    """验证函数：生成样本图像用于监控训练进度"""
    logger.info("运行验证...")
    
    # 创建验证管道
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        text_encoder_2=accelerator.unwrap_model(text_encoder_2),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=accelerator.unwrap_model(unet),
        scheduler=scheduler,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # 验证提示词和图像
    validation_prompts = [
        "修复图像中的缺失部分",
        "填补图像中的空白区域",
        "恢复图像的完整性",
    ]
    
    # 创建简单的测试图像和遮罩
    test_image = Image.new('RGB', (1024, 1024), color='white')
    mask_image = Image.new('L', (1024, 1024), color='black')
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle([400, 400, 600, 600], fill='white')

    image_logs = []
    for i, prompt in enumerate(validation_prompts):
        with torch.autocast("cuda"):
            image = pipeline(
                prompt=prompt,
                image=test_image,
                mask_image=mask_image,
                num_inference_steps=20,
                generator=generator,
            ).images[0]

        image_logs.append(
            {"validation_image": image, "validation_prompt": prompt}
        )

    # 保存验证图像
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["validation_image"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []

                formatted_images.append(np.asarray(images))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                image = log["validation_image"]
                validation_prompt = log["validation_prompt"]
                formatted_images.append(wandb.Image(image, caption=validation_prompt))

            tracker.log({"validation": formatted_images})

    del pipeline
    torch.cuda.empty_cache()


def make_train_dataset(args, tokenizer, tokenizer_2, accelerator):
    """创建训练数据集"""
    # 数据变换
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def tokenize_captions(examples, is_train=True):
        """标记化文本描述"""
        captions = []
        for caption in examples[args.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{args.caption_column}` should contain either strings or lists of strings."
                )

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs2 = tokenizer_2(
            captions, max_length=tokenizer_2.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids, inputs2.input_ids

    def preprocess_train(examples):
        """预处理训练数据"""
        images = [image.convert("RGB") for image in examples[args.image_column]]
        original_sizes = [(image.height, image.width) for image in images]
        crop_top_lefts = []
        
        # 应用变换
        images = [train_transforms(image) for image in images]
        
        # 创建遮罩（这里使用随机遮罩作为示例）
        masks = []
        masked_images = []
        
        for image in images:
            # 创建随机遮罩
            mask = torch.zeros(1, image.shape[1], image.shape[2])
            # 随机选择遮罩区域
            h, w = image.shape[1], image.shape[2]
            mask_h = random.randint(h//8, h//4)
            mask_w = random.randint(w//8, w//4)
            mask_y = random.randint(0, h - mask_h)
            mask_x = random.randint(0, w - mask_w)
            mask[:, mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 1
            
            # 创建被遮罩的图像
            masked_image = image.clone()
            masked_image = masked_image * (1 - mask)
            
            masks.append(mask)
            masked_images.append(masked_image)
            crop_top_lefts.append((0, 0))  # 简化处理

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = images
        examples["masks"] = masks
        examples["masked_images"] = masked_images
        examples["input_ids"], examples["input_ids_2"] = tokenize_captions(examples)

        return examples

    # 加载数据集
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names

    # 获取图像和标题列名
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # 预处理数据集
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # 设置变换
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    """数据整理函数"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    masks = torch.stack([example["masks"] for example in examples])
    masked_images = torch.stack([example["masked_images"] for example in examples])
    
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    
    input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "masks": masks,
        "masked_images": masked_images,
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }


def main():
    args = parse_args()
    
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 设置日志记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)

    # 创建输出目录
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # 加载调度器、分词器和模型
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    # 导入模型
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # 当前不支持在DeepSpeed ZeRO3中训练文本编码器
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
        )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # 冻结vae和text_encoder，只训练unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # 创建EMA模型
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 将在默认情况下启用梯度检查点，如果不需要，请设置为False
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 启用TF32以获得更高的训练速度
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # 初始化优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 获取数据集
    train_dataset = make_train_dataset(args, tokenizer, tokenizer_2, accelerator)

    # DataLoaders创建
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # 调度器和数学运算
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # 准备所有内容与accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # 为了混合精度训练，我们将权重转换为half-precision，因为我们只需要它们进行前向传播
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # 将vae, text_encoder移动到gpu并转换为weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # 我们需要重新计算我们的总训练步数，因为数据加载器的大小可能已经改变
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # 之后我们重新计算我们的训练轮数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 我们需要初始化跟踪器，我们使用主进程
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts", None)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # 训练！
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数量 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  每设备批次大小 = {args.train_batch_size}")
    logger.info(f"  总训练批次大小（包括并行、分布式和累积）= {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # 可能从检查点加载训练状态
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
            accelerator.print(
                f"检查点 '{args.resume_from_checkpoint}' 不存在。从头开始训练新模型。"
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"从检查点 {path} 恢复")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="步骤",
        # 只在主进程上显示进度条
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 将图像转换为潜在空间
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 将遮罩和被遮罩的图像转换为潜在空间
                masked_image_latents = vae.encode(
                    batch["masked_images"].reshape(batch["masked_images"].shape[0] * batch["masked_images"].shape[1], 3, args.resolution, args.resolution).to(weight_dtype)
                ).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                masks = batch["masks"]
                # 调整遮罩大小以匹配潜在空间
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                        for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

                # 采样噪声
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # 为每个图像采样随机时间步
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # 根据噪声调度器为每个时间步添加噪声到潜在变量
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 获取文本嵌入
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                encoder_hidden_states_2 = text_encoder_2(batch["input_ids_2"], return_dict=False)[0]
                
                # 连接文本嵌入
                encoder_hidden_states = torch.concat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)

                # 获取目标以预测噪声残差
                target = noise

                # 连接noisy_latents和mask以及masked_image_latents在通道维度上
                latent_model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)

                # 预测噪声残差并计算损失
                added_cond_kwargs = {"text_embeds": encoder_hidden_states_2, "time_ids": torch.cat([torch.tensor(batch["original_sizes"]), torch.tensor(batch["crop_top_lefts"])], dim=-1).to(accelerator.device)}
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if args.prediction_type is not None:
                    # 设置预测类型的调度器
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"未知预测类型 {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # 计算损失权重，如论文中所述：https://arxiv.org/abs/2303.09556
                    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L13
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # 收集所有进程的损失以进行日志记录（如果我们使用分布式训练）
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 检查是否应该保存检查点或进行验证
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"保存检查点到 {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # 存储UNet参数临时用于推理
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    log_validation(
                        vae,
                        text_encoder,
                        text_encoder_2,
                        tokenizer,
                        tokenizer_2,
                        unet,
                        noise_scheduler,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )
                    if args.use_ema:
                        # 切换回原始UNet参数
                        ema_unet.restore(unet.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # 创建管道（注意：unet和vae在不同的dtype上，所以我们需要转换）
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        # 运行最终验证
        if args.validation_prompts is not None:
            log_validation(
                vae,
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2,
                unet,
                noise_scheduler,
                args,
                accelerator,
                weight_dtype,
                step=global_step,
            )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="训练结束",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SDXL inpainting模型微调训练脚本")
    
    # 模型和数据参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="预训练模型的路径或标识符",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="要使用的预训练模型的特定模型版本",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="要使用的预训练模型的变体",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="要训练的数据集名称（来自🤗数据集库）",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="要使用的数据集配置名称",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="包含训练数据的文件夹",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="数据集中包含图像的列"
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="数据集中包含图像标题的列",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="用于调试或更快训练的最大训练样本数",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help="一组用于验证的提示",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-inpainting-finetuned",
        help="保存模型的输出目录",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="缓存预训练模型的目录",
    )
    parser.add_argument("--seed", type=int, default=None, help="用于可重现训练的种子")
    
    # 训练参数
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="输入图像的分辨率",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="是否对图像进行中心裁剪",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="是否随机翻转图像进行数据增强",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="训练批次大小"
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="要执行的总训练步数",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="在执行反向传播/更新之前累积梯度的更新步数",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="是否使用梯度检查点来节省内存",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="初始学习率",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="根据梯度累积步数、批次大小和进程数缩放学习率",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="要使用的调度器类型",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="学习率预热的步数"
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="用于重新加权损失的SNR加权gamma",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="是否使用8位Adam优化器"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="是否允许TF32在Ampere GPU上进行更快的训练",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="用于加载数据的子进程数",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam优化器的beta1参数"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Adam优化器的beta2参数"
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam优化器的权重衰减")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam优化器的epsilon值")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="最大梯度范数")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="调度器使用的预测类型",
        choices=["epsilon", "v_prediction"],
    )
    
    # 日志和保存参数
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard日志目录",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="是否使用混合精度",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="集成报告结果和日志的位置",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="用于分布式训练")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="每X更新保存检查点",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="要保存的最大检查点数",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="是否从之前的检查点恢复训练",
    )
    
    # 验证参数
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="每X步运行验证",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_sdxl_inpainting",
        help="用于wandb和tensorboard的项目名称",
    )
    
    # EMA参数
    parser.add_argument(
        "--use_ema", action="store_true", help="是否使用EMA模型"
    )
    
    # 高级训练参数
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="噪声偏移"
    )
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="输入扰动的数量"
    )
    
    # Hub参数
    parser.add_argument(
        "--push_to_hub", action="store_true", help="是否将模型推送到Hub"
    )
    parser.add_argument(
        "--hub_token", type=str, default=None, help="用于推送到Hub的令牌"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub上的模型ID",
    )
    
    # XFormers参数
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="是否使用xformers"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # 默认验证提示
    if args.validation_prompts is None:
        args.validation_prompts = [
            "修复图像中的缺失部分",
            "填补图像中的空白区域", 
            "恢复图像的完整性",
        ]

    return args


def compute_snr(noise_scheduler, timesteps):
    """
    计算信噪比(SNR)，用于损失权重
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # 扩展到匹配样本数量
    sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]

    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr


if __name__ == "__main__":
    main()