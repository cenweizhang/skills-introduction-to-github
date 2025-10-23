#!/bin/bash
# SDXL Inpainting 训练启动脚本
# 使用前请根据您的硬件配置修改参数

# 基础配置
DATA_DIR="./training_data"
OUTPUT_DIR="./output/sdxl-inpainting-finetuned"
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"

# 训练参数（根据您的GPU显存调整）
RESOLUTION=1024
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
LEARNING_RATE=1e-5
NUM_EPOCHS=100
MIXED_PRECISION="fp16"

# 启动训练
python train_sdxl_inpainting.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --pretrained_model_name_or_path ${PRETRAINED_MODEL} \
    --resolution ${RESOLUTION} \
    --train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --gradient_checkpointing \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler "constant" \
    --lr_warmup_steps 500 \
    --num_train_epochs ${NUM_EPOCHS} \
    --mixed_precision ${MIXED_PRECISION} \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 3 \
    --random_flip \
    --dataloader_num_workers 0 \
    --seed 42

# 如果需要使用8bit Adam优化器以节省显存，添加：
# --use_8bit_adam \

# 如果需要使用EMA，添加：
# --use_ema \

# 如果需要恢复训练，添加：
# --resume_from_checkpoint "latest" \
