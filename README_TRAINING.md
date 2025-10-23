# SDXL Inpainting 模型微调项目

这是一个完整的 SDXL Inpainting 模型微调训练方案，可以让您使用自己的数据重新训练模型。

## 📁 项目文件说明

- **train_sdxl_inpainting.py** - 主训练脚本
- **prepare_dataset.py** - 数据准备工具
- **inference.py** - 推理脚本（使用训练好的模型）
- **run_training.sh** - 训练启动脚本
- **requirements.txt** - Python依赖
- **training_config.yaml** - 训练配置文件（参考）
- **TRAINING_GUIDE.md** - 详细训练指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 方式A: 自动生成mask
```bash
python prepare_dataset.py auto \
    --input_dir /path/to/your/images \
    --output_dir ./training_data \
    --mask_type brush
```

#### 方式B: 使用已有的图像-mask对
```bash
python prepare_dataset.py from-pairs \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./training_data
```

### 3. 编辑提示词

编辑 `training_data/prompts.txt`，为每张图像添加描述。

### 4. 开始训练

```bash
# 方式1: 使用启动脚本（推荐）
bash run_training.sh

# 方式2: 直接运行
python train_sdxl_inpainting.py \
    --data_dir ./training_data \
    --output_dir ./output/sdxl-inpainting-finetuned \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --num_train_epochs 100 \
    --mixed_precision fp16 \
    --checkpointing_steps 500
```

### 5. 使用模型推理

```bash
python inference.py \
    --model_path ./output/sdxl-inpainting-finetuned \
    --image test_image.jpg \
    --mask test_mask.jpg \
    --prompt "A beautiful landscape" \
    --output result.jpg
```

## 💡 重要提示

1. **显存需求**: SDXL模型很大，建议至少12GB显存
2. **数据量**: 建议至少100张训练图像以获得好的效果
3. **训练时间**: 根据数据量和硬件，可能需要数小时到数天
4. **Mask格式**: Mask应该是灰度图，白色(255)表示需要修复的区域

## 📚 详细文档

查看 **TRAINING_GUIDE.md** 获取完整的训练指南，包括：
- 详细的环境配置
- 数据准备最佳实践
- 显存优化技巧
- 常见问题解答
- 进阶技巧

## 🔧 显存优化

如果遇到显存不足，尝试：

```bash
python train_sdxl_inpainting.py \
    --data_dir ./training_data \
    --output_dir ./output/sdxl-inpainting-finetuned \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --resolution 768 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate 1e-5 \
    --num_train_epochs 100 \
    --mixed_precision fp16
```

## 📊 监控训练

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir ./output/sdxl-inpainting-finetuned/logs
```

## 🌟 特性

✅ 完整的SDXL Inpainting微调支持  
✅ 自动数据准备工具  
✅ 灵活的配置选项  
✅ 显存优化（梯度检查点、8bit Adam等）  
✅ 多GPU训练支持  
✅ 断点续训  
✅ TensorBoard/WandB监控  
✅ 详细的文档和示例  

## 🤝 相关资源

- [Diffusers官方文档](https://huggingface.co/docs/diffusers)
- [原始SDXL Inpainting示例](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
- [Stable Diffusion社区](https://huggingface.co/stabilityai)

---

祝您训练成功！如有问题，请查看TRAINING_GUIDE.md中的常见问题部分。
