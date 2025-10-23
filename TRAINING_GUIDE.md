# SDXL Inpainting 模型微调训练指南

本指南将帮助您使用自己的数据微调 SDXL Inpainting 模型。

## 目录
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练配置](#训练配置)
- [开始训练](#开始训练)
- [模型推理](#模型推理)
- [常见问题](#常见问题)

## 环境准备

### 1. 系统要求

- **GPU**: 建议使用 NVIDIA GPU（12GB+ 显存）
- **RAM**: 16GB+
- **存储**: 至少 50GB 可用空间
- **Python**: 3.8+

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 如果在中国大陆，使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 安装 PyTorch (根据您的CUDA版本)

访问 https://pytorch.org 获取适合您系统的安装命令。

例如，CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. 可选：安装 xformers (加速训练)

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu118
```

## 数据准备

训练数据需要以下结构：

```
training_data/
├── images/          # 原始图像
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── masks/           # 对应的mask（白色区域=需要修复的区域）
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
└── prompts.txt      # 文本描述，每行对应一张图像
```

### 方法1: 自动生成mask

如果您只有完整的图像，可以使用脚本自动生成mask：

```bash
# 生成随机形状的mask
python prepare_dataset.py auto \
    --input_dir /path/to/your/images \
    --output_dir ./training_data \
    --mask_type random \
    --num_variants 1

# 生成笔刷风格的mask（更自然）
python prepare_dataset.py auto \
    --input_dir /path/to/your/images \
    --output_dir ./training_data \
    --mask_type brush \
    --num_variants 3  # 每张图生成3个变体
```

### 方法2: 使用已有的图像-mask对

如果您已经有图像和对应的mask：

```bash
python prepare_dataset.py from-pairs \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./training_data \
    --prompts_file /path/to/prompts.txt  # 可选
```

### 编辑提示词

生成数据后，请编辑 `training_data/prompts.txt` 文件，为每张图像添加准确的描述：

```
A beautiful landscape with mountains and lake
A portrait of a young woman with red hair
An interior design of modern living room
...
```

### 验证数据集

```bash
python prepare_dataset.py validate --data_dir ./training_data
```

## 训练配置

### 1. 编辑配置文件

打开 `training_config.yaml` 并根据您的需求调整参数：

```yaml
# 最重要的参数：
data:
  data_dir: "./training_data"
  resolution: 1024  # SDXL默认1024，显存不足可降至768

training:
  train_batch_size: 1  # 根据显存调整
  gradient_accumulation_steps: 4  # 模拟batch_size=4
  learning_rate: 1.0e-5
  num_train_epochs: 100

advanced:
  mixed_precision: "fp16"  # 启用混合精度
  gradient_checkpointing: true  # 节省显存

output:
  output_dir: "./output/sdxl-inpainting-finetuned"
  checkpointing_steps: 500
```

### 2. 显存优化建议

根据您的GPU显存选择配置：

#### 24GB 显存 (RTX 3090/4090)
```yaml
train_batch_size: 1
gradient_accumulation_steps: 4
resolution: 1024
mixed_precision: "fp16"
gradient_checkpointing: true
use_8bit_adam: false
```

#### 16GB 显存 (RTX 4070Ti)
```yaml
train_batch_size: 1
gradient_accumulation_steps: 8
resolution: 1024
mixed_precision: "fp16"
gradient_checkpointing: true
use_8bit_adam: true
```

#### 12GB 显存 (RTX 3060)
```yaml
train_batch_size: 1
gradient_accumulation_steps: 16
resolution: 768  # 降低分辨率
mixed_precision: "fp16"
gradient_checkpointing: true
use_8bit_adam: true
```

## 开始训练

### 使用命令行参数

```bash
python train_sdxl_inpainting.py \
    --data_dir ./training_data \
    --output_dir ./output/sdxl-inpainting-finetuned \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --num_train_epochs 100 \
    --mixed_precision fp16 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 3 \
    --random_flip \
    --seed 42
```

### 使用配置文件（推荐）

创建一个启动脚本 `run_training.sh`:

```bash
#!/bin/bash

# 从配置文件读取参数并启动训练
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
    --checkpointing_steps 500 \
    --random_flip
```

然后运行：

```bash
chmod +x run_training.sh
./run_training.sh
```

### 多GPU训练

如果您有多个GPU：

```bash
accelerate config  # 首次运行需要配置

accelerate launch train_sdxl_inpainting.py \
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

### 监控训练进度

#### TensorBoard
```bash
tensorboard --logdir ./output/sdxl-inpainting-finetuned/logs
```

然后在浏览器中访问 http://localhost:6006

#### WandB（可选）
```bash
# 在训练命令中添加
--report_to wandb
```

## 模型推理

训练完成后，使用微调后的模型进行推理：

```python
from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image

# 加载微调后的模型
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "./output/sdxl-inpainting-finetuned",
    torch_dtype=torch.float16,
).to("cuda")

# 加载图像和mask
image = Image.open("test_image.jpg")
mask = Image.open("test_mask.jpg")

# 生成修复后的图像
prompt = "A beautiful landscape with clear sky"
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]

result.save("inpainted_result.jpg")
```

### 完整推理脚本

创建 `inference.py`:

```python
#!/usr/bin/env python3
import argparse
from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()
    
    # 加载模型
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    # 加载图像
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    
    # 推理
    result = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    ).images[0]
    
    result.save(args.output)
    print(f"结果已保存到 {args.output}")

if __name__ == "__main__":
    main()
```

使用：

```bash
python inference.py \
    --model_path ./output/sdxl-inpainting-finetuned \
    --image test.jpg \
    --mask mask.jpg \
    --prompt "A beautiful scene" \
    --output result.jpg
```

## 常见问题

### 1. 显存不足 (CUDA Out of Memory)

**解决方案：**
- 降低 `train_batch_size` 到 1
- 增加 `gradient_accumulation_steps`
- 降低 `resolution` (从1024降至768或512)
- 启用 `use_8bit_adam`
- 确保启用 `gradient_checkpointing`
- 启用 `mixed_precision="fp16"`

### 2. 训练速度慢

**解决方案：**
- 安装 xformers: `pip install xformers`
- 增加 `dataloader_num_workers`
- 使用更快的存储（SSD）
- 考虑多GPU训练

### 3. 训练损失不下降

**解决方案：**
- 检查数据集质量
- 调整学习率（尝试 1e-6 或 5e-6）
- 增加训练步数
- 检查mask是否正确（白色=需要修复的区域）
- 确保prompts准确描述图像内容

### 4. 生成质量不佳

**解决方案：**
- 增加训练数据量（建议至少100张以上）
- 提高数据质量（高分辨率、清晰的图像）
- 编写更准确的prompts
- 训练更多轮次
- 使用 EMA (指数移动平均): `--use_ema`
- 调整推理参数（`num_inference_steps`, `guidance_scale`）

### 5. 从检查点恢复训练

```bash
python train_sdxl_inpainting.py \
    --resume_from_checkpoint latest \
    # ... 其他参数
```

或指定具体检查点：
```bash
--resume_from_checkpoint ./output/sdxl-inpainting-finetuned/checkpoint-1000
```

### 6. 使用中国大陆模型源

如果下载Hugging Face模型很慢，可以使用镜像：

```bash
# 使用ModelScope
--pretrained_model_name_or_path AI-ModelScope/stable-diffusion-xl-base-1.0
```

或使用环境变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 进阶技巧

### 1. LoRA微调（显存更少）

如果显存非常有限，考虑使用LoRA：

```bash
# 需要安装 peft
pip install peft

# 使用LoRA训练（需要修改训练脚本支持LoRA）
```

### 2. 数据增强

在训练时使用更多数据增强：
- 颜色抖动
- 旋转
- 缩放
- 不同形状和大小的mask

### 3. 迁移学习

如果目标领域与通用图像差异大（如医学图像、艺术风格），可以：
1. 先在大型通用数据集上训练
2. 再在特定领域数据上fine-tune

### 4. 超参数调优

尝试不同的超参数组合：
- 学习率: [1e-6, 5e-6, 1e-5, 5e-5]
- 学习率调度器: [constant, cosine, linear]
- 批次大小和累积步数的组合

## 资源链接

- [Diffusers文档](https://huggingface.co/docs/diffusers)
- [SDXL论文](https://arxiv.org/abs/2307.01952)
- [Stable Diffusion社区](https://huggingface.co/stabilityai)
- [Accelerate文档](https://huggingface.co/docs/accelerate)

## 支持

如有问题，请检查：
1. 日志文件中的错误信息
2. TensorBoard中的训练曲线
3. 数据集是否格式正确

祝您训练成功！🎉
