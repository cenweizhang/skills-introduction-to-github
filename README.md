# SDXL图像修复模型微调项目

基于Hugging Face Diffusers库的SDXL图像修复模型微调项目，支持使用自定义数据集训练图像修复模型。

## 项目特性

- 🎨 支持SDXL图像修复模型微调
- 📊 多种数据格式支持（JSON、文件夹结构）
- 🔧 灵活的掩码生成策略
- ⚡ 混合精度训练支持
- 📈 Wandb训练监控
- 🚀 高效的推理接口
- 🎯 多种预定义配置

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 16GB+ GPU内存 (推荐)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

### 数据格式

支持以下数据格式：

#### 1. JSON格式
```json
[
  {
    "image_path": "path/to/image.jpg",
    "mask_path": "path/to/mask.png",
    "caption": "图像描述文本"
  }
]
```

#### 2. 文件夹结构
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── masks/
│   │   ├── image1.png
│   │   └── image2.png
│   └── captions/
│       ├── image1.txt
│       └── image2.txt
```

#### 3. 简单文件夹
```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

### 数据要求

- **图像格式**: JPG, PNG
- **图像尺寸**: 建议1024x1024或更高
- **掩码格式**: PNG (黑白图像，白色为需要修复的区域)
- **描述文件**: 纯文本文件，每行一个描述

## 快速开始

### 1. 准备数据

将您的训练数据放置在 `data/train/` 目录下，按照上述格式组织。

### 2. 配置训练参数

编辑 `config.yaml` 文件，调整训练参数：

```yaml
# 数据配置
train_data_dir: "./data/train"
resolution: 1024

# 训练配置
train_batch_size: 1
max_train_steps: 10000
learning_rate: 0.00001

# 其他配置...
```

### 3. 开始训练

```bash
python train_sdxl_inpaint.py --config config.yaml
```

### 4. 使用微调后的模型

```bash
python inference.py \
    --model_path ./sdxl-inpaint-finetuned \
    --image input.jpg \
    --prompt "修复后的图像描述" \
    --output result.png
```

## 详细使用说明

### 训练配置

#### 基础配置
- `pretrained_model_name_or_path`: 预训练模型路径
- `train_data_dir`: 训练数据目录
- `output_dir`: 输出目录
- `resolution`: 图像分辨率
- `train_batch_size`: 训练批次大小

#### 训练参数
- `max_train_steps`: 最大训练步数
- `learning_rate`: 学习率
- `gradient_accumulation_steps`: 梯度累积步数
- `mixed_precision`: 混合精度类型

#### 数据增强
- `mask_generation_strategy`: 掩码生成策略
  - `random`: 随机矩形掩码
  - `irregular`: 不规则掩码
  - `center`: 中心掩码
- `mask_probability`: 掩码概率
- `min_mask_ratio`: 最小掩码比例
- `max_mask_ratio`: 最大掩码比例

### 预定义配置

项目提供了多种预定义配置：

```python
from config import get_config

# 小规模训练
config = get_config("small")

# 中等规模训练
config = get_config("medium")

# 大规模训练
config = get_config("large")

# 调试模式
config = get_config("debug")
```

### 训练监控

启用Wandb监控：

```yaml
use_wandb: true
wandb_project: "my-sdxl-inpaint"
wandb_run_name: "experiment-1"
```

### 推理使用

#### 基础推理
```python
from inference import SDXLInpaintInference

# 创建推理器
inference = SDXLInpaintInference("./sdxl-inpaint-finetuned")

# 执行修复
result = inference.inpaint(
    image="input.jpg",
    mask="mask.png",
    prompt="修复后的图像描述",
    num_inference_steps=50,
    guidance_scale=7.5
)

# 保存结果
result.save("output.png")
```

#### 批量处理
```python
# 批量修复
results = inference.batch_inpaint(
    image_paths=["img1.jpg", "img2.jpg"],
    mask_paths=["mask1.png", "mask2.png"],
    prompts=["描述1", "描述2"],
    output_dir="./results"
)
```

#### 交互式掩码创建
```python
# 加载图像
image = Image.open("input.jpg")

# 创建掩码
mask = inference.create_mask_interactive(image, mask_type="rectangle")

# 执行修复
result = inference.inpaint(image, mask, "修复描述")
```

## 高级功能

### 自定义数据加载器

```python
from data_loader import InpaintDataset

dataset = InpaintDataset(
    data_dir="./data/train",
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    size=1024,
    mask_generation_strategy="irregular",
    mask_probability=0.7
)
```

### 自定义训练循环

```python
from train_sdxl_inpaint import SDXLInpaintTrainer
from config import TrainingConfig

# 加载配置
config = TrainingConfig.from_yaml("config.yaml")

# 创建训练器
trainer = SDXLInpaintTrainer(config)

# 开始训练
trainer.train()
```

### 模型检查点

训练过程中会自动保存检查点：

```
output_dir/
├── checkpoint-1000/
│   ├── unet/
│   └── scheduler/
├── checkpoint-2000/
│   ├── unet/
│   └── scheduler/
└── ...
```

## 性能优化

### 内存优化
- 使用混合精度训练 (`mixed_precision: "fp16"`)
- 启用梯度检查点 (`gradient_checkpointing: true`)
- 调整批次大小和梯度累积步数

### 训练加速
- 使用多GPU训练
- 启用数据并行
- 优化数据加载器

### 推理优化
- 使用FP16精度
- 启用注意力切片
- 模型CPU卸载

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批次大小
   - 增加梯度累积步数
   - 使用混合精度训练

2. **训练速度慢**
   - 检查数据加载器配置
   - 启用混合精度
   - 使用更快的存储设备

3. **模型质量差**
   - 调整学习率
   - 增加训练步数
   - 检查数据质量

### 调试模式

使用调试配置进行快速测试：

```bash
python train_sdxl_inpaint.py --config config_debug.yaml
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目基于MIT许可证开源。

## 致谢

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Stability AI](https://stability.ai/)
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

## 更新日志

### v1.0.0
- 初始版本发布
- 支持SDXL图像修复模型微调
- 提供完整的训练和推理接口
- 支持多种数据格式和配置选项