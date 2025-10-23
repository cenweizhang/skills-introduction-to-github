# SDXL Inpainting 微调训练工具包

🎨 基于Diffusers库的SDXL图像修复模型微调训练框架

## 📋 目录

- [功能特性](#-功能特性)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [数据准备](#-数据准备)
- [配置说明](#-配置说明)
- [训练流程](#-训练流程)
- [模型推理](#-模型推理)
- [常见问题](#-常见问题)
- [进阶使用](#-进阶使用)

## 🌟 功能特性

- ✅ **完整的SDXL Inpainting微调流程**：基于最新的Diffusers库实现
- ✅ **灵活的数据处理**：支持多种数据格式和遮罩生成策略
- ✅ **丰富的训练配置**：支持混合精度、梯度累积、EMA等高级功能
- ✅ **实时监控**：集成TensorBoard和WandB支持
- ✅ **内存优化**：支持xFormers、8bit优化器等内存优化技术
- ✅ **断点续训**：支持从检查点恢复训练
- ✅ **简单易用**：提供配置文件和命令行两种使用方式

## 🔧 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐24GB显存以上)
- **内存**: 32GB以上系统内存
- **存储**: 100GB以上可用空间

### 软件要求
- **Python**: 3.8+
- **CUDA**: 11.8+
- **PyTorch**: 2.0+

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd sdxl-inpainting-finetune

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 可选：安装GPU优化包
pip install xformers bitsandbytes
```

### 2. 数据准备

```bash
# 创建数据目录结构
python data_utils.py --data_root ./data/train --action prepare

# 将您的图像放入 ./data/train/images/ 目录
# 可选：将对应的遮罩放入 ./data/train/masks/ 目录
```

### 3. 创建配置文件

```bash
# 创建默认配置
python run_training.py --create-config

# 或使用预设配置模板
python train_config.py --action create_templates
```

### 4. 开始训练

```bash
# 使用默认配置训练
python run_training.py

# 或指定自定义配置
python run_training.py --config config_high_quality.yaml

# 快速测试训练（小规模）
python run_training.py --config config_quick_test.yaml
```

## 📁 数据准备

### 数据目录结构

```
data/
├── train/
│   ├── images/           # 训练图像
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── masks/            # 遮罩图像（可选）
│   │   ├── image1_mask.png
│   │   ├── image2_mask.png
│   │   └── ...
│   └── captions.json     # 图像描述（可选）
└── README.md
```

### 数据格式要求

#### 图像文件
- **格式**: JPG, JPEG, PNG, BMP, TIFF
- **分辨率**: 建议1024x1024或更高
- **质量**: 清晰，无明显噪声

#### 遮罩文件（可选）
- **格式**: PNG（推荐）或JPG
- **命名**: `图像名_mask.png` 或 `图像名.mask.png`
- **内容**: 白色(255)表示需要修复的区域，黑色(0)表示保持不变

#### 标题文件（可选）
```json
{
  "path/to/image1.jpg": "修复图像中的缺失部分",
  "path/to/image2.png": "填补图像中的空白区域"
}
```

### 自动生成标题和遮罩

```bash
# 为所有图像生成默认标题
python data_utils.py --data_root ./data/train --action create_captions

# 训练时自动生成随机遮罩（在配置文件中设置 mask_type: "random"）
```

## ⚙️ 配置说明

### 主要配置项

```yaml
# 模型配置
model:
  pretrained_model_name_or_path: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
  cache_dir: "./models"

# 数据配置
data:
  train_data_dir: "./data/train"
  caption_file: "./data/captions.json"

# 图像处理配置
image:
  resolution: 1024
  mask_type: "random"  # random, provided, mixed
  mask_ratio_range: [0.1, 0.3]

# 训练配置
training:
  train_batch_size: 1
  num_train_epochs: 100
  learning_rate: 1e-5
  mixed_precision: "fp16"
  use_ema: true

# 日志配置
logging:
  output_dir: "./output"
  checkpointing_steps: 1000
  validation_steps: 500
```

### 预设配置模板

- `config.yaml`: 默认配置
- `config_quick_test.yaml`: 快速测试（5轮训练）
- `config_high_quality.yaml`: 高质量训练（200轮训练）
- `config_large_dataset.yaml`: 大数据集优化配置

## 🏃 训练流程

### 1. 环境检查

```bash
# 检查环境和配置
python run_training.py --check-only
```

### 2. 开始训练

```bash
# 基础训练
python run_training.py

# 自定义参数训练
python run_training.py \
    --data-dir ./my_data \
    --output-dir ./my_output \
    --batch-size 2 \
    --epochs 50
```

### 3. 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir ./output/logs

# 或查看WandB（如果配置了）
# 访问 https://wandb.ai
```

### 4. 断点续训

```bash
# 从最新检查点恢复
python run_training.py --resume latest

# 从指定检查点恢复
python run_training.py --resume ./output/checkpoint-5000
```

## 🔮 模型推理

训练完成后，使用微调后的模型进行推理：

```python
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import torch

# 加载微调后的模型
pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    "./output",  # 训练输出目录
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipeline.to("cuda")

# 加载图像和遮罩
image = Image.open("input_image.jpg")
mask = Image.open("mask_image.png")

# 生成修复结果
prompt = "高质量的图像修复"
result = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    strength=0.8
).images[0]

result.save("repaired_image.jpg")
```

## ❓ 常见问题

### Q: 显存不足怎么办？

A: 尝试以下优化方案：
```yaml
training:
  train_batch_size: 1
  gradient_accumulation_steps: 8
  mixed_precision: "fp16"
  enable_xformers_memory_efficient_attention: true
  gradient_checkpointing: true
```

### Q: 训练速度太慢？

A: 优化建议：
- 使用更大的批次大小
- 启用xFormers内存优化
- 使用多GPU训练
- 减少验证频率

### Q: 如何提高训练质量？

A: 质量优化：
- 增加训练轮数
- 使用更小的学习率
- 启用EMA
- 使用高质量的训练数据
- 调整遮罩生成策略

### Q: 支持多GPU训练吗？

A: 是的，使用accelerate启动：
```bash
accelerate config  # 配置多GPU
accelerate launch train_sdxl_inpainting.py --config config.yaml
```

## 🔬 进阶使用

### 自定义数据集类

```python
from data_utils import InpaintingDataset

class CustomInpaintingDataset(InpaintingDataset):
    def __getitem__(self, idx):
        # 自定义数据加载逻辑
        item = super().__getitem__(idx)
        # 添加自定义处理
        return item
```

### 自定义遮罩生成

```python
from data_utils import MaskGenerator

# 创建自定义遮罩生成器
mask = MaskGenerator.generate_brush_mask((1024, 1024))
mask = MaskGenerator.generate_object_mask((1024, 1024))
mask = MaskGenerator.generate_edge_mask((1024, 1024))
```

### 训练回调和钩子

```python
# 在train_sdxl_inpainting.py中添加自定义回调
def custom_validation_callback(pipeline, step):
    # 自定义验证逻辑
    pass
```

### 模型导出和部署

```python
# 导出为ONNX格式
from diffusers import StableDiffusionXLInpaintPipeline
import torch

pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("./output")
# 导出逻辑...
```

## 📊 性能基准

| 配置 | GPU | 批次大小 | 训练时间/轮 | 显存占用 |
|------|-----|----------|-------------|----------|
| 基础 | RTX 4090 | 1 | ~2分钟 | ~20GB |
| 优化 | RTX 4090 | 2 | ~3分钟 | ~22GB |
| 高质量 | A100 | 4 | ~4分钟 | ~35GB |

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用Apache 2.0许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Diffusers](https://github.com/huggingface/diffusers) - 核心扩散模型库
- [Transformers](https://github.com/huggingface/transformers) - 文本编码器支持
- [Accelerate](https://github.com/huggingface/accelerate) - 分布式训练支持

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！