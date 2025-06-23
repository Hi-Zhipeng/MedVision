# MedVision

MedVision 是一个基于 PyTorch Lightning 的医学影像分割框架，提供了训练和推理的简单接口。

## 特点

- 基于 PyTorch Lightning 的高级接口
- 支持常见的医学影像格式（NIfTI、DICOM 等）
- 内置多种分割模型架构（如 UNet）
- 灵活的数据加载和预处理管道
- 模块化设计，易于扩展
- 命令行界面用于训练和推理

## 安装

### 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 基本安装

最简单的安装方式：

```bash
pip install -e .
```

### 从源码安装

```bash
git clone https://github.com/yourusername/medvision.git
cd medvision
pip install -e .
```

### 使用requirements文件

```bash
# 基本环境
pip install -r requirements.txt

# 开发环境
pip install -r requirements-dev.txt
```

### 使用conda环境

推荐使用 conda 创建独立的虚拟环境：

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate medvision

# 安装项目本身
pip install -e .
```

如果您需要更新现有环境：

```bash
conda env update -f environment.yml --prune
```

如果您想删除环境：

```bash
conda env remove -n medvision
```

### 功能模块安装

根据需求选择特定的功能组：

```bash
# 医学影像处理
pip install -e ".[medical]"

# 数据变换
pip install -e ".[transforms]"

# 可视化工具
pip install -e ".[visualization]"

# 评估指标
pip install -e ".[metrics]"

# 开发工具
pip install -e ".[dev]"

# 文档生成
pip install -e ".[docs]"

# 完整安装
pip install -e ".[all]"
```

### 开发环境设置

如果您要参与开发：

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install

# 或使用Makefile
make install-dev
```

### 验证安装

```bash
python -c "import medvision; print(medvision.__version__)"
MedVision --help
```

## 快速入门

### 训练模型

```bash
MedVision train configs/train_config.yml
```

### 测试模型

```bash
MedVision test configs/test_config.yml
```

## 配置格式

### 训练配置示例

```yaml
# General settings
seed: 42

# Model configuration
model:
  type: "segmentation"
  network:
    name: "denseunet"

  in_channels: 1
  out_channels: 1
  features: [32, 64, 128, 256]
  dropout: 0.1
  loss:
    type: "dice"
    smooth: 0.00001
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    type: "plateau"
    patience: 5
    factor: 0.5
    monitor: "val/val_loss" #`train/train_loss`, `train/train_loss_step`, `val/val_loss`, `val/val_dice`, `val/val_iou`, `train/train_loss_epoch`, `train/train_dice`, `train/train_iou`
  metrics:
    dice:
      type: "dice"
      threshold: 0.5
    iou:
      type: "iou"
      threshold: 0.5

# Data configuration
data:
  type: "medical"
  batch_size: 8
  num_workers: 4
  data_dir: "data/2D"
  train_val_split: [0.8, 0.2]
  dataset_args:  
    image_subdir: "images" 
    mask_subdir: "masks"   
    image_suffix: "*.png"  
    mask_suffix: "*.png" 

# 使用简化但完整的MONAI transforms    
  train_transforms:
    # 1. 基础预处理
    Resized:
      keys: ["image", "label"]
      spatial_size: [256, 256]
      mode: ["bilinear", "nearest"]
      align_corners: [false, null]
    
    # 2. 空间变换 - 提升泛化能力
    RandRotated:
      keys: ["image", "label"]
      range_x: 0.2  # ±0.2弧度 ≈ ±11.5度
      range_y: 0.2
      prob: 0.5
      mode: ["bilinear", "nearest"]
      padding_mode: "border"
      align_corners: [false, null]
    
    RandFlipd:
      keys: ["image", "label"]
      spatial_axis: [0, 1]  # 水平和垂直翻转
      prob: 0.5
    
    RandAffined:
      keys: ["image", "label"]
      prob: 0.3
      rotate_range: [0.1, 0.1]  # 小角度旋转
      scale_range: [0.1, 0.1]   # 缩放范围 0.9-1.1
      translate_range: [10, 10] # 平移像素数
      mode: ["bilinear", "nearest"]
      padding_mode: "border"
      align_corners: [false, null]
    
    RandZoomd:
      keys: ["image", "label"]
      min_zoom: 0.85
      max_zoom: 1.15
      prob: 0.3
      mode: ["bilinear", "nearest"]
      align_corners: [false, null]
    
    # 3. 强度变换（仅对图像）
    RandAdjustContrastd:
      keys: ["image"]
      prob: 0.3
      gamma: [0.8, 1.2]  # 对比度调整范围
    
    RandScaleIntensityd:
      keys: ["image"]
      factors: 0.2  # 强度缩放因子
      prob: 0.3
    
    RandShiftIntensityd:
      keys: ["image"]
      offsets: 0.1  # 强度偏移
      prob: 0.3
    
    RandGaussianNoised:
      keys: ["image"]
      prob: 0.2
      mean: 0.0
      std: 0.1
    
    RandGaussianSmoothd:
      keys: ["image"]
      prob: 0.1
      sigma_x: [0.5, 1.0]
      sigma_y: [0.5, 1.0]
    
    RandBiasFieldd:
      keys: ["image"]
      prob: 0.15
      degree: 3
      coeff_range: [0.0, 0.1]
    
    # 4. 归一化
    NormalizeIntensityd:
      keys: ["image"]
      nonzero: true
      channel_wise: false
  val_transforms:
    # 验证时只做基础预处理
    Resized:
      keys: ["image", "label"]
      spatial_size: [256, 256]
      mode: ["bilinear", "nearest"]
      align_corners: [false, null]
    
    NormalizeIntensityd:
      keys: ["image"]
      nonzero: true
      channel_wise: false
      
  test_transforms:
    # 测试时只做基础预处理
    Resized:
      keys: ["image", "label"]
      spatial_size: [256, 256]
      mode: ["bilinear", "nearest"]
      align_corners: [false, null]
    
    NormalizeIntensityd:
      keys: ["image"]
      nonzero: true
      channel_wise: false
      
# Training configuration
training:
  max_epochs: 2
  devices: 1
  accelerator: "auto"
  precision: 16-mixed 
  output_dir: "outputs"
  experiment_name: "brain_tumor_segmentation"
  monitor: "val/val_loss"  #`train/train_loss`, `train/train_loss_step`, `val/val_loss`, `val/val_dice`, `val/val_iou`, `train/train_loss_epoch`, `train/train_dice`, `train/train_iou`
  monitor_mode: "min"
  early_stopping: true
  patience: 10
  save_top_k: 3
  log_every_n_steps: 10
  deterministic: false

```

### 测试配置示例

```yaml
# General settings
seed: 42

# Model configuration
model:
  type: "segmentation"
  network:
    name: "unet"
    
  in_channels: 1
  out_channels: 1
  features: [32, 64, 128, 256]
  dropout: 0.0 
  metrics:
    dice:
      type: "dice"
      threshold: 0.5
    iou:
      type: "iou" 
      threshold: 0.5
    accuracy:
      type: "accuracy"
      threshold: 0.5
  loss:
    type: "dice"
    smooth: 0.00001
# Checkpoint path
checkpoint_path: "outputs/checkpoints/last.ckpt"

# Data configuration
data:
  type: "medical"
  batch_size: 8  
  num_workers: 4
  data_dir: "data/2D"
  
  # 数据集参数 - 与训练配置保持一致
  dataset_args:  
    image_subdir: "images" 
    mask_subdir: "masks"   
    image_suffix: "*.png"  
    mask_suffix: "*.png"
  
  # 测试变换 - 与训练时的验证变换完全一致
  test_transforms:
    Resized:
      keys: ["image", "label"]
      spatial_size: [256, 256]
      mode: ["bilinear", "nearest"]
      align_corners: [false, null]
    
    NormalizeIntensityd:
      keys: ["image"]
      nonzero: true
      channel_wise: false
      
# Testing configuration
testing:
  devices: 1
  accelerator: "auto"
  precision: 16-mixed 
  output_dir: "outputs/predictions"

```

### 推理配置示例
```yaml

# Inference configuration for MedVision
# This config is for pure inference without labels

# General settings
seed: 42

# Model configuration (should match training)
model:
  type: "segmentation"
  network:
    name: "unet"
    
  in_channels: 1
  out_channels: 1
  features: [32, 64, 128, 256]
  dropout: 0.0  # 推理时关闭dropout
  # 推理时仍需要loss配置(但不会使用)
  loss:
    type: "dice"
    smooth: 0.00001

# Checkpoint path - 必须指定训练好的模型
checkpoint_path: "outputs/checkpoints/last.ckpt"

# Inference configuration
inference:
  # 输入图像目录 (只包含图像，不需要标签)
  image_dir: "data/2D/images"
  
  # 输出配置
  output_dir: "outputs/predictions"
  save_format: "png"  # png, npy
  
  # 数据加载配置
  batch_size: 4
  num_workers: 4
  pin_memory: true
  image_suffix: "*.png"
  
  # 硬件配置
  devices: 1
  accelerator: "auto"
  precision: 16-mixed 
  
  # 推理变换 (只处理图像，不需要label)
  transforms:
    Resized:
      keys: ["image"]  # 注意：只有image，没有label
      spatial_size: [256, 256]
      mode: "bilinear"
      align_corners: false
    
    NormalizeIntensity:  
      keys: ["image"]
      nonzero: true


```
## 自定义扩展

### 添加新的模型架构

1. 在 `medvision/models/` 目录下创建新的模型文件
2. 更新 `get_model` 函数以识别新的模型类型

### 添加新的数据集

1. 在 `medvision/datasets/` 目录下创建新的数据集类
2. 更新 `get_datamodule` 函数以识别新的数据集类型

## 许可证

MIT

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 获取详细信息。
