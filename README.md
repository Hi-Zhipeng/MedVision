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
  type: "unet"
  in_channels: 1
  out_channels: 1
  features: [32, 64, 128, 256]
  dropout: 0.1
  loss:
    type: "dice"
    smooth: 1e-5
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    type: "plateau"
    patience: 5
    factor: 0.5
    monitor: "val_loss"
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
  data_dir: "data/brain_tumor"
  train_val_split: [0.8, 0.2]
  train_transforms:
    resize:
      height: 256
      width: 256
    randomrotate90:
      p: 0.5
    flip:
      p: 0.5
    randombrightness:
      limit: 0.2
      p: 0.5
    normalize:
      mean: [0.0]
      std: [1.0]
  val_transforms:
    resize:
      height: 256
      width: 256
    normalize:
      mean: [0.0]
      std: [1.0]
  test_transforms:
    resize:
      height: 256
      width: 256
    normalize:
      mean: [0.0]
      std: [1.0]
      
# Training configuration
training:
  max_epochs: 100
  gpus: 1
  precision: 16
  output_dir: "outputs"
  experiment_name: "brain_tumor_segmentation"
  monitor: "val_loss"
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
  type: "unet"
  in_channels: 1
  out_channels: 1
  features: [32, 64, 128, 256]
  metrics:
    dice:
      type: "dice"
      threshold: 0.5
    iou:
      type: "iou"
      threshold: 0.5

# Checkpoint path
checkpoint_path: "outputs/brain_tumor_segmentation/checkpoints/last.ckpt"

# Data configuration
data:
  type: "medical"
  batch_size: 16
  num_workers: 4
  data_dir: "data/brain_tumor"
  test_transforms:
    resize:
      height: 256
      width: 256
    normalize:
      mean: [0.0]
      std: [1.0]
      
# Testing configuration
testing:
  gpus: 1
  precision: 16
  output_dir: "outputs/predictions"
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
