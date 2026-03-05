"""
MedVision - A Medical Image Segmentation Framework based on PyTorch Lightning
"""

__version__ = "0.1.1"

# 主要接口
from medvision.utils.config import parse_config
from medvision.utils.trainer import train_model
from medvision.utils.evaluator import test_model
from medvision.utils.inference import predict_model

__all__ = [
    "__version__",
    "parse_config",
    "train_model",
    "test_model",
    "predict_model",
]
