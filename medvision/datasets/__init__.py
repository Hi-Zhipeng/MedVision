"""Datasets module for MedVision."""

import os
import torch
from typing import Dict, Any, Optional, List, Tuple, Union
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

from medvision.datasets.medical_dataset import MedicalImageDataset
from medvision.datasets.color_dataset import ColorImageDataset


def get_datamodule(config: Dict[str, Any]) -> pl.LightningDataModule:
    """
    Factory function to create a datamodule based on configuration.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        A LightningDataModule implementation
    """
    dataset_type = config["type"].lower()
    
    if dataset_type == "medical":
        datamodule_class = MedicalDataModule
    elif dataset_type == "color":
        datamodule_class = ColorImageDataModule
    elif dataset_type == "custom":
        # Add your custom datamodule implementation here
        raise NotImplementedError(f"Custom dataset type not implemented")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return datamodule_class(config)


class ColorImageDataModule(pl.LightningDataModule):
    """
    DataModule for color image segmentation datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the color image data module.
        
        Args:
            config: Dataset configuration
        """
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", os.cpu_count() or 4)
        self.pin_memory = config.get("pin_memory", True)
        self.data_dir = config.get("data_dir", "./data")
        self.train_val_split = config.get("train_val_split", [0.8, 0.2])
        self.seed = config.get("seed", 42)
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Download and prepare data if needed.
        """
        # This method is called once and on only one GPU
        pass
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets based on stage.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        # Setup transforms
        from medvision.transforms import get_transforms
        self.train_transforms = get_transforms(self.config.get("train_transforms", {}))
        self.val_transforms = get_transforms(self.config.get("val_transforms", {}))
        self.test_transforms = get_transforms(self.config.get("test_transforms", {}))

        # 分别加载 train/val/test 子目录
        if stage == "fit" or stage is None:
            self.train_dataset = ColorImageDataset(
                data_dir=self.data_dir,
                transform=self.train_transforms,
                mode="train",
                **self.config.get("dataset_args", {})
            )
            self.val_dataset = ColorImageDataset(
                data_dir=self.data_dir,
                transform=self.val_transforms,
                mode="val",
                **self.config.get("dataset_args", {})
            )
        if stage == "test" or stage is None:
            self.test_dataset = ColorImageDataset(
                data_dir=self.data_dir,
                transform=self.test_transforms,
                mode="test",
                **self.config.get("dataset_args", {})
            )
            
    def train_dataloader(self):
        """
        Create training dataloader.
        
        Returns:
            Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """
        Create validation dataloader.
        
        Returns:
            Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        """
        Create test dataloader.
        
        Returns:
            Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class MedicalDataModule(pl.LightningDataModule):
    """
    Base DataModule for medical image segmentation datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the medical data module.
        
        Args:
            config: Dataset configuration
        """
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", os.cpu_count() or 4)
        self.pin_memory = config.get("pin_memory", True)
        self.data_dir = config.get("data_dir", "./data")
        self.train_val_split = config.get("train_val_split", [0.8, 0.2])
        self.seed = config.get("seed", 42)
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Download and prepare data if needed.
        """
        # This method is called once and on only one GPU
        pass
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets based on stage.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        # Setup transforms
        from medvision.transforms import get_transforms
        self.train_transforms = get_transforms(self.config.get("train_transforms", {}))
        self.val_transforms = get_transforms(self.config.get("val_transforms", {}))
        self.test_transforms = get_transforms(self.config.get("test_transforms", {}))

        # 分别加载 train/val/test 子目录
        if stage == "fit" or stage is None:
            self.train_dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                transform=self.train_transforms,
                mode="train",
                **self.config.get("dataset_args", {})
            )
            self.val_dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                transform=self.val_transforms,
                mode="val",
                **self.config.get("dataset_args", {})
            )
        if stage == "test" or stage is None:
            self.test_dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                transform=self.test_transforms,
                mode="test",
                **self.config.get("dataset_args", {})
            )
            
    def train_dataloader(self):
        """
        Create training dataloader.
        
        Returns:
            Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """
        Create validation dataloader.
        
        Returns:
            Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        """
        Create test dataloader.
        
        Returns:
            Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
