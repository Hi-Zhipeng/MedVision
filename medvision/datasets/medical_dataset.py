"""
Medical image dataset implementation.
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, Union, List

import torch
from torch.utils.data import Dataset

from medvision.utils.io import load_image


class MedicalImageDataset(Dataset):
    """
    Generic dataset for medical image segmentation.
    
    Directory structure should be:
    - data_dir/
        - images/
            - img1.nii.gz
            - img2.nii.gz
            - ...
        - masks/
            - img1.nii.gz
            - img2.nii.gz
            - ...
    
    Or provide custom loaders for different directory structures.
    """
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 transform: Optional[Callable] = None,
                 mode: str = "train",
                 image_subdir: str = "images",
                 mask_subdir: str = "masks",
                 image_suffix: str = "*.nii.gz",
                 mask_suffix: str = "*.nii.gz",
                 image_loader: Optional[Callable] = None,
                 mask_loader: Optional[Callable] = None):
        """
        Initialize the medical image dataset.
        
        Args:
            data_dir: Path to data directory
            transform: Transform to apply to image and mask
            mode: Dataset mode ('train', 'val', 'test')
            image_subdir: Subdirectory for images
            mask_subdir: Subdirectory for masks
            image_suffix: Suffix pattern for image files
            mask_suffix: Suffix pattern for mask files
            image_loader: Custom loader for images
            mask_loader: Custom loader for masks
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        # 自动拼接子集路径
        subset_dir = self.data_dir / self.mode
        self.image_dir = subset_dir / image_subdir
        self.mask_dir = subset_dir / mask_subdir
        
        # Loaders
        self.image_loader = image_loader or load_image
        self.mask_loader = mask_loader or load_image
        
        # Get file paths
        # Make directory if it doesn't exist
        try:
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.mask_dir, exist_ok=True)
        except Exception as e:
            print(f"警告: 无法创建数据目录: {e}")
            
        self.image_files = sorted(glob.glob(str(self.image_dir / image_suffix)))
        
        # In test mode, there might not be masks
        if mode == "test" and not os.path.exists(self.mask_dir):
            self.mask_files = [None] * len(self.image_files)
        else:
            self.mask_files = sorted(glob.glob(str(self.mask_dir / mask_suffix)))
        
        # Check if files exist
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.image_dir} with pattern {image_suffix}")
            
        if mode != "test" and len(self.mask_files) == 0:
            raise ValueError(f"No mask files found in {self.mask_dir}")
            
        if mode != "test" and len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of image files ({len(self.image_files)}) doesn't match "
                             f"number of mask files ({len(self.mask_files)})")
    
    def __len__(self):
        """
        Get dataset length.
        
        Returns:
            Number of samples
        """
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, mask)
        """

        # 加载真实数据
        image_path = self.image_files[idx]
        image = self.image_loader(image_path)


        # 加载掩码（如果可用）
        if self.mode != "predict" and self.mask_files[idx] is not None:
            mask_path = self.mask_files[idx]
            mask = self.mask_loader(mask_path)
        else:
            # 创建空掩码
            print("*"*20)
            print("mask is None!")
            print("*"*20)
            mask = torch.zeros_like(image)

        # 应用变换
        if self.transform is not None:
            try:
                # 对于MONAI transforms，输入是字典格式(monai 处理标签需要有通道数)

                # mask = mask.unsqueeze(0)  
                sample_dict = {"image": image, "label": mask}
                transformed = self.transform(sample_dict)
                
                image = transformed["image"]
                mask = transformed["label"]
                mask = mask.squeeze(0)  

            except Exception as e:
                print(f"应用变换出错: {e}")
        
        return image, mask
