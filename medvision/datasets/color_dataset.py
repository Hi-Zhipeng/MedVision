"""
Color image dataset implementation.
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, Union, List
from PIL import Image

import torch
from torch.utils.data import Dataset


class ColorImageDataset(Dataset):
    """
    Generic dataset for color image segmentation.

    Directory structure should be:
    - data_dir/
        - images/
            - img1.jpg
            - img2.png
            - ...
        - masks/
            - img1.png
            - img2.png
            - ...

    Or provide custom loaders for different directory structures.
    """

    def __init__(self,
                 data_dir: Union[str, Path],
                 transform: Optional[Callable] = None,
                 mode: str = "train",
                 image_subdir: str = "images",
                 mask_subdir: str = "masks",
                 image_suffix: str = "*.jpg",
                 mask_suffix: str = "*.png",
                 image_loader: Optional[Callable] = None,
                 mask_loader: Optional[Callable] = None):
        """
        Initialize the color image dataset.

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

        # 根据模式设置目录路径
        if mode in ["train", "val", "test"]:
            # 如果有子目录，使用子目录
            subset_dir = self.data_dir / mode
            if subset_dir.exists():
                self.image_dir = subset_dir / image_subdir
                self.mask_dir = subset_dir / mask_subdir
            else:
                # 如果没有子目录，直接使用根目录下的子目录
                self.image_dir = self.data_dir / image_subdir
                self.mask_dir = self.data_dir / mask_subdir
        else:
            # 默认使用根目录下的子目录
            self.image_dir = self.data_dir / image_subdir
            self.mask_dir = self.data_dir / mask_subdir

        # Loaders
        self.image_loader = image_loader or self._default_image_loader
        self.mask_loader = mask_loader or self._default_mask_loader

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

    def _default_image_loader(self, path: str) -> Image.Image:
        """
        Default image loader for color images.

        Args:
            path: Path to image file

        Returns:
            PIL Image in RGB format
        """
        image = Image.open(path).convert('RGB')
        return image

    def _default_mask_loader(self, path: str) -> Image.Image:
        """
        Default mask loader for color images.

        Args:
            path: Path to mask file

        Returns:
            PIL Image in grayscale format
        """
        mask = Image.open(path).convert('L')  # Convert to grayscale
        return mask

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
        if self.mode != "test" and self.mask_files[idx] is not None:
            mask_path = self.mask_files[idx]
            mask = self.mask_loader(mask_path)
        else:
            # 创建空掩码 - 使用与图像相同的尺寸
            if isinstance(image, Image.Image):
                mask = Image.new('L', image.size, 0)
            else:
                mask = torch.zeros(image.shape[1:], dtype=torch.long)

        # 应用变换
        if self.transform is not None:
            try:
                # 对于torchvision transforms，输入是字典格式
                sample_dict = {"image": image, "mask": mask}
                transformed = self.transform(sample_dict)

                image = transformed["image"]
                mask = transformed["mask"]

                # 确保mask是long类型
                if isinstance(mask, torch.Tensor) and mask.dtype != torch.long:
                    mask = mask.long()

            except Exception as e:
                print(f"应用变换出错: {e}")
                # 如果变换失败，手动转换为tensor
                if isinstance(image, Image.Image):
                    image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
                if isinstance(mask, Image.Image):
                    mask = torch.from_numpy(np.array(mask)).long()

        # 如果没有应用变换，确保转换为tensor
        else:
            if isinstance(image, Image.Image):
                image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
            if isinstance(mask, Image.Image):
                mask = torch.from_numpy(np.array(mask)).long()

        return image, mask