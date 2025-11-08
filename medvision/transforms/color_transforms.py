"""
Color image transforms using torchvision.
"""

from typing import Dict, Any, Optional, Callable, List
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F


def get_color_transforms(config: Dict[str, Any]) -> Optional[Callable]:
    """
    Create torchvision transforms for color images.

    Args:
        config: Transform configuration dictionary

    Returns:
        Transform function
    """
    if not config or "Compose" not in config:
        return None

    # The ColorImageTransform class handles the logic internally based on the config.
    # The old list-building logic is deprecated and removed.
    return ColorImageTransform(config["Compose"])

class ColorImageTransform:
    def __init__(self, config_list):
        from torchvision.transforms import InterpolationMode
        self.config_list = config_list  # 原始配置，按顺序应用
        self.interp = InterpolationMode
        # 仅用于对 image 单独应用的对象（例如 ColorJitter）
        self._cache = {}

    def __call__(self, sample):
        import numpy as np
        from PIL import Image
        image, mask = sample["image"], sample["mask"]

        # Ensure PIL images for spatial ops (if tensors, convert to PIL)
        if isinstance(image, torch.Tensor):
            image_pil = T.ToPILImage()(image)
        else:
            image_pil = image

        if isinstance(mask, torch.Tensor):
            # 如果 mask 是 tensor（H, W）或 (1,H,W)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask_np = (mask.squeeze(0).cpu().numpy()).astype(np.uint8)
            else:
                mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
        else:
            mask_pil = mask

        img = image_pil
        m = mask_pil

        # 按配置顺序应用，每个变换决定是否同时作用于 mask（空间变换）
        for transform_config in self.config_list:
            tname = list(transform_config.keys())[0]
            params = transform_config[tname]

            if tname == "Resize":
                size = params["size"]
                img = F.resize(img, size, interpolation=self.interp.BILINEAR)
                m = F.resize(m, size, interpolation=self.interp.NEAREST)

            elif tname == "RandomHorizontalFlip":
                p = params.get("p", 0.5)
                if torch.rand(1).item() < p:
                    img = F.hflip(img)
                    m = F.hflip(m)

            elif tname == "RandomVerticalFlip":
                p = params.get("p", 0.5)
                if torch.rand(1).item() < p:
                    img = F.vflip(img)
                    m = F.vflip(m)

            elif tname == "RandomRotation":
                degrees = params["degrees"]
                angle = T.RandomRotation.get_params(degrees)
                # 对 image 用双线性，对 mask 用最近邻
                img = F.rotate(img, angle, interpolation=self.interp.BILINEAR)
                m = F.rotate(m, angle, interpolation=self.interp.NEAREST)

            elif tname == "RandomCrop":
                size = params["size"]
                i, j, h, w = T.RandomCrop.get_params(img, size)
                img = F.crop(img, i, j, h, w)
                m = F.crop(m, i, j, h, w)

            elif tname == "CenterCrop":
                size = params["size"]
                img = F.center_crop(img, size)
                m = F.center_crop(m, size)

            elif tname == "ColorJitter":
                # 仅作用于 image（PIL 或 Tensor 都支持）
                if "ColorJitter" not in self._cache:
                    b = params.get("brightness", 0)
                    c = params.get("contrast", 0)
                    s = params.get("saturation", 0)
                    h = params.get("hue", 0)
                    self._cache["ColorJitter"] = T.ColorJitter(b, c, s, h)
                img = self._cache["ColorJitter"](img)

            elif tname == "ToTensor":
                img = T.ToTensor()(img)
                # 不对 mask 用 ToTensor（后面统一转换为 long）
                # 但如果后续有 Normalize，则 img 已为 tensor

            elif tname == "Normalize":
                mean = params["mean"]
                std = params["std"]
                # Ensure image is tensor
                if not isinstance(img, torch.Tensor):
                    img = T.ToTensor()(img)
                img = T.Normalize(mean, std)(img)

            else:
                raise ValueError(f"Unknown transform: {tname}")

        # 最终确保 mask 为 long tensor（单通道 H x W）
        if isinstance(m, Image.Image):
            mask_arr = np.array(m, dtype=np.int64)
            mask_tensor = torch.from_numpy(mask_arr).long()
        else:
            # 如果经过上面操作已经是 tensor（一般不会），转换为 long
            mask_tensor = torch.as_tensor(m).long()

        # 如果 image 还是 PIL（用户未包含 ToTensor），把它转换
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        print(f"Transformed image shape: {img.shape}, mask shape: {mask_tensor.shape}")

        return {"image": img, "mask": mask_tensor}