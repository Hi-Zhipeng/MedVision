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

    transforms_list = []

    for transform_config in config["Compose"]:
        transform_name = list(transform_config.keys())[0]
        transform_params = transform_config[transform_name]

        if transform_name == "Resize":
            size = transform_params["size"]
            transforms_list.append(T.Resize(size))
        elif transform_name == "RandomHorizontalFlip":
            p = transform_params.get("p", 0.5)
            transforms_list.append(T.RandomHorizontalFlip(p))
        elif transform_name == "RandomVerticalFlip":
            p = transform_params.get("p", 0.5)
            transforms_list.append(T.RandomVerticalFlip(p))
        elif transform_name == "RandomRotation":
            degrees = transform_params["degrees"]
            transforms_list.append(T.RandomRotation(degrees))
        elif transform_name == "ColorJitter":
            brightness = transform_params.get("brightness", 0)
            contrast = transform_params.get("contrast", 0)
            saturation = transform_params.get("saturation", 0)
            hue = transform_params.get("hue", 0)
            transforms_list.append(T.ColorJitter(brightness, contrast, saturation, hue))
        elif transform_name == "RandomCrop":
            size = transform_params["size"]
            transforms_list.append(T.RandomCrop(size))
        elif transform_name == "CenterCrop":
            size = transform_params["size"]
            transforms_list.append(T.CenterCrop(size))
        elif transform_name == "ToTensor":
            transforms_list.append(T.ToTensor())
        elif transform_name == "Normalize":
            mean = transform_params["mean"]
            std = transform_params["std"]
            transforms_list.append(T.Normalize(mean, std))
        else:
            raise ValueError(f"Unknown transform: {transform_name}")

    # Create a custom transform that handles both image and mask
    class ColorImageTransform:
        def __init__(self, transforms_list):
            self.transforms = T.Compose(transforms_list)

        def __call__(self, sample):
            image, mask = sample["image"], sample["mask"]

            # Apply transforms to image
            if isinstance(image, torch.Tensor):
                # If already tensor, apply normalization only
                image = self.transforms(image)
            else:
                # Apply all transforms
                image = self.transforms(image)

            # Handle mask transforms
            if isinstance(mask, torch.Tensor):
                # Keep mask as is
                pass
            else:
                # Convert mask to tensor if it's PIL
                mask = T.ToTensor()(mask)
                mask = (mask * 255).long().squeeze(0)  # Convert to long tensor

            return {"image": image, "mask": mask}

    return ColorImageTransform(transforms_list)