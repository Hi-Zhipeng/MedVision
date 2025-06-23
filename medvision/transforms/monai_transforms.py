"""Transforms module for MedVision using MONAI."""

try:
    import monai.transforms as M
    from monai.transforms import Compose
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

import numpy as np
from typing import Dict, Any, List, Callable, Optional, Union

def get_transforms(config: Dict[str, Any]) -> Optional[Callable]:
    """
    Factory function to create MONAI transforms based on configuration.
    
    Args:
        config: Transform configuration dictionary
        
    Returns:
        MONAI transform callable
    """
    if not config:
        return None
    
    if not HAS_MONAI:
        raise ImportError(
            "MONAI is required for transforms. "
            "Install it using `pip install monai`."
        )
    
    transforms = []
    
    for transform_name, transform_params in config.items():
        transform_name_lower = transform_name.lower()
        
        if transform_name_lower == "orientation":
            transforms.append(M.Orientation(
                keys=transform_params.get("keys", ["image", "label"]),
                axcodes=transform_params.get("axcodes", "RAS"),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "spacing":
            transforms.append(M.Spacing(
                keys=transform_params.get("keys", ["image", "label"]),
                pixdim=transform_params["pixdim"],
                mode=transform_params.get("mode", ["bilinear", "nearest"]),
                align_corners=transform_params.get("align_corners", [True, True]),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "cropforeground":
            transforms.append(M.CropForeground(
                keys=transform_params.get("keys", ["image", "label"]),
                source_key=transform_params.get("source_key", "image"),
                k_divisible=transform_params.get("k_divisible", 1),
                mode=transform_params.get("mode", "constant"),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower in ["resize", "resized"]:
            transforms.append(M.Resized(
                keys=transform_params.get("keys", ["image", "label"]),
                spatial_size=transform_params["spatial_size"],
                mode=transform_params.get("mode", ["area", "nearest"]),
                align_corners=transform_params.get("align_corners", [None, None]),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "resizewithpadorcrop":
            transforms.append(M.ResizeWithPadOrCrop(
                keys=transform_params.get("keys", ["image", "label"]),
                spatial_size=transform_params["spatial_size"],
                mode=transform_params.get("mode", "constant"),
                value=transform_params.get("value", 0),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "randcropbyposneglabel":
            transforms.append(M.RandCropByPosNegLabel(
                keys=transform_params.get("keys", ["image", "label"]),
                label_key=transform_params.get("label_key", "label"),
                spatial_size=transform_params["spatial_size"],
                pos=transform_params.get("pos", 1),
                neg=transform_params.get("neg", 1),
                num_samples=transform_params.get("num_samples", 4),
                image_key=transform_params.get("image_key", "image"),
                image_threshold=transform_params.get("image_threshold", 0),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "randrotate90":
            transforms.append(M.RandRotate90(
                keys=transform_params.get("keys", ["image", "label"]),
                prob=transform_params.get("prob", 0.1),
                max_k=transform_params.get("max_k", 3),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "randrotate":
            transforms.append(M.RandRotate(
                keys=transform_params.get("keys", ["image", "label"]),
                range_x=transform_params.get("range_x", 0.0),
                range_y=transform_params.get("range_y", 0.0),
                range_z=transform_params.get("range_z", 0.0),
                prob=transform_params.get("prob", 0.1),
                keep_size=transform_params.get("keep_size", True),
                mode=transform_params.get("mode", ["bilinear", "nearest"]),
                padding_mode=transform_params.get("padding_mode", ["zeros", "zeros"]),
                align_corners=transform_params.get("align_corners", [True, True]),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "randflip":
            transforms.append(M.RandFlip(
                keys=transform_params.get("keys", ["image", "label"]),
                spatial_axis=transform_params.get("spatial_axis", None),
                prob=transform_params.get("prob", 0.1),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower == "randaffine":
            transforms.append(M.RandAffine(
                keys=transform_params.get("keys", ["image", "label"]),
                prob=transform_params.get("prob", 0.1),
                rotate_range=transform_params.get("rotate_range", None),
                shear_range=transform_params.get("shear_range", None),
                translate_range=transform_params.get("translate_range", None),
                scale_range=transform_params.get("scale_range", None),
                mode=transform_params.get("mode", ["bilinear", "nearest"]),
                padding_mode=transform_params.get("padding_mode", ["zeros", "zeros"]),
                cache_grid=transform_params.get("cache_grid", False),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower in ["randzoom", "randzoomd"]:
            transforms.append(M.RandZoomd(
                keys=transform_params.get("keys", ["image", "label"]),
                min_zoom=transform_params.get("min_zoom", 0.9),
                max_zoom=transform_params.get("max_zoom", 1.1),
                mode=transform_params.get("mode", ["area", "nearest"]),
                align_corners=transform_params.get("align_corners", [None, None]),
                prob=transform_params.get("prob", 1.0),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            

            
        elif transform_name_lower in ["randscaleintensity", "randscaleintensityd"]:
            transforms.append(M.RandScaleIntensityd(
                keys=transform_params.get("keys", ["image"]),
                factors=transform_params.get("factors", 0.1),
                prob=transform_params.get("prob", 1.0),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower in ["randshiftintensity", "randshiftintensityd"]:
            transforms.append(M.RandShiftIntensityd(
                keys=transform_params.get("keys", ["image"]),
                offsets=transform_params.get("offsets", 0.1),
                prob=transform_params.get("prob", 1.0),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower in ["randgaussiansmooth", "randgaussiansmoothd"]:
            transforms.append(M.RandGaussianSmoothd(
                keys=transform_params.get("keys", ["image"]),
                sigma_x=transform_params.get("sigma_x", [0.25, 1.5]),
                sigma_y=transform_params.get("sigma_y", [0.25, 1.5]),
                sigma_z=transform_params.get("sigma_z", [0.25, 1.5]),
                prob=transform_params.get("prob", 1.0),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        elif transform_name_lower in ["randbiasfield", "randbiasfieldd"]:
            transforms.append(M.RandBiasFieldd(
                keys=transform_params.get("keys", ["image"]),
                degree=transform_params.get("degree", 3),
                coeff_range=transform_params.get("coeff_range", [0.0, 0.1]),
                prob=transform_params.get("prob", 1.0),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
            
        # ...existing code...
        elif transform_name_lower == "ensuretype":
            transforms.append(M.EnsureType(
                keys=transform_params.get("keys", ["image", "label"]),
                data_type=transform_params.get("data_type", "tensor"),
                dtype=transform_params.get("dtype", None),
                device=transform_params.get("device", None),
                wrap_sequence=transform_params.get("wrap_sequence", False),
                allow_missing_keys=transform_params.get("allow_missing_keys", False),
            ))
    
    if not transforms:
        return None
        
    try:
        return Compose(transforms)
    except Exception as e:
        print(f"Failed to create MONAI transforms: {e}")
        print("Returning identity transform")
        
        def identity_transform(data):
            return data
        
        return identity_transform


def create_basic_transforms_2d(
    spatial_size: tuple = (256, 256),
    pixdim: tuple = (1.0, 1.0),
    intensity_range: tuple = None,
    augmentation: bool = True
) -> Callable:
    """
    Create basic 2D transforms for medical image segmentation.
    
    Args:
        spatial_size: Target spatial size (H, W)
        pixdim: Target pixel spacing (x, y)
        intensity_range: Intensity range for normalization (min, max)
        augmentation: Whether to include data augmentation
        
    Returns:
        MONAI transform compose
    """
    if not HAS_MONAI:
        raise ImportError("MONAI is required for transforms")
    
    transforms = [
        M.Spacing(keys=["image", "label"], pixdim=pixdim, mode=["bilinear", "nearest"]),
        M.Resize(keys=["image", "label"], spatial_size=spatial_size, mode=["area", "nearest"]),
    ]
    
    if intensity_range:
        transforms.append(
            M.ScaleIntensityRange(
                keys=["image"],
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
        )
    else:
        transforms.append(M.NormalizeIntensity(keys=["image"], nonzero=True))
    
    if augmentation:
        transforms.extend([
            M.RandRotate90(keys=["image", "label"], prob=0.1, max_k=3),
            M.RandFlip(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            M.RandFlip(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            M.RandScaleIntensity(keys=["image"], factors=0.1, prob=0.1),
            M.RandAdjustContrast(keys=["image"], prob=0.1),
        ])
    
    transforms.append(M.EnsureType(keys=["image", "label"], data_type="tensor"))
    
    return Compose(transforms)


def create_basic_transforms_3d(
    spatial_size: tuple = (64, 64, 64),
    pixdim: tuple = (1.0, 1.0, 1.0),
    intensity_range: tuple = None,
    augmentation: bool = True
) -> Callable:
    """
    Create basic 3D transforms for medical image segmentation.
    
    Args:
        spatial_size: Target spatial size (D, H, W)
        pixdim: Target pixel spacing (x, y, z)
        intensity_range: Intensity range for normalization (min, max)
        augmentation: Whether to include data augmentation
        
    Returns:
        MONAI transform compose
    """
    if not HAS_MONAI:
        raise ImportError("MONAI is required for transforms")
    
    transforms = [
        M.Orientation(keys=["image", "label"], axcodes="RAS"),
        M.Spacing(keys=["image", "label"], pixdim=pixdim, mode=["bilinear", "nearest"]),
        M.CropForeground(keys=["image", "label"], source_key="image"),
        M.ResizeWithPadOrCrop(keys=["image", "label"], spatial_size=spatial_size),
    ]
    
    if intensity_range:
        transforms.append(
            M.ScaleIntensityRange(
                keys=["image"],
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
        )
    else:
        transforms.append(M.NormalizeIntensity(keys=["image"], nonzero=True))
    
    if augmentation:
        transforms.extend([
            M.RandCropByPosNegLabel(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=4,
            ),
            M.RandRotate90(keys=["image", "label"], prob=0.1, max_k=3),
            M.RandFlip(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            M.RandFlip(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            M.RandFlip(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            M.RandAffine(
                keys=["image", "label"],
                mode=["bilinear", "nearest"],
                prob=0.2,
                spatial_size=spatial_size,
                rotate_range=[0.0, 0.0, np.pi/15],
                scale_range=[0.1, 0.1, 0.1],
            ),
            M.RandScaleIntensity(keys=["image"], factors=0.1, prob=0.1),
            M.RandAdjustContrast(keys=["image"], prob=0.1),
            M.RandGaussianNoise(keys=["image"], prob=0.1, std=0.1),
        ])
    
    transforms.append(M.EnsureType(keys=["image", "label"], data_type="tensor"))
    
    return Compose(transforms)
