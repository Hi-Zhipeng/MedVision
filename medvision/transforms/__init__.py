"""Transforms module for MedVision."""

from typing import Dict, Any, Optional, Callable
from .monai_transforms import get_transforms as get_monai_transforms
from .color_transforms import get_color_transforms


def get_transforms(config: Dict[str, Any]) -> Optional[Callable]:
    """
    Factory function to create transforms based on configuration.

    Args:
        config: Transform configuration dictionary

    Returns:
        Transform function or None if no transforms specified
    """
    if not config:
        return None

    # Check if this is torchvision-style transforms (for color images)
    if "Compose" in config:
        return get_color_transforms(config)
    else:
        # Default to MONAI transforms (for medical images)
        return get_monai_transforms(config)
