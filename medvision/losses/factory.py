"""Loss function factory for creating loss functions from configuration."""

import torch
import torch.nn as nn
from typing import Dict, Any, Callable

from .dice import DiceLoss
from .focal import FocalLoss
from .combined import CombinedLoss
from .crossentropy import CrossEntropyLoss, WeightedCrossEntropyLoss


def get_loss_function(config: Dict[str, Any], num_classes: int = None) -> Callable:
    """
    Factory function to create loss function based on configuration.
    
    Args:
        config: Loss configuration dictionary
        num_classes: Number of classes for automatic one-hot encoding
        
    Returns:
        Loss function
    """
    loss_type = config["type"].lower()
    
    if loss_type == "dice":
        smooth = config.get("smooth", 1e-6)
        # Ensure smooth is a float
        if isinstance(smooth, str):
            smooth = float(smooth)
        return DiceLoss(
            smooth=smooth,
            reduction=config.get("reduction", "mean"),
        )
    elif loss_type == "focal":
        alpha = config.get("alpha", 0.25)
        gamma = config.get("gamma", 2.0)
        # Ensure numeric types
        if isinstance(alpha, str):
            alpha = float(alpha)
        if isinstance(gamma, str):
            gamma = float(gamma)
        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=config.get("reduction", "mean"),
            num_classes=num_classes
        )
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(
            reduction=config.get("reduction", "mean")
        )
    elif loss_type == "ce" or loss_type == "crossentropy":
        weight = config.get("weight", None)
        if weight is not None and isinstance(weight, list):
            weight = torch.tensor(weight)
        
        return CrossEntropyLoss(
            weight=weight,
            ignore_index=config.get("ignore_index", -100),
            reduction=config.get("reduction", "mean"),
            label_smoothing=config.get("label_smoothing", 0.0),
            num_classes=num_classes
        )
    elif loss_type == "weighted_ce" or loss_type == "weighted_crossentropy":
        class_weights = config.get("class_weights", None)
        if class_weights is not None and isinstance(class_weights, list):
            class_weights = torch.tensor(class_weights)
            
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            ignore_index=config.get("ignore_index", -100),
            reduction=config.get("reduction", "mean")
        )
    elif loss_type == "combined":
        losses = []
        weights = []
        
        for loss_config in config["losses"]:
            sub_loss = get_loss_function(loss_config, num_classes=num_classes)
            weight = loss_config.get("weight", 1.0)
            if isinstance(weight, str):
                weight = float(weight)
            
            losses.append(sub_loss)
            weights.append(weight)
        
        return CombinedLoss(losses, weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
