"""Intersection over Union (IoU) metric for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IoU:
    """
    Computes the Intersection over Union (IoU) for image segmentation tasks.

    IoU = |Prediction ∩ GroundTruth| / |Prediction ∪ GroundTruth|
        = Intersection / (Union)
        = (|X ∩ Y|) / (|X| + |Y| - |X ∩ Y|)

    Supports 2D ([B, C, H, W]) and 3D ([B, C, D, H, W]) inputs.
    """

    def __init__(self, smooth: float = 1e-6, threshold: float = 0.5, per_class: bool = False):
        """
        Args:
            smooth (float): Smoothing constant to avoid division by zero.
            threshold (float): Threshold to binarize prediction probabilities.
            per_class (bool): If True, return per-class IoU; else return mean IoU.
        """
        self.smooth = smooth
        self.threshold = threshold
        self.per_class = per_class

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): Predicted logits of shape [B, C, H, W] or [B, C, D, H, W]
            targets (Tensor): Ground truth masks:
                - Label format: [B, H, W] or [B, D, H, W]
                - One-hot format: same as inputs

        Returns:
            Tensor: Mean IoU or per-class IoU depending on `per_class`
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs) if inputs.size(1) == 1 else F.softmax(inputs, dim=1)
        preds = (probs > self.threshold).float()

        targets = nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float() if targets.dim() == 3 else nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        # Flatten spatial dimensions
        preds_flat = preds.view(preds.size(0), preds.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (preds_flat * targets_flat).sum(dim=2)
        union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou.mean(dim=0) if self.per_class else iou.mean()