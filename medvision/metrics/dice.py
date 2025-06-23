"""Dice coefficient metric for medical image segmentation."""

import torch
import torch.nn.functional as F


class DiceCoefficient:
    """
    Computes the Dice coefficient for evaluating segmentation performance.

    Dice = 2 * |Prediction ∩ GroundTruth| / (|Prediction| + |GroundTruth|)

    Supports both 2D ([B, C, H, W]) and 3D ([B, C, D, H, W]) image tensors.
    """

    def __init__(self, smooth: float = 1e-6, threshold: float = 0.5, per_class: bool = False):
        """
        Args:
            smooth (float): Smoothing term to avoid division by zero.
            threshold (float): Threshold to binarize prediction probabilities.
            per_class (bool): If True, return per-class Dice; else return mean Dice.
        """
        self.smooth = smooth
        self.threshold = threshold
        self.per_class = per_class

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice coefficient.

        Args:
            inputs (Tensor): Predicted logits. Shape: [B, C, H, W] or [B, C, D, H, W]
            targets (Tensor): Ground truth. Shape:
                              - Class labels: [B, H, W] or [B, D, H, W]
                              - One-hot/multi-channel masks: same shape as inputs

        Returns:
            Tensor: Dice score (scalar or per class)
        """
        is_3d = inputs.dim() == 5  # Check if input is 3D

        # Apply activation function
        if inputs.size(1) == 1:
            probs = torch.sigmoid(inputs)
        else:
            probs = F.softmax(inputs, dim=1)

        preds = (probs > self.threshold).float()

        # Convert targets to one-hot if necessary
        if targets.dim() == inputs.dim() - 1:
            targets = targets.unsqueeze(1)
        if targets.size(1) != inputs.size(1):
            num_classes = inputs.size(1)
            targets = F.one_hot(targets.long(), num_classes=num_classes)
            if is_3d:
                targets = targets.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
            else:
                targets = targets.permute(0, 3, 1, 2)      # [B, C, H, W]

        targets = targets.float()

        # Flatten spatial dimensions
        preds_flat = preds.view(preds.size(0), preds.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (preds_flat * targets_flat).sum(dim=2)
        union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return dice.mean(dim=0) if self.per_class else dice.mean()
