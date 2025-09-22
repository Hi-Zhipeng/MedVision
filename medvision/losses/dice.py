"""Dice Loss implementation for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice coefficient loss for image segmentation.
    
    Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    
    def __init__(self, smooth=1e-6, reduction='mean'):
        """
        Initialize the Dice loss.
        
        Args:
            smooth: Small constant to avoid division by zero
            reduction: Reduction mode ('mean', 'sum', or 'none')
            num_classes: Number of classes for automatic one-hot encoding
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calculate Dice loss.
        
        Args:
            inputs: Predicted probabilities (logits)
            targets: Ground truth masks (can be class indices or one-hot)
            
        Returns:
            Dice loss
        """
     
        # Apply sigmoid/softmax
        if inputs.size(1) == 1:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = F.softmax(inputs, dim=1)

        targets = nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float() if targets.dim() == 3 else nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        # Ensure same shape
        assert inputs.shape == targets.shape, f"Shape mismatch: {inputs.shape} vs {targets.shape}"

        # Flatten to [B, C, N]
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Dice coefficient
        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
