"""Cross Entropy Loss implementation for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss for image segmentation.
    
    Supports both binary and multi-class segmentation.
    """
    
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, num_classes=None):
        """
        Initialize the Cross Entropy loss.
        
        Args:
            weight: Manual rescaling weight given to each class
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Reduction mode ('mean', 'sum', or 'none')
            label_smoothing: Label smoothing factor (0.0 means no smoothing)
            num_classes: Number of classes for automatic one-hot encoding
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        """
        Calculate Cross Entropy loss.
        
        Args:
            inputs: Predicted logits [B, C, H, W] or [B, C, D, H, W]
            targets: Ground truth masks [B, H, W] or [B, D, H, W] (class indices)
            
        Returns:
            Cross Entropy loss
        """
        # For binary segmentation, use BCEWithLogitsLoss
        if inputs.size(1) == 1:
            # Convert targets to same shape as inputs and ensure float type
            if targets.dim() == inputs.dim() - 1:
                targets = targets.unsqueeze(1).float()
            else:
                targets = targets.float()
            
            # Use BCEWithLogitsLoss for binary case
            bce_loss = nn.BCEWithLogitsLoss(
                weight=self.weight,
                reduction=self.reduction
            )
            return bce_loss(inputs, targets)
        
        # For multi-class segmentation
        else:
            # Ensure targets are long type for multi-class
            targets = targets.long()
            
            # Remove channel dimension from targets if present
            if targets.dim() == inputs.dim():
                targets = targets.squeeze(1)
            
            # Use CrossEntropyLoss for multi-class case
            ce_loss = nn.CrossEntropyLoss(
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing
            )
            return ce_loss(inputs, targets)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy loss for handling class imbalance.
    """
    
    def __init__(self, class_weights=None, ignore_index=-100, reduction='mean'):
        """
        Initialize the Weighted Cross Entropy loss.
        
        Args:
            class_weights: Tensor of weights for each class
            ignore_index: Specifies a target value that is ignored
            reduction: Reduction mode ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calculate Weighted Cross Entropy loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Weighted Cross Entropy loss
        """
        # Convert class_weights to tensor if needed
        if self.class_weights is not None and not isinstance(self.class_weights, torch.Tensor):
            self.class_weights = torch.tensor(self.class_weights, device=inputs.device, dtype=inputs.dtype)
        
        # For binary segmentation
        if inputs.size(1) == 1:
            if targets.dim() == inputs.dim() - 1:
                targets = targets.unsqueeze(1).float()
            else:
                targets = targets.float()
            
            # Calculate positive weight for BCEWithLogitsLoss
            pos_weight = None
            if self.class_weights is not None and len(self.class_weights) == 2:
                pos_weight = self.class_weights[1] / self.class_weights[0]
            
            bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction=self.reduction
            )
            return bce_loss(inputs, targets)
        
        # For multi-class segmentation
        else:
            targets = targets.long()
            if targets.dim() == inputs.dim():
                targets = targets.squeeze(1)
            
            ce_loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )
            return ce_loss(inputs, targets)
