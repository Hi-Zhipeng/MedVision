"""Focal Loss implementation for handling class imbalance."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize Focal loss.
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter
            reduction: Reduction mode ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calculate Focal loss.
        
        Args:
            inputs: Logits
            targets: Ground truth masks
            
        Returns:
            Focal loss
        """
        # Add channel dimension to targets if missing
        if targets.dim() == inputs.dim() - 1:
            targets = targets.unsqueeze(1)
        
        # Ensure targets are float and in [0, 1] range
        targets = targets.float()
        
        # Normalize targets to [0, 1] if they're not already
        if targets.max() > 1.0 or targets.min() < 0.0:
            targets = torch.clamp(targets, 0.0, 1.0)
            # If targets seem to be in 0-255 range, normalize them
            if targets.max() > 1.0:
                targets = targets / 255.0
        
        # For binary segmentation with single channel output
        if inputs.size(1) == 1:
            # Apply sigmoid to get probabilities
            inputs_prob = torch.sigmoid(inputs)
            
            # Flatten for loss calculation
            inputs_prob = inputs_prob.view(-1)
            targets = targets.view(-1)
            
            # Calculate BCE with logits (more numerically stable)
            bce = F.binary_cross_entropy_with_logits(
                inputs.view(-1), targets, reduction='none'
            )
            
            # Calculate pt for focal term
            with torch.no_grad():
                pt = targets * inputs_prob + (1 - targets) * (1 - inputs_prob)
                pt = torch.clamp(pt, 1e-8, 1.0 - 1e-8)  # Avoid numerical issues
            
            # Apply focal term
            focal_weight = (1 - pt).pow(self.gamma)
            
            # Apply alpha term
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            
            # Compute loss
            loss = focal_weight * alpha_weight * bce
        else:
            # Multi-class case
            # Apply log_softmax for numerical stability
            log_pt = F.log_softmax(inputs, dim=1)
            
            # Convert targets to class indices if needed
            if targets.size(1) == inputs.size(1):
                # Targets are already one-hot
                targets_class = targets.argmax(dim=1)
            else:
                # Targets are class indices
                targets_class = targets.squeeze(1).long()
            
            # Gather log probabilities for target classes
            log_pt = log_pt.gather(1, targets_class.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            
            # Apply focal term
            focal_weight = (1 - pt).pow(self.gamma)
            
            # Apply alpha (simplified for multi-class)
            loss = -self.alpha * focal_weight * log_pt
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
