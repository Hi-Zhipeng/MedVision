"""Combined loss implementation for multiple loss functions."""

import torch
import torch.nn as nn
from typing import List


class CombinedLoss(nn.Module):
    """
    Combine multiple losses with weighting factors.
    """
    
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        """
        Initialize combined loss.
        
        Args:
            losses: List of loss functions
            weights: List of weights for each loss
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
        assert len(losses) == len(weights), "Number of losses must match number of weights"
    
    def forward(self, inputs, targets):
        """
        Calculate combined loss.
        
        Args:
            inputs: Predictions
            targets: Ground truth
            
        Returns:
            Weighted sum of losses
        """
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(inputs, targets)
        return total_loss
