import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCoefficient:
    """
    Computes Dice coefficient for segmentation.
    
    Dice = 2 * |Prediction ∩ GroundTruth| / (|Prediction| + |GroundTruth|)
    
    Supports 2D ([B, C, H, W]) and 3D ([B, C, D, H, W]) tensors.
    
    Note: Background class (class 0) is excluded from Dice calculation.
    """

    def __init__(self, smooth: float = 1e-6, threshold: float = 0.5, per_class: bool = False):
        """
        Args:
            smooth: Small constant to avoid division by zero
            threshold: Threshold for converting probabilities to binary predictions
            per_class: If True, returns Dice per foreground class; otherwise returns mean Dice over foreground classes
        """

        self.smooth = smooth
        self.threshold = threshold
        self.per_class = per_class

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # Activation: sigmoid for binary, softmax for multi-class
        probs = torch.sigmoid(inputs) if inputs.size(1) == 1 else F.softmax(inputs, dim=1)
        
        preds = torch.argmax(probs, dim=1)

        preds = F.one_hot(preds, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float() if preds.dim() == 3 else F.one_hot(preds, num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        targets = nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float() if targets.dim() == 3 else nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        # Flatten spatial dimensions
        preds_flat = preds.view(preds.size(0), preds.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        # Compute Dice
        intersection = (preds_flat * targets_flat).sum(dim=2)
        union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # if self.per_class:
        #     return dice.mean(dim=0)[1:]
        # else:
        #     return dice[:, 1:].mean() 
        if self.per_class:
            return dice[:, 1:].mean(dim=0)
        else:
            return dice[:, 1:].mean()