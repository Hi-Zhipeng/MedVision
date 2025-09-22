import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for medical image segmentation.
    Assumes targets are class indices (no one-hot needed).
    Works for binary (C=1) or multi-class segmentation.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', num_classes=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Apply sigmoid for binary, softmax for multi-class

        if inputs.size(1) == 1:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = F.softmax(inputs, dim=1)

        targets = nn.functional.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 4, 1, 2).float() if targets.dim() == 3 else nn.functional.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()


        # Flatten to [B, C, N]
        B, C = inputs.size()[:2]
        inputs = inputs.view(B, C, -1)
        targets = targets.view(B, C, -1)

        # Compute pt
        pt = (inputs * targets + (1 - inputs) * (1 - targets)).clamp(1e-8, 1.0)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Binary cross entropy
        bce = -(targets * torch.log(inputs.clamp(1e-8, 1.0)) + (1 - targets) * torch.log((1 - inputs).clamp(1e-8, 1.0)))
        loss = focal_weight * alpha_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss