# """Dice Loss implementation for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=0):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index  

    def forward(self, inputs, targets):
        if inputs.size(1) == 1:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = F.softmax(inputs, dim=1)

        if targets.dim() == 3:  # [B,H,W]
            targets = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        else:  # [B,D,H,W]
            targets = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        assert inputs.shape == targets.shape, f"Shape mismatch: {inputs.shape} vs {targets.shape}"

        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        mask = torch.ones(dice.size(1), device=dice.device, dtype=torch.bool)
        mask[self.ignore_index] = False
        dice = dice[:, mask]
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss