# """Dice Loss implementation for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs) if inputs.size(1) == 1 else F.softmax(inputs, dim=1)
        num_classes = probs.size(1)

        if targets.dim() == 3:
            targets = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        else:
            targets = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3).float()

        if probs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {probs.shape} vs {targets.shape}")

        probs = probs.flatten(2)
        targets = targets.flatten(2)

        intersection = (probs * targets).sum(dim=2)
        union = probs.sum(dim=2) + targets.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        if 0 <= self.ignore_index < dice.size(1):
            dice = dice[:, [i for i in range(dice.size(1)) if i != self.ignore_index]]

        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss