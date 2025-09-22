import torch
import torch.nn as nn
import torch.nn.functional as F


class Accuracy:
    """
    Computes pixel-wise accuracy for segmentation.

    Accuracy = (# correctly predicted pixels) / (total # of pixels)

    Supports both 2D ([B, C, H, W]) and 3D ([B, C, D, H, W]) inputs.
    """

    def __init__(self, threshold: float = 0.5, per_class: bool = False):
        """
        Args:
            threshold (float): Threshold for binarizing predictions in binary segmentation.
            per_class (bool): If True, return per-class accuracy.
        """
        self.threshold = threshold
        self.per_class = per_class


    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits, shape [B, C, H, W] or [B, C, D, H, W]
            targets: Ground truth labels, shape [B, H, W] or [B, D, H, W] (class indices)
        
        Returns:
            Scalar accuracy or per-class accuracy vector
        """

        preds = (torch.sigmoid(inputs) > self.threshold).float() if inputs.size(1) == 1 else torch.argmax(F.softmax(inputs, dim=1), dim=1, keepdim=True).float()

        targets = nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float() if targets.dim() == 3 else nn.functional.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        preds = preds.view(preds.size(0), preds.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        correct = (preds == targets).float()

        if self.per_class and inputs.size(1) > 1:
            # 每类正确像素 / 每类总像素
            correct_per_class = (preds * targets).sum(dim=2)  # [B, C]
            total_per_class = targets.sum(dim=2)              # [B, C]
            class_acc = (correct_per_class / (total_per_class + 1e-6)).mean(dim=0)  # [C]
            return class_acc
        else:
            correct = (preds == targets).float()
            return correct.mean()