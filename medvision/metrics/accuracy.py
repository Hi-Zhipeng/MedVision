"""Pixel-wise accuracy metric for medical image segmentation."""

import torch
import torch.nn.functional as F


class Accuracy:
    """
    Computes pixel-wise accuracy for segmentation tasks.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Supports both 2D ([B, C, H, W]) and 3D ([B, C, D, H, W]) inputs.
    """

    def __init__(self, threshold: float = 0.5, per_class: bool = False):
        """
        Args:
            threshold (float): Threshold for binarizing binary predictions.
            per_class (bool): If True, return per-class accuracy for multi-class segmentation.
        """
        self.threshold = threshold
        self.per_class = per_class

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): Predicted logits, shape [B, C, H, W] or [B, C, D, H, W]
            targets (Tensor): Ground truth, shape:
                - Class labels: [B, H, W] or [B, D, H, W]
                - One-hot/multi-channel: same shape as inputs

        Returns:
            Tensor: Scalar accuracy or per-class accuracy vector
        """
        # --- Step 1: Convert predictions ---
        if inputs.size(1) == 1:
            preds = (torch.sigmoid(inputs) > self.threshold).float()
        else:
            preds = torch.argmax(F.softmax(inputs, dim=1), dim=1, keepdim=True).float()

        # --- Step 2: Prepare targets ---
        if targets.dim() == inputs.dim() - 1:
            targets = targets.unsqueeze(1)
        if targets.size(1) != 1:
            # If one-hot, convert to class index
            targets = torch.argmax(targets, dim=1, keepdim=True).float()
        else:
            targets = targets.float()

        # --- Step 3: Compute accuracy ---
        correct = (preds == targets).float()

        if self.per_class and inputs.size(1) > 1:
            # Compute per-class accuracy
            num_classes = int(targets.max().item()) + 1
            class_accuracies = []
            for c in range(num_classes):
                mask = (targets == c)
                total = mask.sum()
                if total > 0:
                    acc = ((preds == c) & mask).sum() / total
                else:
                    acc = torch.tensor(0.0, device=inputs.device)
                class_accuracies.append(acc)
            return torch.stack(class_accuracies)
        else:
            return correct.mean()