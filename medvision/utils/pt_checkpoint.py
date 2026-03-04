"""
Custom callback to save .pt format in Lightning
"""
import os
import torch
from pytorch_lightning.callbacks import Callback


class PTCheckpoint(Callback):
    """Save .pt format instead of .ckpt"""

    def __init__(self, dirpath, filename, monitor="val/val_loss", mode="min"):
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
        os.makedirs(dirpath, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_best = (self.mode == 'min' and current < self.best_score) or \
                  (self.mode == 'max' and current > self.best_score)

        if is_best:
            self.best_score = current
            self.best_model_path = os.path.join(self.dirpath, f"{self.filename}.pt")

            checkpoint = {
                'epoch': trainer.current_epoch,
                'model_state_dict': pl_module.state_dict(),
                'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
                'best_score': float(self.best_score),
                'config': pl_module.config
            }
            torch.save(checkpoint, self.best_model_path)
            torch.save(pl_module.net.state_dict(), os.path.join(self.dirpath, f"{self.filename}_model.pt"))
