"""
Model factory for MedVision.
"""

from typing import Dict, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from medvision.models.unet import UNet
from medvision.models.denseUnet import DenseUNet

from medvision.losses import get_loss_function
from medvision.metrics import get_metrics


def get_model(config: Dict[str, Any]) -> pl.LightningModule:
    """
    Factory function to create a model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        A LightningModule implementation
    """
    model_type = config["type"].lower()
    
    if model_type == "segmentation":
        model_class = SegmentationModel
    elif model_type == "custom":
        # Add your custom model implementation here
        raise NotImplementedError(f"Custom model type not implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_class(config)


class SegmentationModel(pl.LightningModule):
    """
    Base segmentation model that implements standard training and evaluation steps.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the segmentation model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create the network architecture
        self.net = self._create_network()
        
        # Get loss function
        self.loss_fn = get_loss_function(config["loss"])
        
        # Get metrics
        self.metrics = get_metrics(config.get("metrics", {}))

    def _create_network(self) -> nn.Module:
        """
        Create the network architecture based on configuration.
        
        Returns:
            Neural network module
        """
        network = self.config["network"]
        network_name = network.get("name", "unet").lower()

        if network_name == "unet":
            return UNet(
                in_channels=self.config.get("in_channels", 1),
                out_channels=self.config.get("out_channels", 1),
                features=self.config.get("features", [32, 64, 128, 256]),
                dropout=self.config.get("dropout", 0.0),
            )
        elif network_name == "denseunet":
            return DenseUNet(
                in_channels=self.config.get("in_channels", 1),
                out_channels=self.config.get("out_channels", 1),
                dropout=self.config.get("dropout", 0.0),
            )
        elif network_name == "unet3d":
            from medvision.models.Unet3d import UNet3D
            return UNet3D(
                in_channels=self.config.get("in_channels", 1),
                out_channels=self.config.get("out_channels", 1),
                base_channels=self.config.get("base_channels", 32),
            )
        raise ValueError(f"Unknown model name: {network_name}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.net(x)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer configuration
        """
        optim_config = self.config.get("optimizer", {"type": "adam", "lr": 1e-3})
        optim_type = optim_config.get("type", "adam").lower()
        lr = optim_config.get("lr", 1e-3)
        
        # Create optimizer
        if optim_type == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=lr,
                weight_decay=optim_config.get("weight_decay", 0.0),
            )
        elif optim_type == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=lr,
                momentum=optim_config.get("momentum", 0.9),
                weight_decay=optim_config.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optim_type}")
        
        # Configure scheduler if specified
        scheduler_config = self.config.get("scheduler", None)
        if scheduler_config is None:
            return optimizer
        
        scheduler_type = scheduler_config.get("type", "plateau").lower()
        
        if scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=scheduler_config.get("factor", 0.1),
                patience=scheduler_config.get("patience", 10)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_config.get("monitor", "val_loss"),
                },
            }
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("T_max", 10),
                eta_min=scheduler_config.get("eta_min", 0),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        images, masks = batch

        logits = self(images)
        loss = self.loss_fn(logits, masks)
        
        # Log metrics
        self.log("train/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(logits, masks)
            self.log(f"train/train_{metric_name}", metric_value, on_step=False, on_epoch=True)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Validation dictionary
        """
        images, masks = batch

        logits = self(images)
        loss = self.loss_fn(logits, masks)
        
        # Log metrics
        self.log("val/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate metrics - let the metric functions handle dimension compatibility
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(logits, masks)
            self.log(f"val/val_{metric_name}", metric_value, on_step=False, on_epoch=True)
        
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Test dictionary
        """
        images, masks = batch

        logits = self(images)
        loss = self.loss_fn(logits, masks)
        
        # Log metrics
        self.log("test/test_loss", loss, on_step=False, on_epoch=True)
        
        results = {"test_loss": loss}
        
        # Calculate metrics - let the metric functions handle dimension compatibility
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(logits, masks)
            self.log(f"test/test_{metric_name}", metric_value, on_step=False, on_epoch=True)
            results[f"test_{metric_name}"] = metric_value
        
        return results
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference.
        
        Args:
            batch: Input batch (can be images only or (images, filenames))
            batch_idx: Batch index
            
        Returns:
            Predictions
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            # Case: (images, filenames)
            images, filenames = batch
            if isinstance(images, list):
                images = torch.stack(images)
        else:
            # Case: only images
            images = batch
            if isinstance(images, list):
                images = torch.stack(images)
            filenames = [f"pred_{batch_idx}_{i}.png" for i in range(images.shape[0])]
        
        # Forward pass
        logits = self(images)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)
        
        return probabilities, filenames

