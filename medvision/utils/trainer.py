"""
Training module for MedVision.
"""

import os
import torch
from typing import Dict, Any
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from medvision.models import get_model
from medvision.datasets import get_datamodule


def train_model(config: Dict[str, Any]) -> None:
    """
    Train a model based on the provided configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("Starting training...")
    
    # Set random seed for reproducibility
    pl.seed_everything(config.get("seed", 42))
    
    # Create the model
    model = get_model(config["model"])
    
    # Create data module
    datamodule = get_datamodule(config["data"])
    
    # Configure callbacks
    callbacks = []
    
    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["training"]["output_dir"], "checkpoints"),
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor=config["training"].get("monitor", "val_loss"),
        mode=config["training"].get("monitor_mode", "min"),
        save_top_k=config["training"].get("save_top_k", 3),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # EarlyStopping callback
    if config["training"].get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor=config["training"].get("monitor", "val_loss"),
            mode=config["training"].get("monitor_mode", "min"),
            patience=config["training"].get("patience", 10),
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        
    # Configure logger
    logger = TensorBoardLogger(
        save_dir=config["training"]["output_dir"],
        name=config["training"].get("experiment_name", "medvision"),
        version=config["training"].get("version", None),
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"].get("max_epochs", 100),
        devices=config["training"].get("devices", None),
        accelerator=config["training"].get("accelerator", "auto"),
        precision=config["training"].get("precision", 32),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["training"].get("log_every_n_steps", 10),
        deterministic=config["training"].get("deterministic", False),
        gradient_clip_val=config["training"].get("gradient_clip_val", 0.0),
    )
    
    # Train the model
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)
    
    print(f"Training completed. Model checkpoints saved at: {checkpoint_callback.dirpath}")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score:.4f}")