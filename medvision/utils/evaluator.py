"""
Evaluation module for MedVision.
"""

import os
from typing import Dict, Any
from pathlib import Path
import pytorch_lightning as pl

from medvision.models import get_model
from medvision.datasets import get_datamodule


def test_model(config: Dict[str, Any]) -> None:
    """
    Test a model based on the provided configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("Starting model evaluation...")
    
    # Set random seed for reproducibility
    pl.seed_everything(config.get("seed", 42))
    
    # Load checkpoint if specified
    if "checkpoint_path" in config:

        from medvision.models import SegmentationModel
        model = SegmentationModel.load_from_checkpoint(
            config["checkpoint_path"],
            config=config["model"]
        )
    else:
        print("Warning: No checkpoint path provided, using initialized model.")
        # Load the model normally
        model = get_model(config["model"])
    
    # Create data module
    datamodule = get_datamodule(config["data"])
    
    # Create trainer for testing
    trainer = pl.Trainer(
        devices=config["testing"].get("devices", None),
        accelerator=config["testing"].get("accelerator", "auto"),
        precision=config["testing"].get("precision", 32),
    )
    
    # Test the model
    results = trainer.test(model, datamodule=datamodule)
    
    # Save results if output directory is specified
    if "output_dir" in config["testing"]:
        output_dir = Path(config["testing"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Here you would implement saving of predictions, visualizations, etc.
        print(f"Results saved to: {output_dir}")
    
    print("Evaluation completed.")
