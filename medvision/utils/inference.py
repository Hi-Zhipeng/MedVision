"""
Inference module for MedVision - Pure prediction without labels.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

from medvision.models import SegmentationModel
from medvision.transforms import get_transforms
from medvision.utils.io import load_image


class InferenceDataset(Dataset):
    """
    Dataset for inference - only loads images, no labels required.
    """
    
    def __init__(self, image_dir: str, transform=None, image_suffix="*.png"):
        """
        Initialize inference dataset.
        
        Args:
            image_dir: Directory containing images
            transform: Transform to apply to images
            image_suffix: File pattern for images
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Find all image files
        self.image_paths = sorted(glob.glob(str(self.image_dir / image_suffix)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir} with pattern {image_suffix}")
        
        print(f"Found {len(self.image_paths)} images for inference")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = load_image(image_path)
        
        # Apply transforms if provided
        if self.transform:
            # For inference, we only have image, no label
            data = {"image": image}
            transformed = self.transform(data)
            return transformed["image"], Path(image_path).name
        
        return image, Path(image_path).name


def create_inference_dataloader(config: Dict[str, Any]) -> DataLoader:
    """
    Create dataloader for inference.
    
    Args:
        config: Inference configuration
        
    Returns:
        DataLoader for inference
    """
    # Get transforms - only for images
    transforms = get_transforms(config.get("transforms", {}))
    
    # Create dataset
    dataset = InferenceDataset(
        image_dir=config["image_dir"],
        transform=transforms,
        image_suffix=config.get("image_suffix", "*.png")
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True)
    )
    
    return dataloader


def save_predictions(predictions: List[torch.Tensor], 
                    filenames: List[str], 
                    output_dir: str,
                    save_format: str = "png"):
    """
    Save prediction results.
    
    Args:
        predictions: List of prediction tensors
        filenames: List of corresponding filenames
        output_dir: Output directory
        save_format: Save format ('png', 'npy', etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(predictions)} predictions to {output_dir}")
    
    for pred, filename in zip(predictions, filenames):
        # Convert prediction to numpy
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy()
        else:
            pred_np = pred
        
        # Remove batch dimension if present
        if pred_np.ndim == 4:
            pred_np = pred_np[0]  # Remove batch dim
        if pred_np.ndim == 3 and pred_np.shape[0] == 1:
            pred_np = pred_np[0]  # Remove channel dim
        
        # Apply threshold to get binary mask
        pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
        
        # Save based on format
        base_name = Path(filename).stem
        
        if save_format.lower() == "png":
            save_path = output_path / f"{base_name}_pred.png"
            Image.fromarray(pred_binary).save(save_path)
        elif save_format.lower() == "npy":
            save_path = output_path / f"{base_name}_pred.npy"
            np.save(save_path, pred_np)
        
        print(f"Saved: {save_path}")


def predict_model(config: Dict[str, Any]) -> None:
    """
    Run inference on images using trained model.
    
    Args:
        config: Configuration dictionary
    """
    print("Starting model inference...")
    
    # Set random seed for reproducibility
    pl.seed_everything(config.get("seed", 42))
    
    # Load model from checkpoint
    if "checkpoint_path" not in config:
        raise ValueError("checkpoint_path must be specified for inference")
    
    print(f"Loading model from: {config['checkpoint_path']}")
    model = SegmentationModel.load_from_checkpoint(
        config["checkpoint_path"],
        config=config["model"]
    )
    model.eval()
    
    # Create inference dataloader
    inference_config = config["inference"]
    dataloader = create_inference_dataloader(inference_config)
    
    # Create trainer for prediction
    trainer = pl.Trainer(
        devices=inference_config.get("devices", 1),
        accelerator=inference_config.get("accelerator", "auto"),
        precision=inference_config.get("precision", 32),
        logger=False,  # Disable logging for inference
        enable_checkpointing=False,  # Disable checkpointing
        enable_progress_bar=True,
    )
    
    # Run predictions
    print("Running predictions...")
    predictions = trainer.predict(model, dataloader)
    
    # Flatten predictions and collect filenames
    all_predictions = []
    all_filenames = []
    
    for batch_preds in predictions:
        if isinstance(batch_preds, tuple):
            preds, filenames = batch_preds
        else:
            preds = batch_preds
            filenames = [f"pred_{i}.png" for i in range(len(preds))]
        
        # Handle batch predictions
        if isinstance(preds, torch.Tensor):
            for i in range(preds.shape[0]):
                all_predictions.append(preds[i])
                all_filenames.append(filenames[i] if i < len(filenames) else f"pred_{i}.png")
        else:
            all_predictions.extend(preds)
            all_filenames.extend(filenames)
    
    # Save predictions
    output_dir = inference_config.get("output_dir", "outputs/predictions")
    save_format = inference_config.get("save_format", "png")
    
    save_predictions(all_predictions, all_filenames, output_dir, save_format)
    
    print(f"Inference completed. Results saved to: {output_dir}")
    print(f"Total predictions: {len(all_predictions)}")
