"""
Training module for MedVision.
"""

import os
import torch
from typing import Dict, Any
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from medvision.models import get_model
from medvision.datasets import get_datamodule
from medvision.utils.onnx_utils import convert_models_to_onnx, generate_triton_config


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
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["training"]["output_dir"], "checkpoints"),
        filename=f"{config['training'].get('experiment_name')}",
        monitor=config["training"].get("monitor", "val_loss"),
        mode=config["training"].get("monitor_mode", "min"),
        save_top_k=config["training"].get("save_top_k", 1),
        save_last=False,
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
        name="logs",
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
        gradient_clip_val=config["training"].get("gradient_clip_val", 0.0)
    )
    
    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # 保留最好的Top k模型后，进行模型转换
    convert_to_onnx = config["training"].get("convert_to_onnx", config["training"].get("export_onnx", False))
    converted_models = []
    onnx_dir = None
    
    if convert_to_onnx:
        print("\n" + "="*50)
        print("Converting models to ONNX format...")
        print("="*50)
        try:
            converted_models, onnx_dir = convert_models_to_onnx(
                checkpoint_callback, 
                model.__class__, 
                config["model"], 
                datamodule
            )
            print(f"\n✓ ONNX conversion completed!")
            print(f"✓ ONNX models saved to: {onnx_dir}")
            print(f"✓ Successfully converted {len(converted_models)} models to ONNX format")
        except Exception as e:
            print(f"❌ ONNX conversion failed: {str(e)}")
        print("="*50 + "\n")
    

    train_results = trainer.logged_metrics

    test_results = trainer.test(model, datamodule=datamodule)
    
    
    print(f"Training completed. Model checkpoints saved at: {checkpoint_callback.dirpath}")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score:.4f}")


    save_metrics = config["training"].get("save_metrics", True)
    if save_metrics:
        import json

        # 过滤 callback_metrics，只保留 train/val 部分
        train_val_metrics = {
            k: float(v) for k, v in train_results.items()
            if isinstance(v, torch.Tensor) and (k.startswith("val/") or k.startswith("train/"))
        }

        # 处理 test 结果
        test_metrics = {
            k: float(v) for k, v in test_results[0].items()
        } if test_results else {}

        # 汇总并保存
        final_metrics = {
            "train_val_metrics": train_val_metrics,
            "test_metrics": test_metrics,
            "best_model_path": checkpoint_callback.best_model_path,
            "best_model_score": float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None else None,
            "monitor": config["training"].get("monitor", "val_loss"),
        }
        
        # 添加ONNX转换信息
        if convert_to_onnx and converted_models:
            final_metrics["onnx_conversion"] = {
                "converted_count": len(converted_models),
                "onnx_directory": onnx_dir,
                "models": converted_models
            }

        result_path = os.path.join(config["training"]["output_dir"], "results.json")
        with open(result_path, "w") as f:
            json.dump(final_metrics, f, indent=4)

        print(f"Final metrics saved to: {result_path}")
