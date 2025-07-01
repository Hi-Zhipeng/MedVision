"""Metric factory for creating metrics from configuration."""

from typing import Dict, Any, Callable

from .dice import DiceCoefficient
from .iou import IoU
from .accuracy import Accuracy
from .hausdorff import HausdorffDistance


def get_metrics(config: Dict[str, Any]) -> Dict[str, Callable]:
    """
    Factory function to create metrics based on configuration.
    
    Args:
        config: Metrics configuration dictionary
        
    Returns:
        Dictionary of metric functions
    """
    metrics = {}
    
    for metric_name, metric_config in config.items():
        metric_type = metric_config["type"].lower()
        if metric_type == "dice":
            metrics[metric_name] = DiceCoefficient(
                smooth=metric_config.get("smooth", 1e-6),
                threshold=metric_config.get("threshold", 0.5),
                per_class=metric_config.get("per_class", False)
            )
        elif metric_type == "iou":
            metrics[metric_name] = IoU(
                smooth=metric_config.get("smooth", 1e-6),
                threshold=metric_config.get("threshold", 0.5),
                per_class=metric_config.get("per_class", False)
            )
        elif metric_type == "accuracy":
            metrics[metric_name] = Accuracy(
                threshold=metric_config.get("threshold", 0.5),
                per_class=metric_config.get("per_class", False)
            )
        elif metric_type == "hausdorff":
            metrics[metric_name] = HausdorffDistance(
                threshold=metric_config.get("threshold", 0.5),
                percentile=metric_config.get("percentile", 95)
            )
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
            
    return metrics
