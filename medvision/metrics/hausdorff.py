"""Hausdorff Distance metric for medical image segmentation."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class HausdorffDistance:
    """
    Hausdorff Distance metric for image segmentation evaluation.
    
    Measures the maximum distance from a point in one set to the nearest point in the other set.
    """
    
    def __init__(self, threshold=0.5, percentile=95):
        """
        Initialize Hausdorff Distance metric.
        
        Args:
            threshold: Threshold to convert predicted probabilities to binary mask
            percentile: Use percentile Hausdorff distance (e.g., 95th percentile)
        """
        self.threshold = threshold
        self.percentile = percentile
    
    def __call__(self, inputs, targets):
        """
        Calculate Hausdorff Distance.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Hausdorff Distance
        """
        # Apply sigmoid for binary classification
        if inputs.size(1) == 1:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = F.softmax(inputs, dim=1)
        
        # Apply threshold
        inputs = (inputs > self.threshold).float()
        
        # Convert to numpy
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        batch_size = inputs_np.shape[0]
        distances = []
        
        for b in range(batch_size):
            # Get contour points for prediction and target
            pred_points = self._get_contour_points(inputs_np[b])
            target_points = self._get_contour_points(targets_np[b])
            
            if len(pred_points) == 0 or len(target_points) == 0:
                # If no contour found, use maximum possible distance
                distances.append(float('inf'))
                continue
            
            # Calculate bidirectional Hausdorff distance
            dist1 = directed_hausdorff(pred_points, target_points)[0]
            dist2 = directed_hausdorff(target_points, pred_points)[0]
            
            # Use maximum or percentile
            if self.percentile == 100:
                hd = max(dist1, dist2)
            else:
                # Calculate percentile Hausdorff distance
                all_dists = []
                for p in pred_points:
                    min_dist = min(np.linalg.norm(p - t) for t in target_points)
                    all_dists.append(min_dist)
                for t in target_points:
                    min_dist = min(np.linalg.norm(t - p) for p in pred_points)
                    all_dists.append(min_dist)
                
                hd = np.percentile(all_dists, self.percentile)
            
            distances.append(hd)
        
        # Filter out infinite distances and return mean
        finite_distances = [d for d in distances if np.isfinite(d)]
        
        if len(finite_distances) == 0:
            return torch.tensor(float('inf'))
        
        return torch.tensor(np.mean(finite_distances))
    
    def _get_contour_points(self, mask):
        """Extract contour points from binary mask."""
        if mask.ndim == 4:  # [C, D, H, W]
            mask = mask[0, 0]  # Take first channel and slice
        elif mask.ndim == 3:  # [C, H, W] or [D, H, W]
            mask = mask[0] if mask.shape[0] <= 3 else mask[mask.shape[0]//2]
        
        # Simple edge detection
        mask_int = mask.astype(np.uint8)
        
        # Get boundary pixels
        from scipy import ndimage
        boundary = mask_int - ndimage.binary_erosion(mask_int)
        
        # Get coordinates of boundary pixels
        coords = np.where(boundary)
        if len(coords[0]) == 0:
            return np.array([])
        
        return np.column_stack(coords)
