"""
I/O utilities for medical image loading and saving.
"""

import os
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

import torch

# Try to import common medical image libraries
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def load_image(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load medical image from file.
    
    Supported formats:
    - NIfTI (.nii, .nii.gz) via nibabel or SimpleITK
    - DICOM via SimpleITK
    - PNG, JPG, etc. via PIL/SimpleITK
    
    Args:
        path: Path to image file
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Handle different file formats
    if suffix in ['.nii', '.gz']:
        if HAS_NIBABEL:
            return _load_nifti_nibabel(path, return_meta)
        elif HAS_SITK:
            return _load_sitk(path, return_meta)
        else:
            raise ImportError("Neither nibabel nor SimpleITK is installed. "
                              "Please install one of them to load NIfTI files.")
    
    elif suffix in ['.dcm', '.dicom'] or (path.is_dir() and any(p.suffix.lower() == '.dcm' for p in path.glob('*'))):
        if HAS_SITK:
            return _load_dicom_sitk(path, return_meta)
        else:
            raise ImportError("SimpleITK is not installed. "
                              "Please install it to load DICOM files.")
    
    # Other formats, try SimpleITK first
    elif HAS_SITK:
        return _load_sitk(path, return_meta)
    
    # Fallback to PIL/torchvision
    else:
        from PIL import Image
        from torchvision import transforms

        img = Image.open(str(path))
        tensor = transforms.ToTensor()(img)

        if return_meta:
            meta = {
                "spacing": (1.0, 1.0),
                "origin": (0.0, 0.0),
                "direction": (1.0, 0.0, 0.0, 1.0),
                "size": tensor.shape,
                "format": img.format,
                "mode": img.mode,
            }
            return tensor, meta

        return tensor


def _load_nifti_nibabel(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load NIfTI file using nibabel.
    
    Args:
        path: Path to NIfTI file
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    import nibabel as nib
    
    # Load NIfTI file
    nii_img = nib.load(str(path))
    
    # Get image data as numpy array
    img_np = nii_img.get_fdata().astype(np.float32)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_np)
    
    # Add channel dimension if needed
    if len(img_tensor.shape) == 3:  # 3D volume
        img_tensor = img_tensor.unsqueeze(0)  # [C, D, H, W]
    elif len(img_tensor.shape) == 2:  # 2D image
        img_tensor = img_tensor.unsqueeze(0)  # [C, H, W]
    
    if return_meta:
        # Get metadata
        meta = {
            "affine": nii_img.affine,
            "header": nii_img.header,
            "spacing": nii_img.header.get_zooms(),
            "shape": img_np.shape,
        }
        return img_tensor, meta
    
    return img_tensor


def _load_sitk(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load image file using SimpleITK.
    
    Args:
        path: Path to image file
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    import SimpleITK as sitk
    
    # Load image
    img = sitk.ReadImage(str(path))

    # Convert to numpy array
    img_np = sitk.GetArrayFromImage(img)
    
    # SimpleITK uses [z,y,x] ordering, convert to [x,y,z]
    img_np = np.transpose(img_np)

    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_np.astype(np.float32))

    # # Load image
    # img = sitk.ReadImage(str(path))
    
    # # Convert to numpy array: shape [z, y, x]
    # img_np = sitk.GetArrayFromImage(img)  # shape: [D, H, W]
    
    # # Ensure it's at least 3D
    # if img_np.ndim == 2:
    #     img_np = np.expand_dims(img_np, axis=0)  # → [1, H, W]
    # elif img_np.ndim == 3:
    #     img_np = np.expand_dims(img_np, axis=0)  # → [1, D, H, W]（用于3D网络）

    # Convert to torch tensor (float32)
    # img_tensor = torch.from_numpy(img_np.astype(np.float32))
    
    # Add channel dimension if needed
    if len(img_tensor.shape) == 3:  # 3D volume
        img_tensor = img_tensor.unsqueeze(0)  # [C, D, H, W]
    elif len(img_tensor.shape) == 2:  # 2D image
        img_tensor = img_tensor.unsqueeze(0)  # [C, H, W]

    if return_meta:
        # Get metadata
        meta = {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
            "size": img.GetSize(),
            "pixel_type": img.GetPixelIDTypeAsString(),
            "number_of_components": img.GetNumberOfComponentsPerPixel(),
        }
        return img_tensor, meta
    
    return img_tensor


def _load_dicom_sitk(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load DICOM file(s) using SimpleITK.
    
    Args:
        path: Path to DICOM file or directory containing DICOM files
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    import SimpleITK as sitk
    
    path = Path(path)
    
    # Check if path is directory
    if path.is_dir():
        # Get DICOM reader
        reader = sitk.ImageSeriesReader()
        
        # Get DICOM series IDs
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {path}")
        
        # Use first series
        series_id = series_ids[0]
        
        # Get file names for series
        dicom_files = reader.GetGDCMSeriesFileNames(str(path), series_id)
        
        # Set file names and read
        reader.SetFileNames(dicom_files)
        img = reader.Execute()
    else:
        # Read single DICOM file
        img = sitk.ReadImage(str(path))
    
    # Convert to numpy array
    img_np = sitk.GetArrayFromImage(img)
    
    # SimpleITK uses [z,y,x] ordering, convert to [x,y,z]
    if len(img_np.shape) == 3:
        img_np = np.transpose(img_np, (2, 1, 0))
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_np.astype(np.float32))
    
    # Add channel dimension if needed
    if len(img_tensor.shape) == 3:  # 3D volume
        img_tensor = img_tensor.unsqueeze(0)  # [C, D, H, W]
    elif len(img_tensor.shape) == 2:  # 2D image
        img_tensor = img_tensor.unsqueeze(0)  # [C, H, W]
    
    if return_meta:
        # Get metadata
        meta = {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
            "size": img.GetSize(),
            "pixel_type": img.GetPixelIDTypeAsString(),
            "number_of_components": img.GetNumberOfComponentsPerPixel(),
        }
        
        # Get DICOM-specific metadata
        if hasattr(reader, "GetMetaDataKeys"):
            meta["dicom"] = {}
            for key in reader.GetMetaDataKeys(0):
                meta["dicom"][key] = reader.GetMetaData(0, key)
                
        return img_tensor, meta
    
    return img_tensor


def save_image(
    tensor: torch.Tensor,
    path: Union[str, Path],
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save image tensor to file.
    
    Supported formats:
    - NIfTI (.nii, .nii.gz)
    - Others via SimpleITK or PIL
    
    Args:
        tensor: Image tensor
        path: Output path
        meta: Metadata to include in the file
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    # Create directory if needed
    os.makedirs(path.parent, exist_ok=True)
    
    # Handle different file formats
    if suffix in ['.nii', '.gz']:
        if HAS_NIBABEL:
            _save_nifti_nibabel(tensor, path, meta)
        elif HAS_SITK:
            _save_sitk(tensor, path, meta)
        else:
            raise ImportError("Neither nibabel nor SimpleITK is installed. "
                              "Please install one of them to save NIfTI files.")
    
    # Other formats via SimpleITK
    elif HAS_SITK:
        _save_sitk(tensor, path, meta)
    
    # Fallback to PIL/torchvision
    else:
        from PIL import Image
        from torchvision import transforms
        
        # Convert tensor to PIL image
        if tensor.ndim == 4:  # 3D volume, save first slice
            tensor = tensor[0, 0]
        elif tensor.ndim == 3 and tensor.shape[0] > 3:  # Multiple channels, save first channel
            tensor = tensor[0]
            
        # Normalize to [0, 255]
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            
        tensor = tensor * 255
        tensor = tensor.to(torch.uint8)
        
        if tensor.ndim == 2:
            img = Image.fromarray(tensor.numpy())
        else:
            img = transforms.ToPILImage()(tensor)
            
        img.save(str(path))


def _save_nifti_nibabel(
    tensor: torch.Tensor,
    path: Union[str, Path],
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save tensor as NIfTI file using nibabel.
    
    Args:
        tensor: Image tensor
        path: Output path
        meta: Metadata to include in the file
    """
    import nibabel as nib
    
    # Convert tensor to numpy array
    array = tensor.detach().cpu().numpy()
    
    # Remove channel dimension if 1
    if array.shape[0] == 1:
        array = array[0]
    
    # Create affine matrix from metadata or default
    affine = meta["affine"] if meta and "affine" in meta else np.eye(4)
    
    # Create NIfTI image
    nii_img = nib.Nifti1Image(array, affine)
    
    # Update header with metadata
    if meta and "header" in meta:
        for key, value in meta["header"].items():
            if hasattr(nii_img.header, key):
                setattr(nii_img.header, key, value)
    
    # Save image
    nib.save(nii_img, str(path))


def _save_sitk(
    tensor: torch.Tensor,
    path: Union[str, Path],
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save tensor as image file using SimpleITK.
    
    Args:
        tensor: Image tensor
        path: Output path
        meta: Metadata to include in the file
    """
    import SimpleITK as sitk
    
    # Convert tensor to numpy array
    array = tensor.detach().cpu().numpy()
    
    # SimpleITK expects [z,y,x] ordering for 3D, [y,x] for 2D
    if array.ndim == 4:  # [C, D, H, W]
        array = array[0]  # Remove channel dimension
        array = np.transpose(array, (2, 1, 0))  # [W, H, D] -> [D, H, W]
    elif array.ndim == 3 and array.shape[0] <= 3:  # [C, H, W] for RGB/grayscale
        array = np.transpose(array, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    
    # Create SimpleITK image
    img = sitk.GetImageFromArray(array)
    
    # Apply metadata if available
    if meta:
        if "spacing" in meta:
            img.SetSpacing(meta["spacing"])
            
        if "origin" in meta:
            img.SetOrigin(meta["origin"])
            
        if "direction" in meta:
            img.SetDirection(meta["direction"])
    
    # Save image
    sitk.WriteImage(img, str(path))
