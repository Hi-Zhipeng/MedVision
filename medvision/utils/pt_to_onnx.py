"""
Convert PyTorch .pt models to ONNX format
"""
import torch
import torch.onnx
from pathlib import Path
from typing import Tuple


def convert_pt_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    opset_version: int = 18
):
    """
    Convert .pt model to ONNX format

    Args:
        model_path: Path to .pt file
        output_path: Output ONNX file path
        input_shape: Model input shape (batch, channels, height, width)
        opset_version: ONNX opset version
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model architecture from config
    from medvision.models.model_factory import SegmentationModel
    model = SegmentationModel(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"✓ Converted {model_path} -> {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .pt model')
    parser.add_argument('--output', required=True, help='Output ONNX path')
    parser.add_argument('--input-shape', nargs=4, type=int, default=[1, 1, 256, 256])
    args = parser.parse_args()

    convert_pt_to_onnx(args.model, args.output, tuple(args.input_shape))
