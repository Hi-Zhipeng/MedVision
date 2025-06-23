#!/usr/bin/env python
"""
Command line interface for MedVision.
"""

import argparse
import sys
import yaml
from pathlib import Path

from medvision.utils.config import parse_config


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="MedVision - Medical Image Segmentation Framework"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", type=str, help="Path to config file")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a model")
    test_parser.add_argument("config", type=str, help="Path to config file")
    
    # Inference command
    inference_parser = subparsers.add_parser("predict", help="Run inference on images")
    inference_parser.add_argument("config", type=str, help="Path to inference config file")
    inference_parser.add_argument("--image_dir", type=str, help="Directory containing images to predict")
    inference_parser.add_argument("--output_dir", type=str, help="Output directory for predictions")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    # Load configuration
    try:
        config = parse_config(config_path)
    except Exception as e:
        print(f"Error parsing config file: {str(e)}")
        sys.exit(1)
        
    if args.command == "train":
        try:
            from medvision.utils.trainer import train_model
            print("开始训练流程...")
            print(f"配置内容: {yaml.dump(config, default_flow_style=False)}")
            train_model(config)
        except Exception as e:
            import traceback
            print(f"训练过程中发生错误: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            sys.exit(1)
    elif args.command == "test":
        try:
            from medvision.utils.evaluator import test_model
            print("开始测试流程...")
            test_model(config)
        except Exception as e:
            import traceback
            print(f"测试过程中发生错误: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            sys.exit(1)
    elif args.command == "predict":
        try:
            from medvision.utils.inference import predict_model
            print("开始推理流程...")
            
            # 如果命令行提供了参数，覆盖配置文件中的设置
            if args.image_dir:
                config["inference"]["image_dir"] = args.image_dir
            if args.output_dir:
                config["inference"]["output_dir"] = args.output_dir
                
            predict_model(config)
        except Exception as e:
            import traceback
            print(f"推理过程中发生错误: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
