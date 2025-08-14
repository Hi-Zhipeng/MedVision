"""
ONNX conversion utilities for MedVision.
"""

import os
import glob
import torch
from typing import Dict, Any, List, Tuple


def convert_models_to_onnx(
    checkpoint_callback, 
    model_class, 
    config: Dict[str, Any], 
    datamodule
) -> Tuple[List[Dict], str]:
    """
    将保存的top-k模型转换为ONNX格式
    
    Args:
        checkpoint_callback: ModelCheckpoint回调对象
        model_class: 模型类
        config: 配置字典
        datamodule: 数据模块
        
    Returns:
        Tuple[List[Dict], str]: 转换成功的模型列表和ONNX目录路径
    """
    # 配置参数
    opset_version = config.get("onnx_opset_version", 11)
    
    checkpoint_dir = checkpoint_callback.dirpath
    onnx_dir = os.path.join(os.path.dirname(checkpoint_dir), "onnx_models")
    os.makedirs(onnx_dir, exist_ok=True)
    
    # 获取所有检查点文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    # 获取示例输入
    datamodule.setup('fit')
    sample_batch = next(iter(datamodule.train_dataloader()))
    sample_input = sample_batch[0][:1]  # 取一个样本
    
    converted_models = []
    
    print(f"Found {len(checkpoint_files)} checkpoint files to convert...")
    
    # 检查示例输入的设备
    print(f"Sample input device: {sample_input.device}")
    
    for ckpt_path in checkpoint_files:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\nConverting {ckpt_name}...")
        
        try:
            # 加载模型
            model = model_class.load_from_checkpoint(ckpt_path, config=config)
            model.eval()
            
            # 检查模型参数的设备
            model_device = next(model.parameters()).device
            print(f"  Model loaded on device: {model_device}")
            
            # 将模型移动到CPU进行ONNX转换
            if model_device.type == 'cuda':
                print(f"  Moving model from {model_device} to CPU for ONNX export...")
                model = model.cpu()
            
            # 确保示例输入在CPU上
            sample_input_cpu = sample_input.cpu()
            print(f"  Using sample input on device: {sample_input_cpu.device}")
            
            # ONNX文件路径
            onnx_path = os.path.join(onnx_dir, f"{ckpt_name}.onnx")
            
            # 获取输入shape信息
            input_shape = sample_input_cpu.shape
            
            # 转换为ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model.net,
                    sample_input_cpu,
                    onnx_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    },
                    verbose=False
                )
            
            # 验证ONNX模型
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"✓ ONNX model validation passed: {ckpt_name}")
            except ImportError:
                print(f"⚠ ONNX validation skipped (onnx package not installed): {ckpt_name}")
            except Exception as e:
                print(f"⚠ ONNX validation failed: {ckpt_name}, error: {e}")
            
            converted_models.append({
                "checkpoint_path": ckpt_path,
                "onnx_path": onnx_path,
                "model_name": ckpt_name,
                "input_shape": list(input_shape),
                "original_device": str(model_device) if 'model_device' in locals() else "unknown"
            })
            
            print(f"  ✓ Successfully converted {ckpt_name} to ONNX")
            
        except Exception as e:
            print(f"  ❌ Failed to convert {ckpt_name}: {str(e)}")
            import traceback
            print(f"  Full error traceback:")
            traceback.print_exc()
    
    return converted_models, onnx_dir


def convert_single_model_to_onnx(
    checkpoint_path: str,
    model_class,
    config: Dict[str, Any],
    sample_input: torch.Tensor,
    output_path: str,
    opset_version: int = 11
) -> Dict[str, Any]:
    """
    将单个模型转换为ONNX格式
    
    Args:
        checkpoint_path: 模型检查点路径
        model_class: 模型类
        config: 模型配置字典
        sample_input: 示例输入张量
        output_path: 输出ONNX文件路径
        opset_version: ONNX opset版本
        
    Returns:
        Dict[str, Any]: 转换结果信息
    """
    try:
        # 加载模型
        model = model_class.load_from_checkpoint(checkpoint_path, config=config)
        model.eval()
        
        # 检查模型参数的设备
        model_device = next(model.parameters()).device
        
        # 将模型移动到CPU进行ONNX转换
        if model_device.type == 'cuda':
            model = model.cpu()
        
        # 确保示例输入在CPU上
        sample_input_cpu = sample_input.cpu()
        
        # 获取输入shape信息
        input_shape = sample_input_cpu.shape
        
        # 转换为ONNX
        with torch.no_grad():
            torch.onnx.export(
                model.net,
                sample_input_cpu,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        # 验证ONNX模型
        validation_passed = False
        validation_error = None
        
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            validation_passed = True
        except ImportError:
            validation_error = "ONNX package not installed"
        except Exception as e:
            validation_error = str(e)
        
        return {
            "success": True,
            "checkpoint_path": checkpoint_path,
            "onnx_path": output_path,
            "input_shape": list(input_shape),
            "validation_passed": validation_passed,
            "validation_error": validation_error
        }
        
    except Exception as e:
        return {
            "success": False,
            "checkpoint_path": checkpoint_path,
            "onnx_path": output_path,
            "error": str(e)
        }


def validate_onnx_model(onnx_path: str) -> Tuple[bool, str]:
    """
    验证ONNX模型的有效性
    
    Args:
        onnx_path: ONNX模型文件路径
        
    Returns:
        Tuple[bool, str]: (是否验证通过, 错误信息)
    """
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        return True, "Validation passed"
    except ImportError:
        return False, "ONNX package not installed"
    except Exception as e:
        return False, str(e)


def get_onnx_model_info(onnx_path: str) -> Dict[str, Any]:
    """
    获取ONNX模型的信息
    
    Args:
        onnx_path: ONNX模型文件路径
        
    Returns:
        Dict[str, Any]: 模型信息字典
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        # 获取输入信息
        inputs = []
        for input_tensor in model.graph.input:
            input_info = {
                "name": input_tensor.name,
                "type": input_tensor.type.tensor_type.elem_type,
                "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            }
            inputs.append(input_info)
        
        # 获取输出信息
        outputs = []
        for output_tensor in model.graph.output:
            output_info = {
                "name": output_tensor.name,
                "type": output_tensor.type.tensor_type.elem_type,
                "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            }
            outputs.append(output_info)
        
        return {
            "file_path": onnx_path,
            "file_size": os.path.getsize(onnx_path),
            "opset_version": model.opset_import[0].version if model.opset_import else None,
            "inputs": inputs,
            "outputs": outputs,
            "node_count": len(model.graph.node)
        }
        
    except ImportError:
        return {"error": "ONNX package not installed"}
    except Exception as e:
        return {"error": str(e)}
