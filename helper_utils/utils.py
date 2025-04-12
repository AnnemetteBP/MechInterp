from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def make_dir(directory:str) -> str:
    """ Make directory for e.g., stroing models and versions """

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} was created!")
    else:
        print(f"Directory {directory} already exists!")

    return directory


def save_model(model:Any, save_path:str) -> None:
    """ Save model to load as pretrained """
    
    model.save_pretrained(save_path, safe_serialization=True)
    print(f"Model saved to {save_path}")


def save_tokenizer(tokenizer, save_path:str) -> None:
    """ Save tokenizer to load as pretrained """

    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")


def measure_time(func, *args, **kwargs):
    """ Measures execution time of any function """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    print(f"Execution Time for {func.__name__}: {elapsed_time:.4f} seconds")
    
    return result, elapsed_time


def check_quant_config(model:Any) -> None:
    """ Check if the model has the attribute indicating 8-bit or 4-bit quantization """

    if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
        print("The model is loaded in 8-bit")
    elif hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        print("The model is loaded in 4-bit")
    else:
        print("The model is not loaded in 8-bit or 4-bit")


def check_bnb_config(model:Any) -> bool:
    """ Check for the presence of a bnb_config attribute """

    if hasattr(model, 'bnb_config'):
        print("The model has a bnb_config attribute")
        return True
    else:
        print("The model does not have a bnb_config attribute")
        return False


def check_device_mapping(model:Any) -> None:
    """ Check if the model is using device_map (common in `bitsandbytes` quantized models) """

    if hasattr(model, 'device_map'):
        print(f"Device map: {model.device_map}")
    else:
        print("The model does not have a device map")


def check_quant_and_configs(model:Any) -> None:
    """ Check all quantization and configs for models loaded with bnb """

    if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
        print("The model is loaded in 8-bit")
    elif hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        print("The model is loaded in 4-bit")
    else:
        print("The model is not loaded in 8-bit or 4-bit")

    if hasattr(model, 'bnb_config'):
        print("The model has a bnb_config attribute")
    else:
        print("The model does not have a bnb_config attribute")
    
    print("Model parameter data types:")
    for param in model.parameters():
        print(param.dtype)

    if hasattr(model, 'device_map'):
        print(f"Device map: {model.device_map}")
    else:
        print("No device map found")


def check_model_dtypes(model:Any) -> None:
    """ Check dtypes for model parameters """

    for param in model.parameters():
        print(param.dtype) 


def check_named_params(model:Any) -> None:
    """ Check the name parameters """
    for name, param in model.named_parameters():
        print(param.dtype)


def model_to_dtype(model:Any, dtype:torch.dtype) -> Any:
    """ Set model dtype """

    model_dtype = model.to(dtype=dtype)
    
    return model_dtype


def set_params_device(model:Any, device:str) -> Any:
    """ Move model parameters to device """
    
    for param in model.parameters():
        param.data = param.data.to(device)
    
    return model


def print_model_weights(model:torch.nn.Module) -> None:
    """Print the weights of all layers in the model."""
    
    for name, param in model.named_parameters():
        if 'weight' in name:  
            print(f"Layer: {name} - Weights: {param.data}")


def ptq_1bit(model: Any) -> Any:
    """ Apply 1-bit quantization {-1, 1} (binary) to OLMo transformer layers. """
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'layers' in name:
            # Only quantize attention (q_proj, k_proj, v_proj, out_proj) and feedforward (fc1, fc2)
            if any(layer in name for layer in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']):
                weight = module.weight.data
                quantized_weight = torch.sign(weight)  # {-1, 1} binary quantization
                
                module.weight.data = quantized_weight
                print(f"Quantized Layer: {name}\n", quantized_weight)

    return model


def ptq_1_58bit(model: Any, sparsity_ratio: float = 0.25, sample_ratio: float = 0.01) -> Any:
    """ Apply 1.58-bit quantization (ternary {-1, 0, 1}) to transformer layers.
        Example of other precision:
            # Estimate tau in FP32
            abs_weight_fp32 = module.weight.data.abs().to(torch.float32)
            sample = abs_weight_fp32.flatten()[torch.randperm(abs_weight_fp32.numel())[:num_samples]]
            tau = torch.quantile(sample, sparsity_ratio)

            # Quantize in FP16
            weight_fp16 = module.weight.data.to(torch.float16)
            quantized_weight = torch.sign(weight_fp16) * (weight_fp16.abs() > tau)
            quantized_weight[weight_fp16.abs() <= tau] = 0
            module.weight.data = quantized_weight """
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'layers' in name:
            if any(layer in name for layer in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']):
                weight = module.weight.data.to(torch.float32)
                #weight = module.weight.data.to(torch.float16)
                abs_weight = weight.abs()

                # Efficient threshold estimation (quantile-based)
                num_samples = max(1000, int(sample_ratio * abs_weight.numel()))
                sample_values = abs_weight.flatten()[torch.randperm(abs_weight.numel())[:num_samples]]
                tau = torch.quantile(sample_values, sparsity_ratio)

                # Apply ternary quantization
                quantized_weight = torch.sign(weight) * (abs_weight > tau)
                quantized_weight[abs_weight <= tau] = 0  # Set some weights to zero

                module.weight.data = quantized_weight
                print(f"Quantized Layer: {name}\n", quantized_weight)

    return model


def ptq_2bit(model: Any) -> Any:
    """ Apply 2-bit quantization {-1, -0.33, 0.33, 1} to transformer layers. """
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'layers' in name:
            if any(layer in name for layer in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']):
                weight = module.weight.data
                abs_weight = weight.abs()
                scale = abs_weight.mean()  # Compute scale for normalization

                # 2-bit Quantization: 4 levels
                quantized_weight = torch.zeros_like(weight)
                quantized_weight[weight > 0.66 * scale] = 1
                quantized_weight[(weight > 0) & (weight <= 0.66 * scale)] = 0.33
                quantized_weight[(weight < 0) & (weight >= -0.66 * scale)] = -0.33
                quantized_weight[weight < -0.66 * scale] = -1

                module.weight.data = quantized_weight
                print(f"Quantized Layer: {name}\n", quantized_weight)

    return model


def ptq_4bit_uniform(model: Any) -> Any:
    """ Apply 4-bit uniform quantization to transformer layers. """

    num_levels = 2**4  # 16 levels
    levels = torch.linspace(-1, 1, num_levels)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'layers' in name:
            if any(layer in name for layer in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']):
                weight = module.weight.data
                max_val = weight.abs().max()
                if max_val == 0:
                    continue  # avoid division by zero

                # Normalize weights to [-1, 1]
                norm_weight = weight / max_val

                # Find closest quantization level
                quantized = norm_weight.unsqueeze(-1) - levels.to(weight.device)
                quantized_idx = torch.argmin(quantized.abs(), dim=-1)
                quantized_weight = levels[quantized_idx].to(weight.device) * max_val

                module.weight.data = quantized_weight
                print(f"Quantized {name} to 4-bit\n", quantized_weight)

    return model


def ptq_8bit_uniform(model: Any) -> Any:
    """ Apply 8-bit uniform quantization to transformer layers """

    num_levels = 2**8  # 256 levels
    levels = torch.linspace(-1, 1, num_levels)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'layers' in name:
            if any(layer in name for layer in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']):
                weight = module.weight.data
                max_val = weight.abs().max()
                if max_val == 0:
                    continue

                norm_weight = weight / max_val

                quantized = norm_weight.unsqueeze(-1) - levels.to(weight.device)
                quantized_idx = torch.argmin(quantized.abs(), dim=-1)
                quantized_weight = levels[quantized_idx].to(weight.device) * max_val

                module.weight.data = quantized_weight
                print(f"Quantized {name} to 8-bit\n", quantized_weight)

    return model
