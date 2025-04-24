from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F



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


def get_weights(model):
    ws = []
    for name, param in model.named_parameters():
        if 'weight' in name:  
            ws.append(param.data)
            
    return ws


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


def print_unique_weights(model:Any) -> None:
    """ Print the unique weights of any model. """
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: Unique values = {module.weight.data.unique().numel()}")


def add_hooks(model:Any, hook_idx:int) -> Dict:
    """ Add Hooks for Easy Layer Grab:
            Hooks to extract weights or activations during the forward pass.
            Example: model.layers[4].self_attn.q_proj.register_forward_hook(hook_fn("q_proj")) """
    
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    model.layers[hook_idx].self_attn.q_proj.register_forward_hook(hook_fn('q_proj'))
    return activations

def requires_grad(model:Any) -> None:
    model.eval()
    for param in model.parameters():
        if param.requires_grad is True:  # if that matters for hooks
            print(param)

def track_activations(model:Any, input_ids:torch.Tensor) -> None:
    # Example: forward pass with tracking
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)

    # Then inspect:
    for layer, act in model.activations.items():
        print(f"{layer}: {act.shape}, mean={act.mean():.4f}, std={act.std():.4f}")
