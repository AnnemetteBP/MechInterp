from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_linear_wrappers import (
    PTQLinear,
    FFQLinear,
    BitLinear
)

import warnings
warnings.filterwarnings('ignore')


"""
Post Training Dynamic Quantization:
    This is the simplest to apply form of quantization where the weights are quantized ahead of time,
    but the activations are dynamically quantized during inference.
    This is used for situations where the model execution time is dominated by loading weights from memory,
    rather than computing the matrix multiplications. This is true for LSTM and Transformer type models with small batch size.

Post Training Static Quantization:
    Post Training Static Quantization (PTQ static) quantizes the weights and activations of the model.
    It fuses activations into preceding layers where possible.
    It requires calibration with a representative dataset to determine optimal quantization parameters for activations.
    Post Training Static Quantization is typically used when both memory bandwidth,
    and compute savings are important with CNNs being a typical use case.

https://pytorch.org/docs/stable/quantization.html

Quantizing any weight:
    weight = module.weight.data 
    q_weight, q_int, scale, zp = quantize_uniform(weight, n_bits=8, symmetric=True)

Fake 2-bit quantization (symmetric):
min_w = linear_layer.weight.data.min()
max_w = linear_layer.weight.data.max()
scale = max_w.abs().max() / (2**1 - 1)  # 2-bit = 4 levels
quant_layer.scale = torch.tensor(scale)
quant_layer.q_int_weight = (linear_layer.weight.data / scale).round().clamp(-2, 1)

Fusion Frame Quantization (FFQ) Integration based on:
"FrameQuant: Flexible Low-Bit Quantization for Transformers": https://arxiv.org/html/2403.06082v1

BitNet-style: https://huggingface.co/docs/transformers/quantization/bitnet, https://arxiv.org/abs/2402.17764
"""

# ===================== Utility Functions ============================
def get_module_by_name(model:nn.Module, name:str) -> nn.Module:
    for part in name.split('.'):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model


def set_module_by_name(model:nn.Module, name:str, new_module:nn.Module) -> None:
    parts = name.split('.')
    for part in parts[:-1]:
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    if parts[-1].isdigit():
        model[int(parts[-1])] = new_module
    else:
        setattr(model, parts[-1], new_module)


def get_layers_for_quant(model, include_attn=True, include_mlp=True, skip_o_proj=True):
    selected = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'layers' in name:
                if include_attn and any(x in name for x in ['q_proj', 'k_proj', 'v_proj']):
                    selected.append(name)
                if include_mlp and any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
                    selected.append(name)
                if skip_o_proj and 'o_proj' in name:
                    continue
    return selected


def log_quantization_error(orig, quantized, layer_name):
    error = F.mse_loss(orig, quantized).item()
    print(f"[QuantError] {layer_name}: {error:.6f}")


# ===================== Quantization Functions ============================
def quantize_uniform(
        tensor:torch.Tensor, n_bits:int, symmetric:bool=True, dtype:torch.dtype=torch.float32, per_channel=False
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    
    """ With safer scale handling ε / epsilon to avoid division by near-zero """
    EPSILON = 1e-8

    if per_channel:
        if symmetric:
            max_val = tensor.abs().max(dim=1, keepdim=True)[0]
            scale = torch.clamp(max_val, min=EPSILON) / (2**(n_bits - 1) - 1)
            zero_point = 0
        else:
            min_val = tensor.min(dim=1, keepdim=True)[0]
            max_val = tensor.max(dim=1, keepdim=True)[0]
            scale = torch.clamp(max_val - min_val, min=EPSILON) / (2**n_bits - 1)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        scale = max((max_val - min_val).item(), EPSILON) / (2**n_bits - 1)
        zero_point = (-min_val / scale).item()

    q_int = ((tensor / scale) + zero_point).round().clamp(0, 2**n_bits - 1)
    q_tensor = (q_int - zero_point) * scale

    return q_tensor.to(dtype), q_int.to(torch.uint8), scale, zero_point


def quantize_ternary(
        tensor:torch.Tensor, sparsity_ratio:float=0.25, sample_ratio:float=0.01
) -> Tuple[torch.Tensor, float]:

    """ BitNet-style ternary / 1.58-bit quantization. dtype must be float32. """
    
    abs_tensor = tensor.abs()
    num_samples = max(1000, int(sample_ratio * abs_tensor.numel()))
    sample = abs_tensor.flatten()[torch.randperm(abs_tensor.numel())[:num_samples]]
    tau = torch.quantile(sample, sparsity_ratio)
    ternary = torch.sign(tensor) * (abs_tensor > tau)

    return ternary, tau.item()


# ===================== Activation Hooks for PTSQ - uniform symmetric activation quant ============================
def add_activation_hooks(
        model:nn.Module,
        dtype:torch.dtype,
        layers_to_quant:List[str]|None,
        symmetric:bool=True,
        num_bits:int=8,
        track:bool=False
) -> List[Any]:
    
    handles = []
    model.activations = {}  # ← Store here if tracking

    def quantize_activation_uniform(tensor: torch.Tensor) -> torch.Tensor:
        q_tensor, _, _, _ = quantize_uniform(tensor, num_bits, symmetric, dtype)
        return q_tensor

    def get_hook(name):
        def hook_fn(mod, inp, out):
            if track:
                model.activations[name] = out.detach().cpu()  # Store original or quantized
            return quantize_activation_uniform(out)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if layers_to_quant is None or any(layer in name for layer in layers_to_quant):
                hook = module.register_forward_hook(get_hook(name))
                handles.append(hook)
                print(f"[Activation Hook] Added for: {name}")

    return handles

# ===================== FFQ ============================
def fusion_frame_transform(tensor, redundancy:int=2):
    B = redundancy
    in_features = tensor.shape[1]
    U = torch.eye(in_features).repeat(B, 1).to(tensor.device)  # [B * in_features, in_features]
    V = tensor @ U.T  # [out_features, B * in_features]
    return V, U

def inverse_fusion_frame_transform(V, U):
    # Reconstruct original tensor via pseudo-inverse
    return V @ torch.pinverse(U.T)


