from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal

import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaForCausalLM

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

# ===================== QuantLinear ============================
class QuantLinear(nn.Linear):
    def __init__(self:QuantLinear, orig: nn.Linear, dtype:torch.dtype):
        """
        Wrapper class for replacement of quantized nn.Linear layers with QuantLinear Modules
        """
        super().__init__(orig.in_features, orig.out_features, orig.bias is not None)
        
        self.register_buffer('q_int_weight', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.ternary_weight = None
        self.bias = orig.bias
        self.weight = nn.Parameter(orig.weight.data)
        self.dtype = dtype

    def forward(self:QuantLinear, input:torch.Tensor) -> torch.Tensor|Any:
        if self.q_int_weight is not None and self.scale is not None and self.zero_point is not None:
            scale = self.scale if torch.is_tensor(self.scale) else torch.tensor(self.scale, dtype=self.dtype)
            zero_point = self.zero_point if torch.is_tensor(self.zero_point) else torch.tensor(self.zero_point, dtype=self.dtype)
            weight = (self.q_int_weight.float() - zero_point) * scale
        elif self.ternary_weight is not None:
            weight = self.ternary_weight
        else:
            weight = self.weight

        input = input.to(self.dtype)
        weight = weight.to(self.dtype)
        bias = self.bias.to(self.dtype) if self.bias is not None else None

        assert input.dtype == weight.dtype == (bias.dtype if bias is not None else input.dtype), (
            f"[QuantLinear] Dtype mismatch: input={input.dtype}, weight={weight.dtype}, bias={getattr(bias, 'dtype', None)}"
        )

        return F.linear(input, weight, bias)

    def dequantize(self:QuantLinear, tensor:torch.Tensor|Any) -> torch.Tensor|Any:
        """ Helper method to dequantize the weights """
        if self.q_int_weight is not None and self.scale is not None and self.zero_point is not None:
            return (tensor.float().to(self.dtype) - self.zero_point) * self.scale
        else:
            return tensor.float().to(self.dtype)

# ===================== Quantization Functions ============================
def quantize_uniform(
        tensor:torch.Tensor, n_bits:int, symmetric:bool=True, dtype:torch.dtype=torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    
    """ With safer scale handling ε / epsilon to avoid division by near-zero """
    EPSILON = 1e-8

    if symmetric:
        max_val = tensor.abs().max()
        scale = max(max_val.item(), EPSILON) / (2**(n_bits - 1) - 1)
        zero_point = 0
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

def log_quantization_error(orig, quantized, layer_name):
    error = F.mse_loss(orig, quantized).item()
    print(f"[QuantError] {layer_name}: {error:.6f}")

def add_activation_hooks(
        model:nn.Module, dtype:torch.dtype, layers_to_quant:List[str]|None, symmetric:bool=True, num_bits:int=8
    ) -> List[Any]:
    """ Add activation hooks for PTSQ: quantization of activations (here BitNet-style 8-bit symmetric) """
    handles = []

    def quantize_activation_uniform(tensor: torch.Tensor) -> torch.Tensor:
        q_tensor, _, _, _ = quantize_uniform(tensor, num_bits, symmetric, dtype)
        return q_tensor

    def get_hook():
        return lambda mod, inp, out: quantize_activation_uniform(out)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if layers_to_quant is None or any(layer in name for layer in layers_to_quant):
                hook = get_hook()
                handle = module.register_forward_hook(hook)
                handles.append(handle)
                print(f"[Activation Hook] Added for: {name}")

    return handles

# ===================== Main Quantization Logic ============================
def apply_PTQ(
        model:Any,
        mode:str='1.58bit',
        model_half:bool=False,
        quant_half:bool=False,
        layers_to_quant:Optional[List[str]|None]=None,
        act_quant:bool=False,
        act_bits:int=8
    ) -> Any:
    """
    Apply uniform PTQ strategies (PTDQ and PTSQ) to pretrained transformers
    
    PARAMS:
        model:
            Any model (tested for LLaMAs and OLMos)
        mode:str:
            quant styles; '1.58bit', ('1bit_sym', '1bit_asym'), '2bit_sym', '2bit_asym', '4bit_sym', '4bit_asym', '8bit_sym', '8bit_asym'
        model_half:bool:
            if True model in float16 else model in float32
        quant_half:bool:
            if True quant in float16 else model in float32
            if mode is '1.58-bit', dtype is cast to float32 for quantization because of tau, and then converted to float16 if quant_half is True
        layers_to_quant:
            Optional[List[str]|None]
        act_quant:bool:
            if true, symmetric activation ptsq
        act_bits:int:
            default 8-bit symmetric quantization (BitNet-style)

    RETURNS:
        PTDQ or PTSQ float16 or float32 precision model with quantized float16 or float32 nn.Linear transformer layers replaced by QuantLinear Modules
    """
    if  mode  == '1bit_sym' or mode == '1bit_asym':
        raise NotImplementedError("Triggers zero-division error for 1-bit quantization!")
    
    model.to(torch.float16) if model_half else model.to(torch.float32)
    
    if quant_half:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"|| Quant Configs: {mode} | dtype: {dtype} | PTQ as: {str('PTSQ') if act_quant else str('PTDQ')} ||")
    
    def quantize_and_replace(name, module):
        quantized = QuantLinear(module, dtype=dtype)
        w = module.weight.data.to(dtype)

        if mode == '1.58bit':
            w = module.weight.data.to(torch.float32) # since τ / tau, set the dtype for quants to float32
            q_w, tau = quantize_ternary(w)
            quantized.ternary_weight = q_w.to(dtype)
            quantized.weight.data = quantized.ternary_weight
            print(f"[1.58-bit] {name} | τ = {tau:.4f}")
        else:
            n_bits = int(mode.replace('bit', '').replace('_sym', '').replace('_asym', ''))
            symmetric = '_sym' in mode
            q_w, q_int, scale, zp = quantize_uniform(w, n_bits, symmetric=symmetric, dtype=dtype)
            quantized.weight.data = q_w
            quantized.q_int_weight = q_int
            quantized.scale = torch.tensor(scale)
            quantized.zero_point = torch.tensor(zp)
            print(f"[{n_bits}-bit] {name} | scale={scale:.4f} zp={zp:.4f}")

        #log_quantization_error(module.weight.data, quantized.weight.data, name)
        set_module_by_name(model, name, quantized)

    if layers_to_quant is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quantize_and_replace(name, module)
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'layers' in name:
                if any(layer in name for layer in layers_to_quant):
                    quantize_and_replace(name, module)

    if act_quant and layers_to_quant is None:
        add_activation_hooks(model, dtype=dtype, layers_to_quant=None, symmetric=True, num_bits=act_bits)
    elif act_quant and layers_to_quant is not None:
        add_activation_hooks(model, dtype=dtype, layers_to_quant=layers_to_quant, symmetric=True, num_bits=act_bits)
    
    print("Quantization complete!")
    return model

# ===================== LLaMA Wrapper Class ============================
class QuantLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._replace_linear_with_quantlinear()

    def _replace_linear_with_quantlinear(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                parent = self
                path = name.split(".")
                for part in path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, path[-1], QuantLinear(module))

# ===================== Loading and Saving ============================
def save_quantized_model(model, save_path:str)  -> None:
    # Make sure quantization buffers are initialized properly
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if module.q_int_weight is None:
                module.q_int_weight = torch.zeros_like(module.weight.data)  # Initialize with zeros
            if module.scale is None:
                module.scale = torch.tensor(1.0)  # Default scale value
            if module.zero_point is None:
                module.zero_point = torch.tensor(0.0)  # Default zero point value
    
    model.save_pretrained(save_path, safe_serialization=True)
    print(f"Quantized model saved to {save_path}")

def load_quantized_model(model_path:str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=True,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_safetensors=True
    )
    # Ensure that the quantization buffers are loaded correctly
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # Ensure the buffers are correctly loaded
            if module.q_int_weight is None:
                module.q_int_weight = torch.zeros_like(module.weight.data)  # Initialize with zeros
            if module.scale is None:
                module.scale = torch.tensor(1.0)  # Default scale value
            if module.zero_point is None:
                module.zero_point = torch.tensor(0.0)  # Default zero point value

    return model
