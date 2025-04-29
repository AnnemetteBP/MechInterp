from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from .quant_linear_interface import IQuantLinear

import warnings
warnings.filterwarnings('ignore')


# ===================== Ternary quant wrapper ============================ 
class BitLinear(IQuantLinear, nn.Module):
    def __init__(self:BitLinear,
                orig:nn.Linear,
                dtype:torch.dtype,
                name='unknown',
                act_quant=False,
                act_bits=8,
                use_ternary=False,
                smart_alpha=False,
                debug=True,
                plotting:bool=False) -> None:
        
        
        super().__init__()
        IQuantLinear.__init__(self, name=name)
        
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.has_bias = orig.bias is not None
        self.dtype = dtype
        self.act_quant = act_quant
        self.act_bits = act_bits
        self.use_ternary = use_ternary
        self.smart_alpha = smart_alpha 
        self.debug = debug
        self.debugger = None
        self.plotting = plotting  

        # Buffers
        self.register_buffer('orig_weight', orig.weight.data.clone())
        if self.has_bias:
            self.bias = nn.Parameter(orig.bias.data.clone())
        else:
            self.bias = None

        self.register_buffer('ternary_weight', None)
        self.alpha = None
        self.tau = None

        if act_quant:
            self.register_buffer('act_scale', torch.tensor(1.0))
            self.act_scale_initialized = False


    def quantize_ternary(self:BitLinear, tensor:torch.Tensor, sparsity_ratio:float=0.25, sample_ratio:float=0.01, deterministic:bool=True) -> Tuple[torch.Tensor, float, float]:
        """BitNet-style ternary quantization with optional smarter alpha."""
        abs_tensor = tensor.abs()
        total_elems = abs_tensor.numel()

        num_samples = min(max(2048, int(sample_ratio * total_elems)), total_elems)
        flat = abs_tensor.flatten()

        if deterministic:
            sample = flat[:num_samples]
        else:
            indices = torch.randperm(total_elems, device=abs_tensor.device)[:num_samples]
            sample = flat[indices]

        tau = torch.quantile(sample, sparsity_ratio)

        ternary = torch.where(abs_tensor >= tau, torch.sign(tensor), torch.zeros_like(tensor))
        print(f"[DEBUG] Unpacked: {ternary} | ")
        packed_ternary = (ternary + 1).to(torch.uint8)
        print(f"[DEBUG] Packed: {packed_ternary} | ")

        # Smart alpha selection
        if self.smart_alpha:
            non_zero_mask = (abs_tensor >= tau)
            alpha = abs_tensor[non_zero_mask].mean() if non_zero_mask.sum() > 0 else abs_tensor.mean()
        else:
            alpha = abs_tensor.mean()

        return packed_ternary, tau.item(), alpha.item()


    def quantize(self:BitLinear, weight:torch.Tensor, deterministic:bool=False):
        """ Quantize weights to ternary and store them """
        if self.use_ternary:
            q_w, tau, alpha = self.quantize_ternary(weight.float(), deterministic=deterministic)
            self.ternary_weight = q_w
            self.alpha = alpha
            self.tau = tau
        else:
            pass  # placeholder for other quantization types if needed...


    def dequantize(self:BitLinear) -> torch.Tensor:
        """ Reverse ternary quantization """
        if self.ternary_weight is None:
            raise RuntimeError("Ternary weights not initialized! Please call .quantize() first.")
        unpacked_ternary = (self.ternary_weight.float() - 1)  # 0,1,2 → -1,0,1
        return unpacked_ternary * self.alpha


    def quantize_activation(self:BitLinear, input:torch.Tensor) -> torch.Tensor:
        """ Quantize activations to n-bit symmetric integer values. """
        if not hasattr(self, 'act_scale'):
            return input
        
        n_bits = self.act_bits
        qmin = -(2 ** (n_bits - 1) - 1)
        qmax = (2 ** (n_bits - 1)) - 1
        scale = self.act_scale.item()

        input_int = (input / scale).round().clamp(qmin, qmax)
        input_dequant = input_int * scale

        # In quantize_activation, modify this:
        if self.debug and self.debugger is not None:
            self.debugger.compare_activations(self.name, input, input_dequant)
        elif self.plotting:
            self.plot_activation(input, input_dequant)


        return input_dequant
    
    def freeze_quantization(self:BitLinear):
        self.debug = False
        if self.act_quant and not self.act_scale_initialized: # minimal unless tracking hooks etc.
            raise RuntimeError(f"Activation scale for {self.name} not calibrated yet!")

    def calibrate_activation(self:BitLinear, input:torch.Tensor, act_bits:int=8, force:bool=False, percentile:float=0.9999):
        """ Find the scale for activation quantization using percentile calibration """
        if self.act_quant and (not self.act_scale_initialized or force):
            qmax = (2 ** (act_bits - 1)) - 1
            abs_input = input.abs().flatten()
            k = int(abs_input.numel() * percentile)
            k = max(k, 1)

            topk_val, _ = torch.topk(abs_input, k, largest=True, sorted=True)
            max_val = topk_val[-1]

            scale = max(max_val.item(), 1e-5) / qmax   # epsilon
            self.act_scale.copy_(torch.tensor(scale, device=self.act_scale.device))
            self.act_scale_initialized = True

            print(f"[Calibration] {self.name} | Max Activation: {max_val.item():.5f} | Scale: {scale:.8f}")

    
    def forward(self:BitLinear, input:torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"[{self.name}] input.mean(): {input.mean().item()}, input.std(): {input.std().item()}")
            self.input_activation = input.detach().cpu()  # Save for inspection

        if self.act_quant and self.act_scale_initialized:
            input = self.quantize_activation(input)

        weight = self.dequantize() if self.use_ternary else self.orig_weight
        if weight.dtype != input.dtype:
            weight = weight.to(input.dtype)

        bias = self.bias.to(input.device) if self.bias is not None else None
        return F.linear(input, weight, bias)


    def plot_activation(self:BitLinear, input:torch.Tensor, input_dequant:torch.Tensor):
        """ Optional visualization for debugging quantization quality """
        input = input.detach().cpu().flatten()
        input_dequant = input_dequant.detach().cpu().flatten()

        plt.figure(figsize=(8, 3))
        plt.hist(input.numpy(), bins=100, alpha=0.5, label='Original')
        plt.hist(input_dequant.numpy(), bins=100, alpha=0.5, label='Quantized+Dequantized')
        plt.legend()
        plt.title(f"Activation Histogram ({self.name})")
        plt.show()