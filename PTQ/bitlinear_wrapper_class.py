from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


class BitLinear(nn.Module):
    def __init__(self,
                 orig:nn.Linear,
                 dtype:torch.dtype,
                 name:str='unknown',
                 act_quant:bool=False,
                 act_bits:int=8,
                 use_ternary:bool=False,
                 smart_alpha:bool=False,
                 deterministic:bool=True,
                 debug:bool=True,
                 plotting:bool=False) -> None:
        
        super().__init__()

        self.name = name or "UnnamedBitLinear"
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.has_bias = orig.bias is not None
        self.dtype = dtype
        self.act_quant = act_quant
        self.act_bits = act_bits
        self.use_ternary = use_ternary
        self.smart_alpha = smart_alpha
        self.deterministic = deterministic
        self.orig_weight = orig.weight.detach().clone()
        self.orig_bias = orig.bias.detach().clone() if orig.bias is not None else None
        self.act_scale_initialized = False
        self.debug = debug
        self.debugger = None
        self.plotting = plotting

        # Register buffers for ternary quantized weights and alpha
        self.register_buffer('ternary_weight', torch.zeros_like(orig.weight))
        self.register_buffer('alpha', torch.tensor(0.0))
        #self.register_buffer('tau', torch.tensor(0.0))

        if self.has_bias:
            self.bias = nn.Parameter(orig.bias.data.clone())
        else:
            self.bias = None

        # Register act_scale buffer if activation quantization is enabled
        if act_quant and not hasattr(self, 'act_scale'):
            self.register_buffer('act_scale', torch.tensor(1.0))  # Initialize act_scale buffer

        self.act_scale_initialized = self.act_scale is not None


    def quantize_ternary(self:BitLinear, tensor:torch.Tensor, eps:float=1e-8, deterministic:bool=True) -> Tuple[torch.Tensor, float]:
        """
        Perform ternary quantization on a tensor: quantize tensor values to {-1, 0, 1}.
        """
        gamma = tensor.abs().mean()  # Mean absolute value as scaling factor
        scaled = tensor / (gamma + eps)  # Avoid division by zero
        quantized = torch.round(scaled).clamp(-1, 1)  # Round to nearest ternary value (-1, 0, 1)
        print(f"[DEBUG] Quantized: {quantized}| ")
        packed_ternary = (quantized + 1).to(torch.uint8)  # Convert {-1, 0, 1} to {0, 1, 2} (uint8)
        return packed_ternary, gamma.item()


    def quantize(self:BitLinear, weight:Optional[torch.Tensor]=None) -> None:
        """
        Quantizes ternary weights in-place or from an input tensor.

        If `weight` is None, defaults to self.orig_weight.
        """
        if not self.use_ternary:
            return

        # Use passed weight or fall back to internal one (self.orig_weight)
        weight_to_quant = weight if weight is not None else self.orig_weight

        # Quantize ternary
        q_w, gamma = self.quantize_ternary(weight_to_quant.float())  # No 'deterministic' argument

        # Store ternary and alpha
        self.ternary_weight.data.copy_(q_w.to(self.ternary_weight.dtype))
        self.alpha.data.copy_(torch.tensor(gamma, device=self.alpha.device))
        # Cache dequantized weights for forward()
        self.dequantized_weight = self.dequantize()


    def dequantize(self:BitLinear) -> torch.Tensor:
        """
        Dequantizes ternary weights (maps {0, 1, 2} back to {-1, 0, 1}).
        """
        dequantized = self.ternary_weight.float() - 1  # Map {0, 1, 2} back to {-1, 0, 1}
        return dequantized


    def dequantize_activation(self:BitLinear, act_int8:torch.Tensor) -> torch.Tensor:
        """
        Dequantizes an activation tensor from int8 back to floating-point.
        """
        scale = self.act_scale.item()
        out = act_int8.float() * scale
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

    def quantize_activation(self:BitLinear, input:torch.Tensor, return_int:bool=False) -> torch.Tensor:
        """
        Quantizes activation input to a fixed bit-width and returns the quantized version.
        """
        if self.act_scale is None or not self.act_scale_initialized:
            raise RuntimeError(f"act_scale is not initialized! Please call calibrate_activation() first.")
        
        Qb = (2 ** (self.act_bits - 1)) - 1  # Max value for int8
        scale = max(self.act_scale.item(), 1e-5)  # Avoid division by zero
        input_int = (input / scale).round().clamp(-Qb, Qb).to(torch.int8)

        self.last_quantized_act = input_int.detach().cpu()

        if return_int:
            return input_int
        else:
            input_dequant = input_int.float() * scale
            input_dequant = torch.nan_to_num(input_dequant, nan=0.0, posinf=1e4, neginf=-1e4)
            return input_dequant


    def calibrate_activation(self:BitLinear, input:torch.Tensor, act_bits:int=8, force:bool=False, percentile:float=0.999) -> None:
        """
        Calibrates the activation scale (act_scale) based on the input tensor.
        """
        if not self.act_quant or (self.act_scale_initialized and not force):
            return

        Qb = (2 ** (act_bits - 1)) - 1  # Max value for int8
        abs_input = input.detach().abs().flatten()

        if abs_input.numel() == 0:
            print(f"[WARN] Empty input in {self.name} during calibration.")
            return

        k = max(1, int(abs_input.numel() * percentile))
        topk_val, _ = torch.topk(abs_input, k, largest=True)
        max_val = topk_val[-1]
        scale = max(max_val.item(), 1e-5) / Qb

        if self.act_scale is None:
            self.act_scale = torch.tensor(scale, device=input.device)
        else:
            self.act_scale.copy_(torch.tensor(scale, device=self.act_scale.device))

        self.act_scale_initialized = True
        print(f"[Calibration] {self.name} | Max Activation: {max_val.item():.5f} | Scale: {scale:.8f}")


    def freeze_quantization(self:BitLinear) -> None:
        """
        Freezes quantization by disabling debug mode.
        """
        self.debug = False
        if self.act_quant and not self.act_scale_initialized:
            raise RuntimeError(f"Activation scale for {self.name} not calibrated yet!")


    def forward(self:BitLinear, input:torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized activations and weights.
        """
        if self.debug:
            print(f"[{self.name}] input.mean(): {input.mean().item()}, input.std(): {input.std().item()}")
            self.input_activation = input.detach().cpu()

        # Ensure act_scale is calibrated before quantization
        if not self.act_scale_initialized:
            self.calibrate_activation(input)

        if self.act_quant:
            input = self.quantize_activation(input)

        weight = self.dequantize() if self.use_ternary else self.orig_weight
        bias = self.bias
        if bias is not None:
            bias = bias.to(input.dtype).to(input.device)

        return F.linear(input, weight, bias)


    def pack_ternary_to_bytes(self:BitLinear, ternary:torch.Tensor) -> torch.ByteTensor:
        """
        Packs ternary values in {0, 1, 2} (as uint8) into bytes (4 values per byte).
        Returns a flattened byte tensor.
        """
        ternary = ternary.view(-1)
        padding = (4 - (ternary.numel() % 4)) % 4
        if padding > 0:
            ternary = torch.cat([ternary, torch.zeros(padding, dtype=ternary.dtype, device=ternary.device)])

        ternary = ternary.to(torch.uint8)
        packed = (
            (ternary[0::4] << 6) |
            (ternary[1::4] << 4) |
            (ternary[2::4] << 2) |
            ternary[3::4]
        )
        return packed


    def unpack_bytes_to_ternary(self:BitLinear, packed:torch.ByteTensor, original_shape:torch.Size) -> torch.Tensor:
        """
        Unpacks bytes to ternary values in {0, 1, 2}.
        Returns tensor in original shape.
        """
        unpacked = torch.empty(packed.numel() * 4, dtype=torch.uint8, device=packed.device)
        unpacked[0::4] = (packed >> 6) & 0b11
        unpacked[1::4] = (packed >> 4) & 0b11
        unpacked[2::4] = (packed >> 2) & 0b11
        unpacked[3::4] = packed & 0b11
        return unpacked[:torch.prod(torch.tensor(original_shape))].view(original_shape)


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