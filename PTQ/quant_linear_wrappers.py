from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


# ===================== Abstract ============================
class IQuantLinear(ABC, nn.Module):
    def __init__(self, name='unknown'):
        nn.Module.__init__(self)  # <- FIXED: directly call nn.Module
        self.name = name

    """ Abstract / Interface for custom quantized torch nn.Linear """

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        pass


# ===================== Ternary quant wrapper ============================
class BitLinear(nn.Linear, IQuantLinear):
    def __init__(self:BitLinear,
                 orig:nn.Linear,
                 dtype:torch.dtype,
                 name='unknown') -> None:
        
        nn.Linear.__init__(self, orig.in_features, orig.out_features, bias=(orig.bias is not None))
        IQuantLinear.__init__(self, name=name)
        
        self.weight = nn.Parameter(orig.weight.data.clone())
        self.ternary_weight = None
        self.bias = orig.bias if orig.bias is not None else None
        self.dtype = dtype

    """ Wrapper class for BitLinear """

    def forward(self: BitLinear, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.dtype)
        device = input.device

        batch_size, seq_len, _ = input.shape
        input_reshaped = input.view(batch_size * seq_len, -1)

        # Use the scaled ternary weight and bias (if any)
        weight = self.ternary_weight.to(device=device, dtype=self.dtype)
        bias = self.bias.to(device=device, dtype=self.dtype) if self.bias is not None else None

        # Apply matrix multiplication (ternary weight * input)
        output_reshaped = F.linear(input_reshaped, weight, bias)

        return output_reshaped.view(batch_size, seq_len, -1)

    
    def dequantize(self: BitLinear, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().to(self.dtype)


# ===================== FFQ Wrapper ============================
class FFQLinear(IQuantLinear, nn.Linear):
    def __init__(self:FFQLinear,
                 orig:nn.Linear,
                 dtype:torch.dtype,
                 dropout_prob=0.0,
                 name='unknown') -> None:

        nn.Linear.__init__(self, orig.in_features, orig.out_features, bias=(orig.bias is not None))
        IQuantLinear.__init__(self, name=name)  # safe to call

        self.register_buffer('q_int_weight', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.register_buffer('dropout_mask', None)

        self.weight = nn.Parameter(orig.weight.data.clone())
        self.bias = orig.bias if orig.bias is not None else None

        self.dtype = dtype
        self.dropout_prob = dropout_prob

    """ Wrapper class for FFQ """

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        input = input.to(self.dtype)
        batch_size, seq_len, _ = input.shape
        input_reshaped = input.view(batch_size * seq_len, -1)

        scale = self.scale if torch.is_tensor(self.scale) else torch.tensor(self.scale, dtype=self.dtype)
        zero_point = self.zero_point if torch.is_tensor(self.zero_point) else torch.tensor(self.zero_point, dtype=self.dtype)
        weight = (self.q_int_weight.float() - zero_point) * scale
        weight = weight.to(self.dtype)

        bias = self.bias.to(self.dtype) if self.bias is not None else None

        output_reshaped = F.linear(input_reshaped, weight.t(), bias)
        return output_reshaped.view(batch_size, seq_len, -1)


    def dequantize(self, tensor:torch.Tensor) -> torch.Tensor:
        return (tensor.float().to(self.dtype) - self.zero_point) * self.scale
    

# ===================== FFQ Wrapper ============================
class PTQLinear(IQuantLinear, nn.Linear):
    def __init__(self:PTQLinear,
                 orig:nn.Linear,
                 dtype:torch.dtype,
                 dropout_prob=0.0,
                 name='unknown') -> None:

        nn.Linear.__init__(self, orig.in_features, orig.out_features, bias=(orig.bias is not None))
        IQuantLinear.__init__(self, name=name)  # <- safe to call

        self.register_buffer('q_int_weight', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        #self.register_buffer('dropout_mask', None)

        self.weight = nn.Parameter(orig.weight.data.clone())
        self.bias = orig.bias if orig.bias is not None else None

        self.dtype = dtype
        self.dropout_prob = dropout_prob

    """ Wrapper class for PTQ """

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        input = input.to(self.dtype)
        batch_size, seq_len, _ = input.shape
        input_reshaped = input.view(batch_size * seq_len, -1)

        scale = self.scale if torch.is_tensor(self.scale) else torch.tensor(self.scale, dtype=self.dtype)
        zero_point = self.zero_point if torch.is_tensor(self.zero_point) else torch.tensor(self.zero_point, dtype=self.dtype)
        weight = (self.q_int_weight.float() - zero_point) * scale
        weight = weight.to(self.dtype)

        bias = self.bias.to(self.dtype) if self.bias is not None else None

        output_reshaped = F.linear(input_reshaped, weight, bias)
        return output_reshaped.view(batch_size, seq_len, -1)


    def dequantize(self, tensor:torch.Tensor) -> torch.Tensor:
        return (tensor.float().to(self.dtype) - self.zero_point) * self.scale
    