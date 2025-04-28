from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

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
    tensor: torch.Tensor,
    n_bits: int,
    symmetric: bool = True,
    dtype: torch.dtype = torch.float32,
    per_channel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Uniform quantization supporting symmetric/asymmetric & per-channel."""
    
    EPSILON = 1e-8
    tensor = tensor.to(torch.float32)

    if per_channel:
        # Per-channel quantization logic
        scales = []
        zero_points = []

        for i in range(tensor.shape[0]):  # per-channel means per row
            channel = tensor[i]  # each row/channel
            
            if symmetric:
                qmin = -(2 ** (n_bits - 1) - 1)
                qmax = (2 ** (n_bits - 1)) - 1
                max_val = channel.abs().max()
                scale = max(max_val.item(), EPSILON) / qmax
                zero_point = 0  # Symmetric quantization using zero-centered
                
            else:
                qmin = 0
                qmax = 2 ** n_bits - 1
                min_val = channel.min()
                max_val = channel.max()
                scale = max((max_val - min_val).item(), EPSILON) / (qmax - qmin)
                zero_point = round(-min_val.item() / scale)
                zero_point = int(max(qmin, min(zero_point, qmax)))

            # Save the scale and zero-point for each channel
            scales.append(scale)
            zero_points.append(zero_point)

        scales = torch.tensor(scales, dtype=torch.float32, device=tensor.device)
        zero_points = torch.tensor(zero_points, dtype=torch.int32, device=tensor.device)
        
        q_int = ((tensor / scales.unsqueeze(-1)) + zero_points.unsqueeze(-1)).round().clamp(qmin, qmax).to(torch.uint8)
        q_tensor = (q_int.to(torch.float32) - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)

    else:
        # Symmetric quantization (no per-channel)
        qmin = -(2 ** (n_bits - 1) - 1)
        qmax = (2 ** (n_bits - 1)) - 1

        max_val = tensor.abs().max()
        scale = max(max_val.item(), EPSILON) / qmax
        zero_point = 0  # symmetric quantization always uses zero-centered

        q_int = (tensor / scale).round().clamp(qmin, qmax).to(torch.int8)
        q_tensor = q_int.to(torch.float32) * scale

        # For symmetric, no need for separate scales or zero-points
        scales = scale
        zero_points = zero_point

    print(f"[DEBUG] Quantized tensor: {q_tensor} | ")
    return q_tensor.to(dtype), q_int, scales, zero_points



def quantize_ternary(
        tensor:torch.Tensor, return_alpha:bool=False, sparsity_ratio:float=0.25, sample_ratio:float=0.01
) -> Tuple[torch.Tensor, float] | Union[Tuple[torch.Tensor, float], Tuple[torch.Tensor, float, float]]:

    """ BitNet-style ternary / 1.58-bit quantization. dtype must be float32. """
    
    abs_tensor = tensor.abs()
    num_samples = max(1000, int(sample_ratio * abs_tensor.numel()))
    sample = abs_tensor.flatten()[torch.randperm(abs_tensor.numel())[:num_samples]]
    tau = torch.quantile(sample, sparsity_ratio)
    ternary = torch.sign(tensor) * (abs_tensor > tau)

    alpha = abs_tensor.mean()

    if return_alpha:
        return ternary, tau.item(), alpha.item()
    else:
        print(f"[DEBUG] Quantized ternary weight: {ternary} |")
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

    """def get_hook(name):
        def hook_fn(mod, inp, out):
            if track:
                model.activations[name] = out.detach().cpu()  # Store original or quantized
            return quantize_activation_uniform(out)
        return hook_fn"""
    def get_hook(name):
        def hook_fn(mod, inp, out):
            out = quantize_activation_uniform(out)
            if track:
                model.activations[name] = out.detach().cpu()
            return out
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
    # Ensure same dtype before matrix multiply for float16 quantization
    U_pinv = torch.pinverse(U.T).to(V.dtype)
    return V @ U_pinv



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
class BitLinear(IQuantLinear, nn.Module):
    def __init__(self:BitLinear,
                orig:nn.Linear,
                dtype:torch.dtype,
                name='unknown',
                act_quant=False,
                act_bits=8,
                use_ternary=False,
                smart_alpha=False,
                debug=True  
                ) -> None:
        
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

    def quantize_ternary(self, tensor:torch.Tensor, sparsity_ratio:float=0.25, sample_ratio:float=0.01, deterministic:bool=True) -> Tuple[torch.Tensor, float, float]:
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


    def quantize(self, weight:torch.Tensor, deterministic:bool=False):
        """ Quantize weights to ternary and store them """
        if self.use_ternary:
            q_w, tau, alpha = self.quantize_ternary(weight.float(), deterministic=deterministic)
            self.ternary_weight = q_w
            self.alpha = alpha
            self.tau = tau
        else:
            pass  # placeholder for other quantization types if needed...

    def dequantize(self) -> torch.Tensor:
        """ Reverse ternary quantization """
        if self.ternary_weight is None:
            raise RuntimeError("Ternary weights not initialized! Please call .quantize() first.")
        unpacked_ternary = (self.ternary_weight.float() - 1)  # 0,1,2 → -1,0,1
        return unpacked_ternary * self.alpha

    def quantize_activation(self, input:torch.Tensor) -> torch.Tensor:
        """ Quantize activations to n-bit symmetric integer values. """
        if not hasattr(self, 'act_scale'):
            return input
        n_bits = self.act_bits
        qmin = -(2 ** (n_bits - 1) - 1)
        qmax = (2 ** (n_bits - 1)) - 1
        scale = self.act_scale.item()

        input_int = (input / scale).round().clamp(qmin, qmax)
        input_dequant = input_int * scale

        if self.debug:
            self.plot_activation(input, input_dequant)

        return input_dequant
    
    def calibrate_activation(self, input:torch.Tensor, act_bits:int=8, force:bool=False, percentile:float=0.9999):
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

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        if self.act_quant and self.act_scale_initialized:
            input = self.quantize_activation(input)

        weight = self.dequantize()
        if weight.dtype != input.dtype:
            weight = weight.to(input.dtype)

        bias = self.bias.to(input.device) if self.bias is not None else None
        return F.linear(input, weight, bias)

    def plot_activation(self, input:torch.Tensor, input_dequant:torch.Tensor):
        """ Optional visualization for debugging quantization quality """
        input = input.detach().cpu().flatten()
        input_dequant = input_dequant.detach().cpu().flatten()

        plt.figure(figsize=(8, 3))
        plt.hist(input.numpy(), bins=100, alpha=0.5, label='Original')
        plt.hist(input_dequant.numpy(), bins=100, alpha=0.5, label='Quantized+Dequantized')
        plt.legend()
        plt.title(f"Activation Histogram ({self.name})")
        plt.show()




# ===================== FFQ Wrapper ============================
class FFQLinear(IQuantLinear, nn.Linear):
    def __init__(
        self,
        orig: nn.Linear,
        dtype: torch.dtype,
        n_bits: int = 2,
        symmetric: bool = True,
        redundancy: int = 2,
        frame_dropout_prob: float = 0.0,
        dropout_prob: float = 0.0,
        name: str = 'unknown',
        reconstruct_weight: bool = False,  # <- new option
    ):
        nn.Linear.__init__(self, orig.in_features, orig.out_features, bias=(orig.bias is not None))
        IQuantLinear.__init__(self, name=name)

        # Buffers for quantized values
        self.register_buffer('q_int_weight', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.register_buffer('dropout_mask', None)
        self.register_buffer('U', None)

        # Copy parameters
        self.weight = nn.Parameter(orig.weight.data.clone())
        self.bias = orig.bias if orig.bias is not None else None

        self.dtype = dtype
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.redundancy = redundancy
        self.frame_dropout_prob = frame_dropout_prob
        self.dropout_prob = dropout_prob
        self.reconstruct_weight = reconstruct_weight

        # Actually quantize
        self.pack_weight()

    def quantize_weight(self, w: torch.Tensor):
        """Quantize a weight tensor using FFQ."""
        # Apply dropout if configured
        if self.dropout_prob > 0.0:
            dropout_mask = (torch.rand_like(w) >= self.dropout_prob).float()
            w = w * dropout_mask
            self.dropout_mask = dropout_mask
        else:
            self.dropout_mask = None

        # Handle non-square weights (e.g., q_proj, k_proj, v_proj)
        if w.shape[0] != w.shape[1]:  # Non-square weight (e.g., attention layers)
            print(f"Quantizing non-square weight: {w.shape}")
            q_V, q_int, scale, zp = quantize_uniform(w, self.n_bits, symmetric=self.symmetric, dtype=self.dtype, per_channel=True)
            return q_V, q_int, scale, zp, None  # No U in non-square case
        
        # Fusion frame transform for square weights
        V, U = fusion_frame_transform(w, redundancy=self.redundancy)
        
        # Frame dropout
        if self.frame_dropout_prob > 0.0:
            frame_mask = (torch.rand_like(V) >= self.frame_dropout_prob).float()
            V = V * frame_mask

        # Uniform quantization
        q_V, q_int, scale, zp = quantize_uniform(V, self.n_bits, symmetric=self.symmetric, dtype=self.dtype, per_channel=True)
        return q_V, q_int, scale, zp, U


    def pack_weight(self):
        """Prepare and store quantized weights."""
        w = self.weight.data.to(self.dtype)

        q_V, q_int, scale, zp, U = self.quantize_weight(w)

        # Save quantization buffers
        self.q_int_weight = q_int
        self.scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, dtype=self.dtype, device=w.device)
        self.zero_point = zp if isinstance(zp, torch.Tensor) else torch.tensor(zp, dtype=self.dtype, device=w.device)
        self.U = U

        # Opt reconstruct weight for analysis or init
        if self.reconstruct_weight:
            w_recon = inverse_fusion_frame_transform(q_V, U)
            self.weight.data.copy_(w_recon.to(self.dtype))

        print(f"[FFQLinear] {self.name}: {self.n_bits}bit | scale={self.scale.float().mean():.5f} | zp={self.zero_point.float().mean():.2f} | shape={w.shape}")


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.dtype)
        batch_size, seq_len, _ = input.shape
        input_reshaped = input.view(batch_size * seq_len, -1)

        # Dequantize weight on the fly
        weight = (self.q_int_weight.float() - self.zero_point) * self.scale
        weight = weight.to(self.dtype)

        bias = self.bias.to(self.dtype) if self.bias is not None else None

        output_reshaped = F.linear(input_reshaped, weight.t(), bias)
        return output_reshaped.view(batch_size, seq_len, -1)

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor.float().to(self.dtype) - self.zero_point) * self.scale


    

# ===================== PTQ Wrapper ============================
class PTQLinear(nn.Linear, IQuantLinear):
    def __init__(self, orig: nn.Linear, dtype: torch.dtype, dropout_prob: float = 0.0, name: str = 'unknown') -> None:
        super().__init__(orig.in_features, orig.out_features, bias=(orig.bias is not None))
        IQuantLinear.__init__(self, name=name)

        # Buffers to hold quantization parameters
        self.register_buffer('q_int_weight', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)

        # Copy bias (bias is *not* quantized typically)
        self.weight = nn.Parameter(orig.weight.data.clone())
        self.bias = orig.bias if orig.bias is not None else None
        self.dtype = dtype
        self.dropout_prob = dropout_prob

    def quantize(self, weight: torch.Tensor, n_bits: int = 8, symmetric: bool = True):
        """Quantize and store the integer weight, scale, and zero point."""
        q_w, q_int, scale, zp = quantize_uniform(weight.to(torch.float32), n_bits, symmetric, dtype=self.dtype)

        self.q_int_weight = q_int  # Stored as uint8
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.zero_point = torch.tensor(zp, dtype=torch.float32)

        # Opt set dequantized weights for compatibility
        self.weight.data = q_w

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.dtype)
        device = input.device

        # Dequantize weight on the fly
        scale = self.scale.to(device)
        zero_point = self.zero_point.to(device)
        q_int = self.q_int_weight.to(device)

        weight = (q_int.float() - zero_point) * scale
        weight = weight.to(self.dtype)

        bias = self.bias.to(device, dtype=self.dtype) if self.bias is not None else None
        
        print(f"[DEBUG] Dequantized weight: {weight} |")
        return F.linear(input, weight, bias)

    def dequantize(self) -> torch.Tensor:
        """Dequantize stored quantized weight."""
        return (self.q_int_weight.float() - self.zero_point) * self.scale

