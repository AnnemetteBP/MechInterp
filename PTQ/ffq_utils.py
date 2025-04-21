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
Fusion Frame Quantization (FFQ) Integration based on:
"FrameQuant: Flexible Low-Bit Quantization for Transformers": https://arxiv.org/html/2403.06082v1
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


# ===================== QuantLinear ============================
class QuantLinear(nn.Linear):
    def __init__(
            self:QuantLinear,
            orig:nn.Linear,
            dtype:torch.dtype,
            dropout_prob:float=0.0,
            name:str='unknown'
        ) -> None:
        
        super().__init__(orig.in_features, orig.out_features, orig.bias is not None)
        
        self.register_buffer('q_int_weight', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.register_buffer('dropout_mask', None)

        self.weight = nn.Parameter(orig.weight.data.clone())
        self.ternary_weight = None
        self.bias = orig.bias

        self.dtype = dtype
        self.dropout_prob = dropout_prob
        self.name = name
        
        """print(f"[QuantLinear INIT] [{self.name}] weight shape: {self.weight.shape}, "
              f"in_features: {self.in_features}, out_features: {self.out_features}")"""

    @classmethod
    def from_float(cls, float_module: nn.Linear, name='unknown'):
        quant_weight = quantize_uniform(float_module.weight)
        return cls(float_module, float_module.weight.dtype, name=name)

    def forward(self: QuantLinear, input: torch.Tensor) -> torch.Tensor | Any:
        """QuantLinear forward pass"""
        assert input.shape[-1] == self.weight.shape[1], \
            f"Mismatch: input {input.shape}, weight {self.weight.shape}"
        
        # Reshape the input to match the expected input size for linear transformation
        batch_size, seq_len, _ = input.shape  
        # Reshape input to [batch_size * seq_len, input_features]
        input_reshaped = input.view(batch_size * seq_len, -1) 

        if self.q_int_weight is not None and self.scale is not None and self.zero_point is not None:
            scale = self.scale if torch.is_tensor(self.scale) else torch.tensor(self.scale, dtype=self.dtype)
            zero_point = self.zero_point if torch.is_tensor(self.zero_point) else torch.tensor(self.zero_point, dtype=self.dtype)
            weight = (self.q_int_weight.float() - zero_point) * scale
            #print(f"[QuantLinear] q_int_weight shape: {weight.shape}")
        elif self.ternary_weight is not None:
            weight = self.ternary_weight
            #print(f"[QuantLinear] ternary_weight shape: {weight.shape}") # Debugging
        else:
            weight = self.weight
            #print(f"[QuantLinear] original weight shape: {weight.shape}")
        
        #print(f"[QuantLinear] Input shape after dtype conversion: {input.shape}") # Debugging

        input = input.to(self.dtype)
        weight = weight.to(self.dtype)
        bias = self.bias.to(self.dtype) if self.bias is not None else None

        assert input.dtype == weight.dtype == (bias.dtype if bias is not None else input.dtype), (
            f"[QuantLinear] Dtype mismatch: input={input.dtype}, weight={weight.dtype}, bias={getattr(bias, 'dtype', None)}"
        )

        # Final matrix multiplication
        try:
            output_reshaped = F.linear(input_reshaped, weight.t(), bias) # Perform matrix multiplication (linear layer)
        except RuntimeError as e:
            raise RuntimeError(f"[{self.name}] F.linear failed: input={input_reshaped.shape}, "
                               f"weight={weight.shape}, bias={None if bias is None else bias.shape}") from e

        output = output_reshaped.view(batch_size, seq_len, -1)  # Reshape the output back to the expected shape: [batch_size, seq_len, output_features]
        #print(f"[{self.name}] Output shape: {output.shape}")
        return output

    def dequantize(self: QuantLinear, tensor: torch.Tensor | Any) -> torch.Tensor | Any:
        """ Helper method to dequantize the weights """
        if self.q_int_weight is not None and self.scale is not None and self.zero_point is not None:
            return (tensor.float().to(self.dtype) - self.zero_point) * self.scale
        else:
            return tensor.float().to(self.dtype)

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


def add_activation_hooks(
        model: nn.Module,
        dtype: torch.dtype,
        layers_to_quant: List[str] | None,
        symmetric: bool = True,
        num_bits: int = 8,
        track: bool = False
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
def fusion_frame_transform(tensor, redundancy: int = 2):
    B = redundancy
    in_features = tensor.shape[1]
    U = torch.eye(in_features).repeat(B, 1).to(tensor.device)  # [B * in_features, in_features]
    V = tensor @ U.T  # [out_features, B * in_features]
    return V, U

def inverse_fusion_frame_transform(V, U):
    # Reconstruct original tensor via pseudo-inverse
    return V @ torch.pinverse(U.T)

# ===================== Main Quantization Logic ============================
def apply_FFQ(
        model:Any,
        mode:str='1.58bit',
        model_half:bool=False,
        quant_half:bool=False,
        layers_to_quant:Optional[List[str]|None]=None,
        act_quant:bool=False,
        act_bits:int=8,
        dropout_prob:float=0.1,  # ← 10% dropout during quantization
        redundancy:int=2,
        frame_dropout_prob:float=0.0
    ) -> Any:
    """
    Apply PTQ / FFQ / FrameQuant: Flexible Low-Bit Quantization for Transformers

    PARAMS:
        model:
            Any model (tested for LLaMAs and OLMos)
        mode:str:
            quant styles; '1.58bit', '1bit_sym', '1bit_asym', '2bit_sym', '2bit_asym', '4bit_sym', '4bit_asym', '8bit_sym', '8bit_asym'
        model_half:bool:
            if True model in float16 else model in float32
        quant_half:bool:
            if True quant in float16 else model in float32
            if mode is '1.58-bit', dtype is cast to float32 for quantization because of tau, and then converted to float16 if quant_half is True
        dropout_prob:float:
            dropout_prob needed for FFQ
        layers_to_quant:Optional[List[str]|None]:
            Example usage:
                style_1 = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
                style_2 = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj' ]
                style_3 = ['q_proj', 'k_proj', 'v_proj']
                style_4 = ['q_proj', 'k_proj', 'v_proj', 'mlp']
        act_quant:bool:
            if true, symmetric activation ptsq
        act_bits:int:
            default 8-bit symmetric quantization (BitNet-style)
        dropout_prob:float:
            dropout during quantization
        redundancy:int
        frame_dropout_prob:float

    RETURNS:
        PTDQ or PTSQ float16 or float32 precision model with quantized float16 or float32 nn.Linear transformer layers replaced by QuantLinear Modules
    """
    model.to(torch.float16) if model_half else model.to(torch.float32)
    dtype = torch.float16 if quant_half else torch.float32

    print(f"|| Quant Configs: {mode} | dtype: {dtype} | dropout prob: {dropout_prob} | FFQ as: {'PTSQ' if act_quant else 'PTDQ'} ||")

    def get_quant_policy(name:str):
        """ Dynamically define how each layer should be quantized """
        quantize = True
        this_mode = mode
        this_redundancy = redundancy

        # Never quantize these
        if any(skip in name for skip in ['embed_tokens', 'lm_head', 'norm', 'layernorm']):
            quantize = False

        # Q/K/V projections: use safer config
        if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']):
            #this_mode = '8bit_sym'  # Use higher-precision quantization
            this_redundancy = 1

        return quantize, this_mode, this_redundancy

    def quantize_and_replace(name, module):
        quantize, this_mode, this_redundancy = get_quant_policy(name)
        if not quantize:
            print(f"[SKIP] {name} | Policy: skip quantization")
            return

        # Avoid fused QKV (already handled)
        if any(k in name.lower() for k in ['qkv', 'qkvs', 'qk_proj', 'proj_qkv']) and module.weight.shape[0] % 3 == 0:
            print(f"[SKIP] {name} | Likely fused QKV projection")
            return

        if module.weight.shape[0] != module.weight.shape[1]:
            print(f"[SKIP] {name} | Non-square weight: {module.weight.shape}")
            return

        quantized = QuantLinear(module, dtype=dtype, dropout_prob=dropout_prob, name=name)
        w = module.weight.data.to(dtype)

        if dropout_prob > 0.0:
            dropout_mask = (torch.rand_like(w) >= dropout_prob).float()
            w = w * dropout_mask
            quantized.dropout_mask = dropout_mask

        if this_mode == '1.58bit':
            w = module.weight.data.to(torch.float32)
            q_w, tau = quantize_ternary(w)
            # SCALE ternary weights with alpha (important!) output = input @ (α * {-1, 0, 1}) else output = input @ {-1, 0, 1}
            alpha = w.abs().mean()  # Or use something smarter if you want, like layer-wise std or per-channel scaling
            # BitNet sometimes normalize weights by max(|w|) or use quantiles instead of the mean!
            q_w_scaled = (q_w * alpha).to(dtype)
            quantized.ternary_weight = q_w_scaled
            quantized.weight.data = q_w_scaled  # Optionally used as backup
            #print(f"[1.58-bit] {name} | τ = {tau:.4f}")
        else:
            n_bits = int(this_mode.replace('bit', '').replace('_sym', '').replace('_asym', ''))
            symmetric = '_sym' in this_mode
            V, U = fusion_frame_transform(w, redundancy=this_redundancy)
            V = V * (torch.rand_like(V) > frame_dropout_prob)
            q_V, q_int, scale, zp = quantize_uniform(V, n_bits, symmetric=symmetric, dtype=dtype)
            w_recon = inverse_fusion_frame_transform(q_V, U)
            quantized.weight.data = w_recon.to(dtype)

            quantized.q_int_weight = q_int
            quantized.scale = torch.tensor(scale)
            quantized.zero_point = torch.tensor(zp)
            quantized.U = U
            #print(f"[{n_bits}-bit] {name} | scale={scale:.4f} zp={zp:.4f}")

        set_module_by_name(model, name, quantized)
        #print(f"[QuantLinear WRAP] Replacing {name} | Orig weight shape: {module.weight.shape}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if layers_to_quant is None or any(layer in name for layer in layers_to_quant):
                quantize_and_replace(name, module)

    if act_quant:
        add_activation_hooks(
            model,
            dtype=dtype,
            layers_to_quant=layers_to_quant,
            symmetric=True,
            num_bits=act_bits
        )

    print("Quantization complete!")
    return model
