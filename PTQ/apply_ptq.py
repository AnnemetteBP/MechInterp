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

from .ptq_utils import (
    quantize_ternary,
    quantize_uniform,
    fusion_frame_transform,
    inverse_fusion_frame_transform,
    set_module_by_name,
    add_activation_hooks
)

import warnings
warnings.filterwarnings('ignore')

# ===================== Main Quantization Logic ============================
def applyPTQ(
        model:Any,
        mode:str='1.58bit',
        ffq:bool=False,
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
    Apply Post Training Quantization to transformers.

    `model` should be familiar from the transformers library.

    `model` should be a `transformers.PreTrainedModel` e.g. LLaMA or OLMo architectures

    The `mode` arguments:
        `1.58bit`, (`1bit_sym`, `1bit_asym`), `2bit_sym`, `2bit_asym`, `4bit_sym`, `4bit_asym`, `8bit_sym`, `8bit_asym`
    
    The boolean arguments `ffq`, `model_half`, `quant_half`, `act_quant` controls:

        - Fusion FrameQuant (the default `ffq` is False):
            - False: apply 'basic' ptq
            - True:  use Fusion FrameQuant for ptq application (slower)

        - Model Precision (`model_half`):
            - False: all model parameters to float32 
            - True:  all model parameters to float16 
        
        - Quantized Precision (`quant_half`):
            - False: linear and activation quant in float32 precision
            - True:  linear and activation quant in float16 precision

        - Activation Quantization (PTSQ, `act_quant`):
            - False: apply quantization as PTDQ
            - True:  apply quantization as PTSQ

    Other params:
    `layers_to_quant`: list[str], `act_bits`: int[4 or 8], `dropout_prob`: float, `redundancy`:int, and `frame_dropout_prob`: float

    """
        
    model.to(torch.float16) if model_half else model.to(torch.float32)
    dtype = torch.float16 if quant_half else torch.float32

    print(f"|| Quant Configs: {mode} | dtype: {dtype} | dropout prob: {dropout_prob} | {'FFQ' if ffq else 'PTQ'} as: {'PTSQ' if act_quant else 'PTDQ'} ||")

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
        
        if this_mode == '1.58bit':
            quantized = BitLinear(module, dtype=dtype, name=name)
            w = module.weight.data.to(torch.float32)
            q_w, tau = quantize_ternary(w)
            alpha = w.abs().mean()
            q_w_scaled = (q_w * alpha).to(dtype)

            quantized.ternary_weight = q_w_scaled
            quantized.weight.data = q_w_scaled  # Optional

        elif ffq and 'bit' in this_mode:
            quantized = FFQLinear(module, dtype=dtype, dropout_prob=dropout_prob, name=name)
            w = module.weight.data.to(dtype)

            if dropout_prob > 0.0:
                dropout_mask = (torch.rand_like(w) >= dropout_prob).float()
                w = w * dropout_mask
                quantized.dropout_mask = dropout_mask

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
        
        else:
            quantized = PTQLinear(module, dtype=dtype, dropout_prob=dropout_prob, name=name)
            w = module.weight.data.to(dtype)
            
            n_bits = int(mode.replace('bit', '').replace('_sym', '').replace('_asym', ''))
            symmetric = '_sym' in mode
            q_w, q_int, scale, zp = quantize_uniform(w, n_bits, symmetric=symmetric, dtype=dtype)
            quantized.weight.data = q_w
            quantized.q_int_weight = q_int
            quantized.scale = torch.tensor(scale) # redundant probably
            quantized.zero_point = torch.tensor(zp) # redundant probably
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