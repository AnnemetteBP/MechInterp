from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ptq_utils import (
    PTQLinear,
    FFQLinear,
    BitLinear
)

from .ptq_utils import set_module_by_name

import warnings
warnings.filterwarnings('ignore')


# === Dry Forward for Static Quant Calibration (PTSQ only) ===
@torch.no_grad()
def dry_forward(model:nn.Module, dummy_input:torch.Tensor):
    model.eval()
    _ = model(dummy_input)
    print("[PTSQ] Since quantizing activations, dry forward done for calibration!")

def get_dummy_input(model, seq_len=128):
    hidden = model.config.hidden_size
    return torch.randn(1, seq_len, hidden).to(next(model.parameters()).device)

def text_to_input_ids(tokenizer:Any, text:str) -> torch.Tensor:
    """ Encode inputs properly for embedding """
    toks = tokenizer.encode(text, return_tensors="pt")  # LongTensor
    return toks

# ===================== Main Quantization Logic ============================
def applyPTQ(
        model:Any,
        tokenizer:Any,
        calibration_input:Optional[str|None],
        mode:str='1.58bit',
        qkv_safety:bool=True,
        safer_quant:bool=False,
        ffq:bool=False,
        model_half:bool=False,
        quant_half:bool=False,
        layers_to_quant:Optional[List[str]|None]=None,
        act_quant:bool=False,
        act_bits:int=8,
        dropout_prob:float=0.0, 
        redundancy:int=3, # or 2, ...
        frame_dropout_prob:float=0.0
    ) -> Any:

        
    model.to(torch.float16) if model_half else model.to(torch.float32)
    dtype = torch.float16 if quant_half else torch.float32

    if mode != '1.58bit':
        raise NotImplementedError("[ERROR] Only '1.58bit' mode currently supported!")

    print(f"|| Quant Configs: {mode} | dtype: {dtype} | dropout prob: {dropout_prob} | {'FFQ' if ffq else 'PTQ'} as: {'PTSQ' if act_quant else 'PTDQ'} ||")

    def get_quant_policy(name:str) -> Tuple[bool, str, int]:
        """ Dynamically define quantization policy per layer. """
        quantize = True
        this_mode = mode
        this_redundancy = redundancy

        if layers_to_quant is not None:
            quantize = any(layer in name for layer in layers_to_quant)

        elif safer_quant:
            if any(skip in name for skip in ['embed_tokens', 'lm_head', 'norm', 'layernorm']):
                quantize = False

        # QKV safety fallback
        if qkv_safety:
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']):
                this_redundancy = 1

        return quantize, this_mode, this_redundancy

    def quantize_and_replace(name, module):
        quantize, this_mode, this_redundancy = get_quant_policy(name)
        if not quantize:
            print(f"[SKIP] {name} | Policy: Skip quantization")
            return

        if ffq:
            if any(k in name.lower() for k in ['qkv', 'qkvs', 'qk_proj', 'proj_qkv']) and module.weight.shape[0] % 3 == 0:
                print(f"[SKIP] {name} | Likely fused QKV projection")
                return
        
        if this_mode == '1.58bit':
            quantized = BitLinear(module, dtype=dtype, name=name, act_quant=act_quant, act_bits=act_bits, use_ternary=True, smart_alpha=True)
            quantized.quantize(module.weight.data, deterministic=True) 
            print(f"[1.58-bit] {name} | τ={quantized.tau:.4f} | α={quantized.alpha:.4f} | shape={module.weight.shape}")
            set_module_by_name(model, name, quantized)
            return

        elif ffq and 'bit' in this_mode:
            if module.weight.shape[0] != module.weight.shape[1]:
                print(f"[SKIP][FFQ] {name} | Non-square weight: {module.weight.shape}")
                return
            
            n_bits = int(this_mode.replace('bit', '').replace('_sym', '').replace('_asym', ''))
            symmetric = '_sym' in this_mode

            quantized = FFQLinear(
                orig=module,
                dtype=dtype,
                n_bits=n_bits,
                symmetric=symmetric,
                redundancy=this_redundancy,
                frame_dropout_prob=frame_dropout_prob,
                dropout_prob=dropout_prob,
                name=name
            )
            
            set_module_by_name(model, name, quantized)
            return

        else:
            quantized = PTQLinear(module, dtype=dtype, dropout_prob=dropout_prob, name=name)

            n_bits = int(mode.replace('bit', '').replace('_sym', '').replace('_asym', ''))
            symmetric = '_sym' in mode

            quantized.quantize(module.weight.data, n_bits=n_bits, symmetric=symmetric)

            print(f"[PTQ-{n_bits}bit] {name} | scale={quantized.scale.item()} zp={quantized.zero_point.item()} | shape={module.weight.shape} | unique={len(torch.unique(quantized.q_int_weight))}")

            set_module_by_name(model, name, quantized)
            return

    print(">> Starting quantization pass...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantize_and_replace(name, module)

    if act_quant:
        print(">> Starting activation calibration...")
        if tokenizer is None or calibration_input is None:
            raise ValueError("tokenizer and calibration_input must be provided for activation calibration.")

        dummy_input = text_to_input_ids(tokenizer=tokenizer, text=calibration_input)
        dummy_input = dummy_input.to(next(model.parameters()).device)

        @torch.no_grad()
        def calibrate_model(model, dummy_input):
            model.eval()

            def collect_activation(module, input, output):
                if isinstance(module, BitLinear) and module.act_quant:
                    module.calibrate_activation(input[0], act_bits=act_bits, force=True, percentile=0.9999) # 0.9995

            hooks = []
            for module in model.modules():
                if isinstance(module, BitLinear) and module.act_quant:
                    hooks.append(module.register_forward_hook(collect_activation))

            _ = model(dummy_input)

            for hook in hooks:
                hook.remove()

        calibrate_model(model, dummy_input)

    print(">> Quantization complete!")
    return model