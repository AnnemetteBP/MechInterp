from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ptq_utils import set_module_by_name

from .bitlinear_wrapper_class import BitLinear
from .quant_debugger import QuantDebugger
from .activation_stats_collector import ActivationStatsCollector

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
    toks = tokenizer.encode(text, max_length=256, truncation=True, padding="max_length", return_tensors="pt")  # LongTensor
    return toks

# ===================== Main Quantization Logic ============================
def applyPTQ(
        model:Any,
        tokenizer:Any,
        calibration_input:Optional[str|None],
        mode:str='1.58bit',
        safer_quant:bool=True,
        model_half:bool=False,
        quant_half:bool=False,
        layers_to_quant_weights:Optional[List[str]]|None=None,
        layers_to_quant_activations:Optional[List[str]]|None=None,
        fragile_layers:bool=False,
        act_quant:bool=False,
        act_bits:int|None=8,
        smart_aplha:bool=True,
        deterministic:bool=True,
        debugging:bool=False,
        plot_debugging:bool=False,
        plot_quantization:bool=False, # same as plot debugging now so remove
        freeze_modules:bool=True,
    ) -> Any:

    
    if calibration_input is None:
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time, a knight rode into battle.",
            "Artificial intelligence is reshaping the future.",
            "Data science blends statistics and programming.",
            "Quantum mechanics challenges our understanding of reality."
        ]
    else:
        calibration_texts = calibration_input

    model.to(torch.float16) if model_half else model.to(torch.float32)
    dtype = torch.float16 if quant_half else torch.float32

    if mode != '1.58bit':
        raise NotImplementedError("[ERROR] Only '1.58bit' mode currently supported!")

    debugger = QuantDebugger(log_activations=True, log_weights=True, plot=plot_debugging) if debugging else None

    collector = ActivationStatsCollector()
    hooks = collector.register_hooks(model)

    default_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    if layers_to_quant_weights is None:
        layers_to_quant_weights = default_layers
    
    if layers_to_quant_activations is None:
        layers_to_quant_activations = default_layers

    print(f"|| Quant Configs: {mode} | BitNet-style PTQ as: {'PTSQ' if act_quant else 'PTDQ'} ||")

    def get_quant_policy(name: str) -> Tuple[bool, bool, str, int]:
        quantize_weights = any(layer in name for layer in layers_to_quant_weights)
        quantize_activations = any(layer in name for layer in layers_to_quant_activations) if act_quant else False
        this_mode = mode

        if safer_quant:
            if any(skip in name for skip in ['embed_tokens', 'lm_head', 'norm', 'layernorm']):
                quantize_weights = False
                quantize_activations = False

        return quantize_weights, quantize_activations, this_mode

        
    bitlinear_layers = []
    print(">> [STEP 1] Wrapping Linear layers (no weight quant yet)...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantize_weights, quantize_activations, this_mode = get_quant_policy(name)
            if not quantize_weights:
                print(f"[SKIP] {name} | Policy: Skip weight quantization!")
                continue
            if this_mode == '1.58bit':
                quantized = BitLinear(
                    module,
                    dtype=dtype,
                    name=name,
                    act_quant=quantize_activations,
                    act_bits=act_bits,
                    use_ternary=True,
                    smart_alpha=smart_aplha,
                    plotting=plot_quantization
                )
                """if 'layers.0' in name or 'layers.1' in name:
                    module.act_quant = False"""
                if 'layers.0' in name or 'layers.1' in name:
                    quantized.act_quant = False
                if debugger is not None:
                    quantized.debugger = debugger
                set_module_by_name(model, name, quantized)
                bitlinear_layers.append((name, quantized))
            else:
                raise NotImplementedError("[ERROR] Only '1.58bit' mode with BitLinear currently supported!")

    # Ensure calibration_texts is always a list
    if not isinstance(calibration_texts, list):
        calibration_texts = [calibration_texts]
    # Disable ternary mode globally before activation calibration
    for _, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.use_ternary = False

    if act_quant:
        print(">> [STEP 2] Run calibration pass with full-precision weights to collect correct activation stats...")
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for activation calibration!")

        @torch.no_grad()
        def calibrate_model(model, tokenizer):
            model.eval()
            scale_log = {}

            def collect_activation(module, input, output):
                if isinstance(module, BitLinear) and module.act_quant:
                    print(f"[Hook] {module.name} input shape: {input[0].shape}")
                    module.calibrate_activation(input[0], act_bits=act_bits, force=True)
                    scale_log[module.name] = module.act_scale.item()
                    for name, scale in scale_log.items():
                        print(f"{name:40s} | scale: {scale:.6f}")

            hooks = []
            for m in model.modules():
                if isinstance(m, BitLinear) and m.act_quant:
                    hook = m.register_forward_hook(collect_activation)
                    hooks.append(hook)

            for text in calibration_texts:
                if not text.strip():
                    continue
                dummy_input = text_to_input_ids(tokenizer=tokenizer, text=text).to(next(model.parameters()).device)
                if dummy_input.numel() == 0:
                    continue
                _ = model(dummy_input)  # Run forward pass

            # Remove hooks after calibration
            for hook in hooks:
                hook.remove()

        calibrate_model(model, tokenizer)

        if fragile_layers:
            fragile_layers = [name for name, metrics in collector.stats.items() if metrics['std'] < 1e-5] # 1e-4
            print(f"[INFO] Fragile layers detected (std < 1e-5): {fragile_layers}")
            print("[INFO] Deactivating act quant for fragile layers:")
            for name in fragile_layers:
                print(f"  - {name}")
                for n, m in model.named_modules():
                    if isinstance(m, BitLinear) and name in n:
                        m.act_quant = False
            # Clean up the stats hooks
            for h in hooks:
                h.remove()

    # Re-enable ternary quantization before weight quantization
    for _, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.use_ternary = True

    print(">> [STEP 3]: Quantizing weights (after activation calibration)...")
    for name, module in bitlinear_layers:
        module.quantize(module.orig_weight, deterministic=deterministic)
        print(f"[1.58-bit] {name} | τ={module.tau:.4f} | α={module.alpha:.4f} | shape={module.orig_weight.shape}")

    if debugger:
        debugger.summarize()

    collector.report(sort_by='std', top_k=20)
    print(">> Quantization complete!")

    if freeze_modules:
        print(">> [STEP 4]: Freezing modules...")
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.freeze_quantization()

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and hasattr(module, "input_activation"):
            print(name, "mean act:", module.input_activation.mean().item())

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module.act_quant:
            print(f"{name:50s} | act_scale: {module.act_scale.item():.6f}")

    return model