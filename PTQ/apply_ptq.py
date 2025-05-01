from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal
import random, numpy as np
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

def set_deterministic(seed=42):
    """ 
    Forces PyTorch to use only deterministic operations (e.g., disables non-deterministic GPU kernels).
    This is crucial for reproducibility: given the same inputs and model state, to get the same outputs every time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

def save_quantized(model:Any, save_path:str) -> None:
    torch.save(model.state_dict(), f"{save_path}.pt")
    print(f"[INFO] Model saved to: {save_path}!")

def load_quantized(model:Any, model_path:str, model_to_eval:bool=False) -> Any:
    saved_model = model.load_state_dict(torch.load(f"{model_path}.pt"))
    model.eval() if model_to_eval else f"[INFO] Model not set to eval mode!"
    return saved_model

def force_eval_mode(model:Any) -> None:
    model.eval()
    for m in model.modules():
        if hasattr(m, 'eval'):
            m.eval()
    print("[INFO] Model set to eval mode!")

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
        torch_backends:bool=True,
        debugging:bool=False,
        plot_debugging:bool=False,
        plot_quantization:bool=False,
        freeze_modules:bool=True,
        save_model_path:str|None=None
    ) -> Any:
    """
    Apply post-training quantization (PTQ) to a transformer-style model using BitNet-style ternary weight quantization (1.58-bit)
    and optional activation quantization. Supports fragile layer detection, layerwise policy control, and calibration over a text corpus.
    Model can be e.g., olmo, llama.

    Args:
        model (Any): A PyTorch model with `nn.Linear` modules to be replaced by BitLinear.
        tokenizer (Any): A tokenizer for converting text into model input.
        calibration_input (str | List[str] | None): Text(s) for calibration.
        mode (str): Quantization mode. Only '1.58bit' is supported.
        safer_quant (bool): Skip quantization on sensitive modules (e.g. LayerNorm).
        model_half (bool): Cast model to FP16 before quantization.
        quant_half (bool): Quantize weights/activations in FP16 instead of FP32.
        layers_to_quant_weights (List[str]): Layer substrings for weight quantization.
        layers_to_quant_activations (List[str]): Layer substrings for activation quantization.
        fragile_layers (bool): Disable activation quantization on low-variance layers.
        act_quant (bool): Enable activation quantization.
        act_bits (int): Bits for symmetric activation quantization (default: 8).
        smart_aplha (bool): Use adaptive alpha scaling for ternary weights.
        deterministic (bool): Use deterministic ternary quantization.
        debugging (bool): Enable hooks to log quantization process.
        plot_debugging (bool): plot debugger.
        plot_quantization (bool): plot activation quantization from BitLinear.
        freeze_modules (bool): Freeze quantized modules (no further training).

    Returns:
        model (Any): Quantized model with `BitLinear` modules replacing eligible `nn.Linear`.

    Quantization Validity (Formalism):
        Let Q_w(m) := "weights of module m are quantized to ternary {−α, 0, +α}"  
        Let Q_a(m) := "activations of module m are quantized to N-bit fixed-point"  
        Let BL(m) := "m is instance of BitLinear"

        For all m in model.modules:
            BL(m) => (Q_w(m) ∨ Q_a(m)) ∧ (Q_w(m) ↔ m.orig_weight != None and is_discrete(m.weight))

        Where:
            is_discrete(w) := ∀x ∈ w, x ∈ {−α, 0, +α}
    """
    
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
    
    if model_half:
        torch.backends.cuda.matmul.allow_tf32 = False

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
                    deterministic=deterministic,
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

        if torch_backends:
            set_deterministic()
        
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
    print(">> [INFO] Quantization complete!")

    if freeze_modules:
        print(">> [STEP 4]: Freezing modules...")
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.freeze_quantization() # disables recalibration, debug mode, etc.
            for param in m.parameters(): 
                param.requires_grad = False  # prevents accidental updates

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and hasattr(module, "input_activation"):
            print(name, "mean act:", module.input_activation.mean().item())

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module.act_quant:
            print(f"{name:50s} | act_scale: {module.act_scale.item():.6f}")

    if save_model_path:
        save_quantized(model=model, save_path=save_model_path)

    model.eval()
    for m in model.modules():
        if isinstance(m, BitLinear):
            for p in m.parameters():
                p.requires_grad = False
    
    return model
