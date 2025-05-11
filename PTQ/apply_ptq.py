from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal, Callable
import os
import random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .ptq_utils import set_module_by_name
from .bitlinear_wrapper_class import BitLinear
from .quant_debugger import QuantDebugger
from .activation_stats_collector import ActivationStatsCollector

import warnings
warnings.filterwarnings('ignore')


# Load a model with pretrained weights and quantized parameters
def load_quantized_model(model_path, model_class=LlamaForCausalLM, tokenizer_class=AutoTokenizer):
    model = model_class.from_pretrained(model_path)
    state_dict = torch.load(model_path, map_location="cpu")

    # Load quantized weights and parameters
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Load ternary weight, alpha, tau and activation scale for each BitLinear layer
            bitlinear = BitLinear.from_linear(module.linear_layer, act_quant=True, use_ternary=True)
            bitlinear.ternary_weight = state_dict.get(f"{name}.ternary_weight")
            bitlinear.alpha = state_dict.get(f"{name}.alpha")
            bitlinear.tau = state_dict.get(f"{name}.tau")
            bitlinear.act_scale = state_dict.get(f"{name}.act_scale")
            model._modules[name] = bitlinear

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_dummy_input(model, seq_len=128):
    hidden = model.config.hidden_size
    return torch.randn(1, seq_len, hidden).to(next(model.parameters()).device)

def text_to_input_ids(tokenizer: Any, text: str) -> Dict[str, torch.Tensor]:
    """Encode input text into token IDs and attention mask for model input."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

def save_quantized_model_with_bitlinear(model: nn.Module, save_path: str):
    # Save the model state dict (including quantized parameters and buffers)
    model_state = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Ensure we save the ternary weights and other quantization buffers
            model_state[f'{name}.ternary_weight'] = module.ternary_weight
            model_state[f'{name}.alpha'] = module.alpha
            model_state[f'{name}.tau'] = module.tau
            if module.act_quant:
                model_state[f'{name}.act_scale'] = module.act_scale

    torch.save(model_state, save_path)
    print(f"Quantized model saved to {save_path}")


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
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    #torch.set_float32_matmul_precision("high")

def strip_unused_quant_buffers(model):
    for module in model.modules():
        if isinstance(module, BitLinear):
            # Optionally keep orig_weight if you want to support dequantization later
            if hasattr(module, "orig_weight"):
                delattr(module, "orig_weight")
            if hasattr(module, "last_quantized_act"):
                delattr(module, "last_quantized_act")

def applyPTQ(
        model:Any,
        tokenizer:Any,
        calibration_input:Optional[str|None],
        mode:str='1.58bit',
        safer_quant:bool=True,
        q_lmhead:bool=True,
        model_half:bool=False,
        quant_half:bool=False,
        layers_to_quant_weights:Optional[List[str]]|None=None,
        layers_to_quant_activations:Optional[List[str]]|None=None,
        fragile_layers:bool=False,
        act_quant:bool=True,
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
    if layers_to_quant_weights is None or layers_to_quant_activations is None:
        layers_to_quant_weights = default_layers
        layers_to_quant_activations = default_layers

    print(f"|| Quant Configs: {mode} | BitNet-style PTQ as: {'PTSQ' if act_quant else 'PTDQ'} ||")

    def get_quant_policy(name: str) -> Tuple[bool, bool, str, int]:
        quantize_weights = any(layer in name for layer in layers_to_quant_weights)
        quantize_activations = any(layer in name for layer in layers_to_quant_activations) if act_quant else False
        this_mode = mode
        q_layers = ['embed_tokens', 'lm_head', 'norm', 'layernorm'] if q_lmhead else ['embed_tokens', 'norm', 'layernorm']
        if safer_quant:
            if any(skip in name for skip in q_layers): # no impact 'lm_head' for 7B+: https://medium.com/@NeuralCompressor/10-tips-for-quantizing-llms-and-vlms-with-autoround-923e733879a7#:~:text=However%2C%20quantizing%20the%20LM%2Dhead,provides%20a%20reasonable%20compression%20rate.
                #quantize_weights = False
                quantize_weights = True if layers_to_quant_weights == ["*"] else any(layer in name for layer in layers_to_quant_weights)
                quantize_activations = True if layers_to_quant_activations == ["*"] else any(layer in name for layer in layers_to_quant_activations)

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
                    quantized.act_quant = False"""
                if debugger is not None:
                    quantized.debugger = debugger
                set_module_by_name(model, name, quantized)
                bitlinear_layers.append((name, quantized))
            else:
                raise NotImplementedError("[ERROR] Only '1.58bit' mode with BitLinear currently supported!")

    if not isinstance(calibration_texts, list):
        calibration_texts = [calibration_texts]
    # Disable ternary mode globally before activation calibration
    for _, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.use_ternary = False
    
    if torch_backends:
        set_deterministic()

    # Ensure that calibration is done before freezing quantization
    if act_quant:
        print(">> [STEP 2] Running activation calibration forward pass...")
        model.eval()
        with torch.no_grad():
            for text in calibration_texts:
                input_dict = text_to_input_ids(tokenizer=tokenizer, text=text)
                input_dict = {k: v.to(next(model.parameters()).device) for k, v in input_dict.items()}
                if input_dict["input_ids"].numel() == 0:
                    continue
                _ = model(**input_dict)


        # Optional: print calibrated scales
        for name, module in model.named_modules():
            if isinstance(module, BitLinear) and module.act_quant:
                print(f"[Calibrated] {name:50s} | act_scale: {module.act_scale.item():.6f}")

    # Re-enable ternary quantization before weight quantization
    for _, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.use_ternary = True
    print(">> [STEP 3] Applying weight quantization...")
    
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module.use_ternary:
            module.quantize()  # Now supported with no args
            print(f"[Quantized] {name} | Ternary weights applied.")

    # Freezing quantization after calibration is done
    if freeze_modules:
        print(">> [STEP 4]: Freezing modules...")
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.freeze_quantization()  # disables recalibration, debug mode, etc.
            for param in m.parameters(): 
                param.requires_grad = False  # prevents accidental updates

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and hasattr(module, "input_activation"):
            print(name, "mean act:", module.input_activation.mean().item())

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module.act_quant:
            print(f"{name:50s} | act_scale: {module.act_scale.item():.6f}")

    if save_model_path:
        strip_unused_quant_buffers(model)
        model.save_pretrained(save_model_path, safe_serialization=True)
        #save_quantized_model_with_bitlinear(model=model, save_path=save_model_path)

    model.eval()
    for m in model.modules():
        if isinstance(m, BitLinear):
            for p in m.parameters():
                p.requires_grad = False
    
    return model

