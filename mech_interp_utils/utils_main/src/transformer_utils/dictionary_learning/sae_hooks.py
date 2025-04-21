from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


""" ******************** ACTIVATION ******************** """
def get_layer_activations(model:Any, tokenizer:Any, text:str, target_layer_idx:int|List[int]) -> Tuple[Any,Any]:
    """ Hook Helper: Tokenize inputs and make activation layer hooks """

    model.eval()
    
    try:
        input_ids = tokenizer(text=text, return_tensors='pt').to(model.device)['input_ids']
    except Exception as e:
        print(f"[Tokenizer error] {e}")
        return None

    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach().cpu())

    try:
        handle = model.model.layers[target_layer_idx].register_forward_hook(hook_fn)
    except Exception as e:
        print(f"[Hook registration error] {e}")
        return None

    try:
        with torch.no_grad():
            _ = model(input_ids)
    except Exception as e:
        print(f"[Forward pass error] {e}")
        return None
    finally:
        handle.remove()

    if not activations:
        print(f"[Hook error] No activations captured for layer {target_layer_idx}")
        return None

    return input_ids[0], activations[0].squeeze(0)
