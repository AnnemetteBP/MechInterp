from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_binary_tensor(tensor: torch.Tensor) -> torch.ByteTensor:
    """ Assumes binary tensor with values in {-1, 1} - Binary Packing Example (8 weights in 1 byte) """

    binary = (tensor > 0).to(torch.uint8)
    padded = F.pad(binary.view(-1), (0, (8 - binary.numel() % 8) % 8))
    packed = torch.zeros(padded.numel() // 8, dtype=torch.uint8)

    for i in range(8):
        packed |= (padded[i::8] << (7 - i))

    return packed


def pack_ternary_tensor(tensor: torch.Tensor) -> torch.ByteTensor:
    """ Ternary Packing (1.58 bits/weight) - Trickier since ternary needs 2 bits/weight to store safely, but one can do:
            Map: {-1 → 0b00, 0 → 0b01, 1 → 0b10}
            Pack 4 weights into 1 byte → bit_shift = [6, 4, 2, 0] """
    
    mapping = {-1: 0b00, 0: 0b01, 1: 0b10}
    ternary = torch.tensor([mapping[int(w.item())] for w in tensor.flatten()])
    padded = F.pad(ternary, (0, (4 - len(ternary) % 4) % 4))
    packed = torch.zeros(padded.numel() // 4, dtype=torch.uint8)
    
    for i in range(4):
        packed |= (padded[i::4] << (6 - 2*i))

    return packed


def pack_int4_tensor(tensor: torch.Tensor) -> torch.ByteTensor:
    """ Assumes tensor values are in 0..15 (unsigned 4-bit) - int4 Packing (2 weights per byte) """
    
    tensor = torch.clamp(tensor, 0, 15)

    if tensor.numel() % 2 != 0:
        tensor = F.pad(tensor.view(-1), (0, 1))

    flat = tensor.flatten().to(torch.uint8)
    high = flat[0::2] << 4
    low = flat[1::2]
    return (high | low)


def pack_8bit_uniform(tensor: torch.Tensor) -> torch.ByteTensor:
    """ Flattens and casts the tensor to 8-bit for analysis.
        Assumes input is already quantized to range [-1, 1] mapped to 256 levels. """
    
    return tensor.flatten().to(torch.uint8)
