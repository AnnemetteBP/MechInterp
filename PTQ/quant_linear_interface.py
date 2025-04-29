from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any,  Literal, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

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