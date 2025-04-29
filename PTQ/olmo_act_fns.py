from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn

class PatchedOlmoMLP(nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj, activation='SiLU'):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        
        act_fns = {
            'SiLU': nn.SiLU(),
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU(),
            'SquaredReLU': lambda x: torch.relu(x) ** 2
        }
        if activation == 'SquaredReLU':
            self.act_fn = act_fns['SquaredReLU']  
        else:
            self.act_fn = act_fns[activation]

    def forward(self, x):
        gated = self.gate_proj(x)
        up = self.up_proj(x)
        act = self.act_fn(up)
        return self.down_proj(gated * act)


def patch_olmo_mlp(model:Any, act_fn:int|None) -> None:
    """ Just select one from the list atm. """
    
    act_fns = ['SquaredReLU', 'ReLU', 'GELU', 'SiLU']

    for i, layer in enumerate(model.model.layers):
        layer.mlp = PatchedOlmoMLP(
            gate_proj=layer.mlp.gate_proj,
            up_proj=layer.mlp.up_proj,
            down_proj=layer.mlp.down_proj,
            activation=act_fns[act_fn] if act_fn else act_fns[1]
        )