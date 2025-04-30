from __future__ import annotations
from typing import List, Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ActivationStatsCollector:
    def __init__(self) -> None:
        self.stats = {}

    """ Apply wrapper before quantixation """

    def hook(self, name):
        def collect(module, input, output):
            act = input[0].detach().cpu().flatten()
            self.stats[name] = {
                'max': act.max().item(),
                'min': act.min().item(),
                'std': act.std().item(),
                'mean': act.mean().item(),
                'sparsity': (act == 0).float().mean().item(),
                'unique_vals': act.unique().numel()
            }
        
        return collect

    def register_hooks(self:ActivationStatsCollector, model:Any) -> List:
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook_fn = self.hook(name)
                hooks.append(module.register_forward_hook(hook_fn))
        
        return hooks

    def report(self:ActivationStatsCollector, sort_by:str='std', top_k:int=10) -> None:
        stats_list = []
        for name, metrics in self.stats.items():
            stats_list.append((name, metrics))
        
        stats_list.sort(key=lambda x: x[1][sort_by], reverse=False) # small std = more dangerous
        print(f"\n[Sensitivity Report] Sorted by {sort_by}:")
        
        for name, metrics in stats_list[:top_k]:
            print(f"{name} | max: {metrics['max']:.5e} | std: {metrics['std']:.5e} | sparsity: {metrics['sparsity']:.2%}")
        

    def visualize(self:ActivationStatsCollector, layer_name:Any) -> None:
        if layer_name not in self.stats:
            print(f"No data for {layer_name}")
            return
        # e.g., for deeper analysis, store activations in a dict
