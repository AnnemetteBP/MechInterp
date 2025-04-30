from __future__ import annotations
from typing import Dict
import os
import torch
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt


class QuantDebugger:
    def __init__(self,
                 log_activations:bool=True,
                 log_weights:bool=True,
                 plot:bool=True) -> None:
        
        self.activation_diffs = {}
        self.weight_diffs = {}
        self.plot = plot
        self.log_activations = log_activations
        self.log_weights = log_weights


    def compare_activations(self:QuantDebugger, name:str, original:torch.Tensor, quantized:torch.Tensor) -> None:
        """ Compute MAE between original and quantized activations """
        
        if not self.log_activations:
            return

        diff = (original - quantized).abs().mean().item()
        self.activation_diffs[name] = diff

        if self.plot:
            self.plot_histograms(original, quantized, title=f"Activation: {name}")


    def compare_weights(self:QuantDebugger, name:str, original:torch.Tensor, quantized:torch.Tensor) -> None:
        """ Compute MAE between original and quantized weights """

        if not self.log_weights:
            return

        diff = (original - quantized).abs().mean().item()
        self.weight_diffs[name] = diff

        if self.plot:
            self.plot_histograms(original, quantized, title=f"Weight: {name}")


    def plot_histograms(self:QuantDebugger, original, quantized, title="", save_path=None):
        original = original.detach().cpu().flatten().numpy()
        quantized = quantized.detach().cpu().flatten().numpy()

        plt.figure(figsize=(8, 3))
        plt.hist(original, bins=100, alpha=0.5, label="Original")
        plt.hist(quantized, bins=100, alpha=0.5, label="Quantized")
        plt.title(title)
        plt.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"[DEBUG] Saved histogram to {save_path}")
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"[WARNING] Plotting failed: {e}")
        
        plt.close()


    def summarize(self:QuantDebugger) -> None:
        """ Print or return a summary of all MAEs """

        print("\n==== Activation Distortion Summary ====")
        for name, diff in sorted(self.activation_diffs.items(), key=lambda x: -x[1]):
            print(f"{name}: {diff:.6f}")

        print("\n==== Weight Distortion Summary ====")
        for name, diff in sorted(self.weight_diffs.items(), key=lambda x: -x[1]):
            print(f"{name}: {diff:.6f}")
