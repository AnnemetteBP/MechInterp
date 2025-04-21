from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os, json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.stats import wasserstein_distance, entropy

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

from .sae_class import SAE


""" ******************** PLOTTING ******************** """
def plot_norms(activations:Any, full_path:str) -> None:
    """ Activation Norms per Layer (single layer SAE):
            Compute and plot the L2 norm (or other norm) of hidden states or intermediate activations per layer.
            Quantized models often reduce the range of activation values, which can be detected via shrinking norms.
        Input example: activations = [layer_data['hidden'] for layer_data in all_layer_outputs.values()] """

    norms = [torch.norm(activation, dim=-1).mean().item() for activation in activations]
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(norms)
    plt.savefig(full_path) # actnorms_
    

def compare_norms(norm_list_fp16:List, norm_list_ptq:List, full_path:str) -> None:
    """ Compare activation norms
        Input example:
            norms_fp16 = [torch.norm(layer['hidden'], dim=-1).mean().item() for layer in fp16_data.values()]
            norms_ptq = [torch.norm(layer['hidden'], dim=-1).mean().item() for layer in ptq_data.values()]
            compare_norms(norms_fp16, norms_ptq) """
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(8,6))
    
    plt.plot(norm_list_fp16, label='FP16', marker='o')
    plt.plot(norm_list_ptq, label='PTQ', marker='x')
    
    plt.title("Activation Norms per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean L2 Norm")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(full_path) # norm_comparison_
    plt.show()


def compare_code_distributions(code_a, code_b) -> Tuple[Any,Any]:
    """ Wasserstein or KL Divergence Between SAE Code Distributions
        Useful for comparing latent concept drift between models!
        Example: compare_code_distributions(fp16_codes, ptq_codes) """
    
    # Flatten across tokens
    a = code_a.flatten().cpu().numpy()
    b = code_b.flatten().cpu().numpy()
    
    # Normalize histograms
    hist_a, _ = np.histogram(a, bins=100, range=(-5,5), density=True)
    hist_b, _ = np.histogram(b, bins=100, range=(-5,5), density=True)

    wd = wasserstein_distance(hist_a, hist_b)
    kl = entropy(hist_a + 1e-8, hist_b + 1e-8)
    
    print(f"Wasserstein: {wd:.4f} | KL Div: {kl:.4f}")
    return wd, kl


def visualize_concepts(codes:Any, full_path:str|None, method:str, layer=None) -> None:
    """ Project and visualize using PCA / 'pca' or TSNE / 'tsne' """

    codes_np = codes.cpu().numpy()

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced = reducer.fit_transform(codes_np)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=20, c='royalblue')

    title = f"{method.upper()} Projection of SAE Codes"
    if layer is not None:
        title += f" (Layer {layer})"
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    if full_path is not None:
        plt.savefig(full_path) # concepts_
    
    plt.show()