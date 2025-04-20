from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



def plot_weight_distribution(weights_list:List[Any], labels:Any, save_path:str) -> None:
    """ 
    Weight Distribution Histograms / KDEs - Compare how the weights are distributed:
        Plots: Flattened weights from q_proj.weight (or other layers) as a histogram or kernel density estimate (KDE)
        Insights:
            Symmetric quantizers will often have "centered" weight distributions
            Asymmetric ones might shift to accommodate range
            Low-bit quantized weights (when dequantized) will show discrete clusters or,
            “spiky” histograms due to the few quantized values and per-group scaling.
    """

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 5))

    for w, label in zip(weights_list, labels):
        sns.kdeplot(w.flatten().cpu().numpy(), label=label, bw_adjust=0.5)
    
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
    plt.legend()
    
    plt.savefig(save_path)
    plt.show()


def plot_singular_values(weight:Any, label:Any, save_path:str) -> None:
    """
    SVD / PCA on Weights - Run Singular Value Decomposition on the weight matrices.
     Insights:
        The spectrum of singular values shows how much information is preserved.
        Quantization often reduces low-rank structure.
    """

    u, s, v = torch.svd(weight)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(s.cpu().numpy(), label=label)
    plt.savefig(save_path)


def plot_q_hist(tensors:List[torch.Tensor], labels:List[str], save_path:str):

    for t, label in zip(tensors, labels):
        values = t.flatten().cpu().numpy()
        plt.hist(values, bins=100, alpha=0.6, label=label)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.legend()
    plt.title("Distribution of Quantized Integer Values")
    plt.xlabel("Quantized Value")
    plt.ylabel("Frequency")

    plt.savefig(save_path)
    plt.show()
