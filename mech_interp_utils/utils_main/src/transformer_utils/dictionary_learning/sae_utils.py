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
from helper_utils.enum_keys import DirPath


""" ******************** TOPK ******************** """

def get_top_tokens_per_concept(codes:Any, tokens:Any, top_k:int=5) -> Dict:
    """ Returns a dict mapping concept neuron index -> top-k tokens that activate it. """

    top_tokens = {}
    for i in range(codes.shape[1]):  # num of concept neurons
        concept_vals = codes[:, i]
        top_indices = torch.topk(concept_vals, k=min(top_k, len(tokens))).indices
        top_tokens[i] = [tokens[idx] for idx in top_indices]
    
    return top_tokens


def print_top_concepts(top_tokens_dict:Dict, num_concepts:int=5) -> None:
    """ Print topk """

    print(f"Top {num_concepts} Concepts & Their Top Tokens:")
    for i in range(min(num_concepts, len(top_tokens_dict))):
        print(f"Concept {i:03}: {top_tokens_dict[i]}")


""" ******************** MISC ******************** """

def sae_reconstruction(hidden:Any, sae:object) -> torch.Tensor:
    """ SAE Reconstruction Error (per token / feature)
        Error between original activations and SAE reconstructions.
        Usable this as a proxy for how interpretable or compressible the representations are, which may change post-quantization. """
    
    recon = sae(hidden)
    error = torch.nn.functional.mse_loss(recon, hidden)

    return error


def cos_sim_compare(fp_acts:Any, ptq_acts:Any, dim:int=1) -> torch.Tensor:
    """ Cosine Similarity Between Models
            Compute similarity between activation vectors for same prompt across models.
            High divergence indicates representation drift post-quantization. """
    
    cos = torch.nn.functional.cosine_similarity(fp_acts, ptq_acts, dim)
    return cos


def concept_entropy(codes:Any) -> torch.Tensor:
    """ Concept Entropy per Token:
            Tokens with low entropy = few concepts dominate = more interpretable.
            High entropy = distributed / ambiguous.
            → can be plotted alongside saliency. """
    
    probs = torch.softmax(codes.abs(), dim=1)
    entropy = -(probs * probs.log()).sum(dim=1)
    return entropy


def dict_orthogonality(dict_weights):
    """ Dictionary Diversity / Orthogonality:
            This helps one learn: are the learned concepts distinct?
            → A high score here means concepts aren't redundant (which is nice!). """
    
    # Cosine sim of dictionary atoms
    normed = F.normalize(dict_weights, dim=1)
    cosine_sim = normed @ normed.T
    diversity = 1 - cosine_sim.abs().mean()
    return diversity


def concept_usage(codes):
    """ Concept Usage Histogram:
            How often is each concept used (activated) across the dataset?
            → can plot this to see if a few concepts dominate — or if it's balanced. """
    
    usage = (codes.abs() > 0.01).float().sum(dim=0)
    return usage
