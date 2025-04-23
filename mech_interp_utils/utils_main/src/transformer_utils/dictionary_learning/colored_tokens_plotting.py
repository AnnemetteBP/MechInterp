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
from .sae_hooks import get_layer_activations
from .misc_plotting import visualize_concepts


""" ******************** PLOT COLORED TOKENS ******************** """
def _plot_colored_tokens(
        tokens:Any,
        scores:Any,
        cmap:str='coolwarm'
) -> str:
    """ Plots the colored tokens derived from SAE """

    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)
    colors = plt.cm.get_cmap(cmap)(norm_scores.numpy())
    html = ""

    for tok, col in zip(tokens, colors):
        rgba = f"rgba({int(col[0]*255)}, {int(col[1]*255)}, {int(col[2]*255)}, 0.9)"
        html += f"<span style='background-color:{rgba}; padding:2px; margin:1px; border-radius:4px'>{tok}</span> "
    
    #return html
    return HTML(html)

def _plot_multicolored_tokens(
        tokens:Any,
        feature_token_matrix:Any,
        top_k:int=10,
        cmap:str='coolwarm'
) -> HTML:
    """
    tokens: list[str] - the tokens from the prompt
    feature_token_matrix: [num_features x num_tokens] - abs activations per feature/token
    """

    html = ""
    for i, feature_row in enumerate(feature_token_matrix[:top_k]):
        norm_scores = (feature_row - feature_row.min()) / (feature_row.max() - feature_row.min() + 1e-5)
        colors = plt.cm.get_cmap(cmap)(norm_scores.numpy())

        row_html = f"<b>Feature {i}</b>: "
        for tok, col in zip(tokens, colors):
            rgba = f"rgba({int(col[0]*255)}, {int(col[1]*255)}, {int(col[2]*255)}, 0.9)"
            row_html += f"<span style='background-color:{rgba}; padding:2px; margin:1px; border-radius:4px'>{tok}</span> "
        html += f"<div style='margin-bottom:5px'>{row_html}</div>"

    return HTML(html)


def _run_multi_layer_sae(
        model:Any,
        tokenizer:Any,
        text:str,
        multi_tokens:bool=False,
        do_log:bool=False,
        target_layers:List[int]=[5,10,15],
        vis_projection:str|None=None,
        log_path:str|None=None,
        log_name:str|None=None,
        fig_path:str|None=None    
) -> Dict:
    """ Run multi-layer SAE analysis """

    all_layer_outputs = {}

    for layer_idx in target_layers:
        print(f"\nRunning SAE on layer {layer_idx}")
        token_ids, hidden = get_layer_activations(model, tokenizer, text, target_layer_idx=layer_idx)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        sae = SAE(input_dim=hidden.shape[1], dict_size=512, sparsity_lambda=1e-3)
        sae.train_sae(hidden, epochs=10, batch_size=4)

        codes = sae.encode(hidden).detach()
        normed = F.normalize(codes, dim=1)
        saliency = normed.abs().max(dim=1).values
        """
        if multi_tokens:
            feature_token_matrix = codes.abs().T  # <--- this is what colored_tokens_multi needs
            html = _plot_multicolored_tokens(tokens=tokens, feature_token_matrix=feature_token_matrix, top_k=10)
        else:
            html = _plot_colored_tokens(tokens=tokens, scores=saliency)
        """
        if multi_tokens:
            # Rank neurons by total activation across all tokens
            total_per_feature = codes.abs().sum(dim=0)  # [dict_size]
            topk_indices = total_per_feature.topk(10).indices
            # Build matrix: top-k neurons × tokens
            feature_token_matrix = codes[:, topk_indices].abs().T  # [10, num_tokens]
            html = _plot_multicolored_tokens(
                tokens=tokens,
                feature_token_matrix=feature_token_matrix,
                top_k=10
            )
        else:
            html = _plot_colored_tokens(tokens=tokens, scores=saliency)
        
        display(HTML(f"<h4>Layer {layer_idx}</h4>"))
        #display(HTML(html))
        display(html)

        layer_data = {
            'tokens': tokens,
            'saliency': saliency,
            'codes': codes,
            'hidden': hidden,
            'dict': sae.decoder.weight.detach()
        }

        all_layer_outputs[f"layer_{layer_idx}"] = layer_data

        if do_log:
            # Save each layer separately
            os.makedirs(log_path, exist_ok=True)
            torch.save(hidden, f"{log_path}/{log_name}_ha_ml.pt")
            torch.save(codes, f"{log_path}/{log_name}_cc_ml.pt")
            torch.save(sae.decoder.weight.detach(), f"{log_path}/{log_name}_sae_dict_ml.pt")
            
            with open(f"{log_path}/{log_name}_tokens_ml.json", 'w') as f:
                json.dump(tokens, f)
        
        if vis_projection is not None:
            visualize_concepts(codes=codes, full_path=fig_path, method=vis_projection)

    return all_layer_outputs


def plot_colored_tokens(
        model:Any,
        tokenizer:Any,
        inputs:Any,
        multi_tokens:bool=False,
        do_log:bool=False,
        target_layers:List[int]=[5,10,15],
        vis_projection:str|None=None, # 'pca' | 'tsne' | None
        log_path:str|None=None,
        log_name:str|None=None,
        fig_path:str|None=None
) -> None:
    """ Plots colored tokens from SAE analysis """

    _run_multi_layer_sae(
        model=model,
        tokenizer=tokenizer,
        text=inputs,
        multi_tokens=multi_tokens,
        do_log=do_log,
        target_layers=target_layers,
        vis_projection=vis_projection,
        log_path=log_path,
        log_name=log_name,
        fig_path=fig_path
    )