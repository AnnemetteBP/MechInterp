from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os, json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from IPython.display import display, HTML

from .sae_class import SAE
from .sae_hooks import get_layer_activations


def clean_token(token: str) -> str:
    if token.startswith("Ġ"):
        return " " + token[1:]
    elif token.startswith("Ċ"):
        return "\\n" + token[1:]
    else:
        return token

def _plot_token_heatmap(
    tokens:List[str],
    feature_token_matrix:torch.Tensor,
    top_k:int=5,
    saliency:torch.Tensor|None=None,
    tokens_per_row:int=12,
) -> None:
    num_tokens = len(tokens)
    token_feature_matrix = feature_token_matrix.T  # [num_tokens x top_k_features]

    num_rows = math.ceil(num_tokens / tokens_per_row)
    z = np.zeros((num_rows, tokens_per_row))
    text = [["" for _ in range(tokens_per_row)] for _ in range(num_rows)]
    hovertext = [["" for _ in range(tokens_per_row)] for _ in range(num_rows)]

    # Normalize saliency
    if saliency is not None:
        norm_saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-5)
        norm_saliency = norm_saliency.numpy()

    for idx, token in enumerate(tokens):
        row = idx // tokens_per_row
        col = idx % tokens_per_row

        clean_tok = clean_token(token)
        token_scores = token_feature_matrix[idx]
        top_vals, top_idx = token_scores.topk(top_k)

        z[row][col] = norm_saliency[idx] if saliency is not None else top_vals[0].item()
        text[row][col] = clean_tok

        hover = f"Token: {clean_tok}\nTop-{top_k} Features:\n"
        for f_idx, f_val in zip(top_idx, top_vals):
            val = f_val.item()
            norm_val = (val - top_vals.min().item()) / (top_vals.max().item() - top_vals.min().item() + 1e-5)
            saliency_marker = "●" * int(norm_val * 5)  # Simulated saliency
            hover += f"  {saliency_marker} Feature {f_idx.item()}: {val:.3f}\n"

        hovertext[row][col] = hover

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        hoverinfo="text",
        hovertext=hovertext,
        colorscale='BuPu_r',  # Similar to seaborn 'coolwarm'  (try 'RdBu', 'RdBu_r', 'Plotly3')
        reversescale=True,
        showscale=True,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black", "family": "Times New Roman"},
        xgap=2, ygap=2
    ))

    fig.update_layout(
        title=f"Top-k {top_k} Token Activation",
        title_font=dict(size=12, family="Times New Roman"),
        width=max(600, tokens_per_row * 50),
        height=num_rows * 50 + 100,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showticklabels=False),
        #yaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange='reversed'),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.show()


def _run_multi_layer_sae(
        model:Any,
        tokenizer:Any,
        text:str,
        do_log:bool=False,
        top_k:int=5,
        tokens_per_row:int=12,
        target_layers:List[int]=[5,10,15],
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

        layer_data = {
            'tokens': tokens,
            'saliency': saliency,
            'codes': codes,
            'hidden': hidden,
            'dict': sae.decoder.weight.detach()
        }

        all_layer_outputs[f"layer_{layer_idx}"] = layer_data
        #feature_token_matrix = codes.abs().T
        # Rank neurons by total activation
        total_per_feature = codes.abs().sum(dim=0)  # shape: [dict_size]
        topk_indices = total_per_feature.topk(top_k).indices
        feature_token_matrix = codes[:, topk_indices].abs().T  # shape: [top_k x num_tokens]

        _plot_token_heatmap(
            tokens=tokens,
            feature_token_matrix=feature_token_matrix,
            top_k=top_k,
            saliency=saliency,
            tokens_per_row=tokens_per_row
        )

        if do_log:
            # Save each layer separately
            os.makedirs(log_path, exist_ok=True)
            torch.save(hidden, f"{log_path}/{log_name}_ha_ml.pt")
            torch.save(codes, f"{log_path}/{log_name}_cc_ml.pt")
            torch.save(sae.decoder.weight.detach(), f"{log_path}/{log_name}_sae_dict_ml.pt")
            
            with open(f"{log_path}/{log_name}_tokens_ml.json", 'w') as f:
                json.dump(tokens, f)

    return all_layer_outputs


def plot_sae_heatmap(
        model:Any,
        tokenizer:Any,
        inputs:Any,
        do_log:bool=False,
        top_k:int=5,
        tokens_per_row:int=12,
        target_layers:List[int]=[5,10,15],
        log_path:str|None=None,
        log_name:str|None=None,
        fig_path:str|None=None
) -> None:
    """ Plots colored tokens from SAE analysis """

    _run_multi_layer_sae(
        model=model,
        tokenizer=tokenizer,
        text=inputs,
        do_log=do_log,
        top_k=top_k,
        tokens_per_row=tokens_per_row,
        target_layers=target_layers,
        log_path=log_path,
        log_name=log_name,
        fig_path=fig_path
    )