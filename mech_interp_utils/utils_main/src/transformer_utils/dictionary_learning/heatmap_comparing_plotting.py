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

def _plot_comparing_heatmap(
    tokens:List[str],
    feature_token_matrix:torch.Tensor,
    top_k:int=5,
    saliency:torch.Tensor|None=None,
    tokens_per_row:int=12,
    fp_saliency:torch.Tensor|None=None,
    quant_saliency:torch.Tensor|None=None
) ->  None:
    num_tokens = len(tokens)
    token_feature_matrix = feature_token_matrix.T

    num_rows = math.ceil(num_tokens / tokens_per_row)
    z = np.zeros((num_rows, tokens_per_row))
    text = [["" for _ in range(tokens_per_row)] for _ in range(num_rows)]
    hovertext = [["" for _ in range(tokens_per_row)] for _ in range(num_rows)]

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

        fp_val = f"{fp_saliency[idx].item():.3f}" if fp_saliency is not None else "N/A"
        q_val = f"{quant_saliency[idx].item():.3f}" if quant_saliency is not None else "N/A"
        delta_val = f"{abs(fp_saliency[idx] - quant_saliency[idx]).item():.3f}" if (fp_saliency is not None and quant_saliency is not None) else "N/A"

        # Simulated bar with Unicode blocks
        def bar(val, max_len=10):
            blocks = "▁▂▃▄▅▆▇█"
            idx = int(val * (len(blocks) - 1))
            return blocks[idx] * max_len

        feature_lines = ""
        #for f_idx, f_val in zip(top_idx, top_vals):
            #norm_val = (f_val.item() - top_vals.min().item()) / (top_vals.max().item() - top_vals.min().item() + 1e-5)
            #feature_lines += f"{bar(norm_val, max_len=5)} Feature {f_idx.item()}: {f_val.item():.3f}\n"
        for rank, (f_idx, f_val) in enumerate(zip(top_idx, top_vals), 1):
            norm_val = (f_val.item() - top_vals.min().item()) / (top_vals.max().item() - top_vals.min().item() + 1e-5)
            bg_color = f"rgba({255*(1-norm_val):.0f},{255*(norm_val):.0f},150,0.6)"
            top_token = clean_token(tokens[f_idx.item()]) if f_idx.item() < len(tokens) else f"F{f_idx.item()}"
            feature_lines += f"<span style='background-color:{bg_color};padding:2px;'>Top-{rank}: {top_token} ({f_val.item():.2f})</span><br>"

        hovertext[row][col] = (
            f"Token: {clean_tok}\n"
            f"FP Saliency: {fp_val}\n"
            f"Quant Saliency: {q_val}\n"
            f"Δ Saliency: {delta_val}\n"
            f"\nTop-{top_k} Features:\n{feature_lines}"
        )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        hoverinfo="text",
        hovertext=hovertext,
        colorscale='RdBu',
        reversescale=True,
        showscale=True,
        texttemplate="%{text}",
        #textfont={"size": 11, "color": "black", "family": "Times New Roman"},
        textfont={"size": 10, "family": "Times New Roman"},
        xgap=2, ygap=2
    ))

    fig.update_layout(
        title="Token-Level FP vs Quant Comparison",
        title_font=dict(size=12, family="Times New Roman"),
        width=max(600, tokens_per_row * 50),
        height=num_rows * 50 + 100,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange='reversed'),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.show()


def _run_multi_model_sae(
        models:Tuple[Any,Any],
        tokenizer:Any,
        text:str,
        top_k:int=5,
        tokens_per_row:int=12,
        target_layers:List[int]=[5,10,15],
        models_to_eval:bool=True,
        deterministic_sae:bool=True,
        fig_path:str|None=None    
) -> Dict:
    """
    Run multi-layer SAE analysis
    eval() to disable dropout and uses running statistics for batch norm instead of batch-wise statistics.
    This ensures the model behaves consistently during evaluation or inference (e.g., stable activations for SAE), not training.
    """

    fp_model, quant_model = models
    if models_to_eval:
        fp_model.eval(), quant_model.eval()

    all_outputs = {}

    for layer_idx in target_layers:
        print(f"\nComparing SAE on layer {layer_idx}")

        # Run FP model
        fp_ids, fp_hidden = get_layer_activations(fp_model, tokenizer, text, target_layer_idx=layer_idx)
        tokens = tokenizer.convert_ids_to_tokens(fp_ids)

        sae_fp = SAE(input_dim=fp_hidden.shape[1], dict_size=512, sparsity_lambda=1e-3, deterministic_sae=deterministic_sae)
        sae_fp.train_sae(fp_hidden, epochs=10, batch_size=4)
        fp_codes = sae_fp.encode(fp_hidden).detach()
        fp_saliency = F.normalize(fp_codes, dim=1).abs().max(dim=1).values

        # Run Quant model
        _, quant_hidden = get_layer_activations(quant_model, tokenizer, text, target_layer_idx=layer_idx)
        sae_q = SAE(input_dim=quant_hidden.shape[1], dict_size=512, sparsity_lambda=1e-3, deterministic_sae=deterministic_sae)
        sae_q.train_sae(quant_hidden, epochs=10, batch_size=4)
        quant_codes = sae_q.encode(quant_hidden).detach()
        quant_saliency = F.normalize(quant_codes, dim=1).abs().max(dim=1).values

        # Difference metric (can switch this to something else)
        diff_saliency = (fp_saliency - quant_saliency).abs()

        # Use FP model's top features for heatmap anchors
        total_per_feature = fp_codes.abs().sum(dim=0)
        topk_indices = total_per_feature.topk(top_k).indices
        fp_top_matrix = fp_codes[:, topk_indices].abs().T  # [top_k x num_tokens]

        _plot_comparing_heatmap(
            tokens=tokens,
            feature_token_matrix=fp_top_matrix,
            top_k=top_k,
            saliency=diff_saliency,  # difference instead of raw saliency
            tokens_per_row=tokens_per_row,
            fp_saliency=fp_saliency,
            quant_saliency=quant_saliency
        )

        all_outputs[f"layer_{layer_idx}"] = {
            'tokens': tokens,
            'fp_saliency': fp_saliency,
            'quant_saliency': quant_saliency,
            'diff_saliency': diff_saliency,
            'fp_codes': fp_codes,
            'quant_codes': quant_codes
        }

    return all_outputs


def plot_comparing_heatmap(
        models:Tuple[Any,Any],
        tokenizer:Any,
        inputs:Any,
        top_k:int=5,
        tokens_per_row:int=12,
        target_layers:List[int]=[5,10,15],
        models_to_eval:bool=True,
        deterministic_sae:bool=True,
        fig_path:str|None=None
) -> None:
    """ Plots colored tokens from SAE analysis """

    _run_multi_model_sae(
        models=models,
        tokenizer=tokenizer,
        text=inputs,
        top_k=top_k,
        tokens_per_row=tokens_per_row,
        target_layers=target_layers,
        models_to_eval=models_to_eval,
        deterministic_sae=deterministic_sae,
        fig_path=fig_path
    )