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



""" ******************** ACTIVATION ******************** """
def get_layer_activations(model:Any, tokenizer:Any, text:str, target_layer_idx:int|List[int]) -> Tuple[Any,Any]:
    """ Hook Helper: Tokenize inputs and make activation layer hooks """

    model.eval()
    
    try:
        input_ids = tokenizer(text=text, return_tensors='pt').to(model.device)['input_ids']
    except Exception as e:
        print(f"[Tokenizer error] {e}")
        return None

    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach().cpu())

    try:
        handle = model.model.layers[target_layer_idx].register_forward_hook(hook_fn)
    except Exception as e:
        print(f"[Hook registration error] {e}")
        return None

    try:
        with torch.no_grad():
            _ = model(input_ids)
    except Exception as e:
        print(f"[Forward pass error] {e}")
        return None
    finally:
        handle.remove()

    if not activations:
        print(f"[Hook error] No activations captured for layer {target_layer_idx}")
        return None

    return input_ids[0], activations[0].squeeze(0)



""" ******************** PLOTLY INTERACTIVE COLORED TOKENS ******************** """
def plot_tokens_plotly(tokens, feature_token_matrix, top_k=10):
    fig = go.Figure()
    for i in range(min(top_k, feature_token_matrix.shape[0])):
        fig.add_trace(go.Heatmap(
            z=[feature_token_matrix[i].numpy()],
            x=tokens,
            y=[f"Feature {i}"],
            colorscale='RdBu',
            zmin=0, zmax=feature_token_matrix.max().item()
        ))
    fig.update_layout(height=50 * top_k)
    fig.show()


""" ******************** PLOT COLORED TOKENS ******************** """
def plot_colored_tokens(tokens:Any, scores:Any, cmap:str='coolwarm'):
    """ Plots the colored tokens derived from SAE """

    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)
    colors = plt.cm.get_cmap(cmap)(norm_scores.numpy())
    html = ""

    for tok, col in zip(tokens, colors):
        rgba = f"rgba({int(col[0]*255)}, {int(col[1]*255)}, {int(col[2]*255)}, 0.9)"
        html += f"<span style='background-color:{rgba}; padding:2px; margin:1px; border-radius:4px'>{tok}</span> "
    
    return html


def colored_tokens_multi_custom(tokens, feature_token_matrix, top_k=10, cmap='coolwarm'):
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


""" ******************** SAE ANALYSIS ******************** """

"""
Saliency socre:
    take the absolute value of this gradient, then take the maximum value over the 3 input channels;
    the final saliency map thus has shape (H, W) and all entries are nonnegative

Interpretation:
    a proxy for "concept certainty" or "selectivity" — if a token activates just one concept really strongly,
    that might suggest a highly interpretable or distinct representation.
    Low saliency might mean it's spread out or ambiguous.

Activation norms:
    In FP16 models, activations often have a nice spread.
    In quantized models (e.g. 8-bit or 4-bit), activation ranges are expected to shrink due to reduced dynamic precision.
    So if one observe significantly smaller norms post-quantization → it could mean:
    Loss of representational capacity.
    Suppressed activations → harder for downstream neurons to detect patterns.

Interpretation:
    Lower average norms might correlate with reduced semantic richness or degraded internal feature geometry.
"""

def run_sae(model:Any, tokenizer:Any, full_path:str, file_name:str, text:str, layer_idx:int) -> Tuple[Any,torch.Tensor]:
    """ Run single layer SAE analysis """

    token_ids, hidden = get_layer_activations(model, tokenizer, text, target_layer_idx=layer_idx)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    sae = SAE(input_dim=hidden.shape[1], dict_size=512, sparsity_lambda=1e-3)
    sae.train_sae(hidden, epochs=10, batch_size=4)

    code = sae.encode(hidden).detach()
    normed = F.normalize(code, dim=1)
    saliency = normed.abs().max(dim=1).values  # strong concept activation

    html = plot_colored_tokens(tokens, saliency)
    display(HTML(html))

    # Save results
    os.makedirs(full_path, exist_ok=True)
    torch.save(hidden, f"{full_path}/{file_name}_ha_sl.pt")
    torch.save(code, f"{full_path}/{file_name}_cc_sl.pt")
    torch.save(sae.decoder.weight.detach(), f"{full_path}/{file_name}_sae_dict_sl.pt")

    with open(f"{full_path}/{file_name}_tokens_sl.json", 'w') as f:
        json.dump(tokens, f)

    return tokens, saliency


def run_multi_layer_sae(model:Any, tokenizer:Any, full_path:str, file_name:str, text:str, target_layers:List[int]) -> Dict:
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

        html = plot_colored_tokens(tokens, saliency)
        display(HTML(f"<h4>Layer {layer_idx}</h4>"))
        display(HTML(html))

        layer_data = {
            'tokens': tokens,
            'saliency': saliency,
            'codes': codes,
            'hidden': hidden,
            'dict': sae.decoder.weight.detach()
        }

        all_layer_outputs[f"layer_{layer_idx}"] = layer_data

        # Save each layer separately
        os.makedirs(full_path, exist_ok=True)
        torch.save(hidden, f"{full_path}/{file_name}_ha_ml.pt")
        torch.save(codes, f"{full_path}/{file_name}_cc_ml.pt")
        torch.save(sae.decoder.weight.detach(), f"{full_path}/{file_name}_sae_dict_ml.pt")
        
        with open(f"{full_path}/{file_name}_tokens_ml.json", 'w') as f:
            json.dump(tokens, f)

    return all_layer_outputs



""" ******************** FULL DICT LEARNING INTERP ******************** """

def plot_summary_metrics(summary:Any, layer_idx:Any, save_path:str=DirPath.DICT_VIS.value) -> None:
    saliency = summary['saliency']
    entropy = summary['entropy']
    usage = summary['usage']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0,0].plot(saliency.numpy())
    axs[0,0].set_title("Saliency (per token)")

    axs[0,1].plot(entropy.numpy())
    axs[0,1].set_title("Concept Entropy (per token)")

    axs[1,0].bar(range(len(usage)), usage.numpy())
    axs[1,0].set_title("Concept Usage Frequency")

    axs[1,1].set_title("Usage Histogram")
    axs[1,1].hist(usage.numpy(), bins=20)

    fig.suptitle(f"SAE Metrics - Layer {layer_idx}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_path}/summary_layer_{layer_idx}.jpg")
    plt.close()


def analyze_sae_layer(model:Any, tokenizer:Any, sae:Any, text:Any, layer_idx:Any, save_path:str, topk:int=5) -> Dict:
    """ Full analysis pipeline for a given layer """

    os.makedirs(save_path, exist_ok=True)

    # === Get hidden states
    token_ids, hidden = get_layer_activations(model, tokenizer, text, target_layer_idx=layer_idx)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # === SAE codes and recon
    with torch.no_grad():
        recon, codes = sae(hidden)
    
    normed_codes = F.normalize(codes, dim=1)
    saliency = normed_codes.abs().max(dim=1).values
    entropy = -(F.softmax(codes.abs(), dim=1) * F.log_softmax(codes.abs(), dim=1)).sum(dim=1)
    usage = (codes.abs() > 0.01).float().sum(dim=0)
    recon_error = F.mse_loss(recon, hidden).item()
    
    # === Dictionary orthogonality
    diversity = dict_orthogonality(sae.decoder.weight.detach())

    # === Save all to dict
    summary = {
        "tokens": tokens,
        "saliency": saliency,
        "entropy": entropy,
        "usage": usage,
        "recon_error": recon_error,
        "diversity": diversity,
    }

    torch.save(summary, f"{save_path}/layer_{layer_idx}_summary.pt")
    # === Plot visual summary
    plot_summary_metrics(summary, layer_idx, save_path)

    return summary


def explain_concept(codes:Any, tokens:Any, concept_idx:Any, top_k:int=5) -> List[Any]:
    """ Show top tokens that activate a concept most """

    concept_vals = codes[:, concept_idx]
    top_indices = torch.topk(concept_vals, k=top_k).indices
    
    return [tokens[i] for i in top_indices]


def run_single_layer(codes:Any, tokens:Any, topk:int=5) -> Tuple[Any|int,Any|List[Any]]:
    for i in range(topk):  # First concept neurons
        top_toks = explain_concept(codes, tokens, concept_idx=i)
        print(f"Concept {i:02d}: {top_toks}")
    
    return i, top_toks


def run_all_layers(model:Any, tokenizer:Any, sae_dict:Dict, text:Any, layers:Any, save_root:str) -> Dict:
    all_results = {}
    for layer in layers:
        print(f"Running analysis for layer {layer}")
        sae = sae_dict[layer]
        layer_save = os.path.join(save_root, f"layer_{layer}")
        res = analyze_sae_layer(model, tokenizer, sae, text, layer_idx=layer, save_path=layer_save)
        all_results[layer] = res
    
    return all_results
