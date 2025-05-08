from functools import partial
from typing import Tuple, List, Dict, Any
import torch
import numpy as np

import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa
import plotly.graph_objects as go
from ..util.python_utils import make_print_if_verbose

from .hooks import make_lens_hooks
from .layer_names import make_layer_names


# ===================== Probs and logits for topk = 1 ============================
def text_to_input_ids(tokenizer:Any, text:str) -> torch.Tensor:
    """ Encode inputs """

    toks = tokenizer.encode(text, return_tensors="pt")
    return torch.as_tensor(toks).view(1, -1).cpu()

def collect_logits(model, input_ids, layer_names, decoder_layer_names):

    model._last_resid = None

    with torch.no_grad():
        out = model(input_ids)
    del out
    model._last_resid = None

    layer_logits = np.concatenate(
        [model._layer_logits[name] for name in layer_names],
        axis=0,
    )

    return layer_logits, layer_names

# ===================== Probs and logits for topk > 1 and for topk plot ============================
def postprocess_logits_tokp(layer_logits:Any, normalize_probs=False, top_n:int=5, return_scores:bool=True) -> Tuple[Any,Any,Any]:
        
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs

    layer_preds = layer_logits.argmax(axis=-1)

    # op-N token values (per layer, per token position): gives an array of shape [layers, tokens] with mean of top-N probs
    top_n_scores = np.mean(
        np.sort(layer_probs, axis=-1)[:, :, -top_n:], axis=-1
    )

    if return_scores:
        return layer_preds, layer_probs, top_n_scores
    else:
        return layer_preds, layer_probs

# ===================== Clipping and metric helpers ============================
def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)

def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark

def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)

def compute_entropy(probs):
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_probs, axis=-1)

def maybe_batchify(p):
    """ Normalize shape for e.g., wasserstein """
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    return p

def min_max_scale(arr, new_min, new_max):
    """Scales an array to a new range [new_min, new_max] for plotting and comparison of normalized values across models."""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr

def _plot_logit_lens_plotly(
    layer_logits,
    layer_preds,
    layer_probs,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    top_k=5,
    normalize=True,
    metric_type:str|None=None,
    map_color='Cividis',
    value_matrix=None,
    title:str|None=None
):
    num_layers, num_tokens, vocab_size = layer_logits.shape
    end_ix = start_ix + num_tokens

    input_token_ids = input_ids[0][start_ix:end_ix]
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_token_ids]

    # === Fixed top-axis label logic ===
    full_token_ids = input_ids[0]
    next_token_ids = full_token_ids[start_ix + 1:end_ix + 1]
    next_token_text = [tokenizer.decode([tid]) for tid in next_token_ids]
    if len(next_token_text) < num_tokens:
        next_token_text.append("")  # pad with empty label if no next token

    # === Predicted token text for each layer/token ===
    pred_token_text = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)

    # === Top-k tokens and scores ===
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)
    topk_indices = np.argsort(layer_probs, axis=-1)[:, :, -top_k:][:, :, ::-1]
    topk_scores = np.take_along_axis(layer_probs, topk_indices, axis=-1)
    topk_tokens = np.vectorize(lambda idx: tokenizer.decode([idx]), otypes=[str])(topk_indices)

    hovertext = []
    for i in range(len(layer_names)):
        row = []
        for j in range(num_tokens):
            pred_tok = pred_token_text[i, j]
            #hover = f"<b>Pred:</b> {pred_tok}<br><b>Top-{top_k}:</b><br>"
            val = value_matrix[::-1][i, j]  # because you already flipped the matrix
            if metric_type == "entropy":
                hover_val = f"<b>Entropy:</b> {val:.3f}<br>"
            elif metric_type == "logits":
                hover_val = f"<b>Logit:</b> {val:.3f}<br>"
            else:  # default to prob
                hover_val = f"<b>Prob:</b> {val:.3f}<br>"

            hover = hover_val + f"<b>Pred:</b> {pred_tok}<br><b>Top-{top_k}:</b><br>"

            for k in range(top_k):
                hover += f"&nbsp;&nbsp;{topk_tokens[i, j, k]}: {topk_scores[i, j, k]:.3f}<br>"
            row.append(hover)
        hovertext.append(row)

    # === Apply normalization for better heatmap scaling ===
    if normalize:
        if metric_type == 'entropy':
            value_matrix = min_max_scale(value_matrix, 0, 10)
        else:
            value_matrix = min_max_scale(value_matrix, 0, 1)

    # === Flip Y for top-down layer order ===
    value_matrix = value_matrix[::-1]
    pred_token_text = pred_token_text[::-1]
    hovertext = hovertext[::-1]
    zmin = np.min(value_matrix)
    zmax = np.max(value_matrix)
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=value_matrix,
        x=list(range(num_tokens)),
        y=list(range(len(layer_names))),
        xaxis='x',
        yaxis='y',
        text=pred_token_text,
        texttemplate="%{text}",
        hovertext=hovertext,
        hoverinfo='text',
        colorscale=map_color,
        #zmin = np.percentile(value_matrix, 1),
        #zmax = np.percentile(value_matrix, 99),
        zmin = zmin,
        zmax = zmax,
        colorbar=dict(title=metric_type),
    ))

    fig.update_layout(
        title=title,
        font=dict(family="Times New Roman", size=14),

        xaxis=dict(
            tickmode='array',
            tickvals=list(range(num_tokens)),
            ticktext=input_tokens_str,
            title='Input Token',
            side='bottom',
            anchor='y',
            domain=[0.0, 1.0]
        ),

        xaxis2=dict(
            tickmode='array',
            tickvals=list(range(num_tokens)),
            ticktext=next_token_text,
            overlaying='x',
            side='top',
            anchor='free',
            position=1.0
        ),

        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(layer_names))),
            ticktext=layer_names[::-1],
            title='Layer',
            autorange='reversed',
        ),

        width=min(1400, 90 * num_tokens + 200),
        height=400 + 25 * len(layer_names),
        margin=dict(l=80, r=80, t=100, b=80),
    )

    fig.show()

def plot_logit_lens_plotly(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    start_ix: int,
    end_ix: int,
    topk: int = 5,
    save_fig_path: str | None = None,
    probs: bool = False,
    entropy: bool = False,
    block_step: int = 1,
    include_input: bool = True,
    force_include_output: bool = True,
    include_subblocks: bool = False,
    decoder_layer_names: list = ['norm', 'lm_head'],
    top_down: bool = False,
    verbose: bool = False
):
    metric_type: str | None = None

    layer_names = make_layer_names(
        model,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names
    )

    make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names,
                    decoder_layer_names=decoder_layer_names, verbose=verbose)

    input_ids = text_to_input_ids(tokenizer, input_ids)

    layer_logits, layer_names = collect_logits(
        model, input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names
    )

    layer_preds, layer_probs, _ = postprocess_logits_tokp(layer_logits, False, top_n=topk)

    # === Metric selection logic ===
    if entropy:
        map_color = 'RdBu_r'
        value_matrix = compute_entropy(layer_probs)
        metric_type = 'entropy'
        title:str = f"Entropy (topk-{topk} predicted tokens)"
    elif probs:
        map_color = 'Blues'
        value_matrix = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1).squeeze(-1)
        metric_type = 'probs'
        title:str = f"Probabilities (topk-{topk} predicted tokens)"
    else:
        map_color = 'Cividis'
        value_matrix = np.take_along_axis(layer_logits, layer_preds[..., None], axis=-1).squeeze(-1)
        metric_type = 'logits'
        title:str = f"Logits (topk-{topk} predicted tokens)"

    _plot_logit_lens_plotly(
        layer_logits=layer_logits,
        layer_preds=layer_preds,
        layer_probs=layer_probs,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        layer_names=layer_names,
        top_k=topk,
        normalize=True,
        metric_type=metric_type,
        map_color=map_color,
        value_matrix=value_matrix,
        title=title
    )