from functools import partial
from typing import Tuple, List, Dict, Any, Union, Optional
import json
from pathlib import Path
import torch
import numpy as np

import scipy.special
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa
import plotly.graph_objects as go
from ..util.python_utils import make_print_if_verbose

from .hooks import make_lens_hooks
from .layer_names import make_layer_names


# ===================== Clear Cache ============================
def clear_cuda_cache():
    """Clear GPU cache to avoid memory errors during operations"""
    torch.cuda.empty_cache()

def safe_cast_logits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8]:
        tensor = tensor.to(torch.float32)
    return torch.nan_to_num(tensor, nan=-1e9, posinf=1e9, neginf=-1e9)

def numpy_safe_cast(x):
    x = x.astype(np.float32)
    return np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)

# ===================== Tokenize input texts ============================
def text_to_input_ids(tokenizer: Any, text: Union[str, List[str]], model: Optional[torch.nn.Module] = None, add_special_tokens: bool = True, pad_to_max_length=False) -> torch.Tensor:
    """
    Tokenize the inputs, respecting padding behavior.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure EOS token is used if padding is missing

    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    # Padding to the longest sequence in the batch or to max length
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest" if not pad_to_max_length else True,  # Padding only to longest sequence
        truncation=True,
        add_special_tokens=add_special_tokens,
    )["input_ids"]

    if model is not None:
        device = next(model.parameters()).device
        tokens = tokens.to(device)

    return tokens  # shape: [batch_size, seq_len]

# ===================== Layer Logits ============================
def collect_logits(model, input_ids, layer_names, decoder_layer_names):
    model._last_resid = None
    
    # Handle single vs batch input
    if input_ids.ndim == 2:  # Single prompt
        batch_size = 1
    else:  # Multiple prompts (batch)
        batch_size = input_ids.shape[0]
    
    with torch.no_grad():
        out = model(input_ids)
    del out
    model._last_resid = None
    
    # Ensure we gather logits for each layer (whether for single or batch)
    try:
        layer_logits = np.concatenate(
            [model._layer_logits.get(name, np.zeros((batch_size, model.config.hidden_size))) for name in layer_names],
            axis=0,
        )
    except KeyError as e:
        print(f"[Error] Missing layer logits for {e}")
        layer_logits = np.zeros((batch_size, len(layer_names), model.config.hidden_size))

    return layer_logits, layer_names


# ===================== Probs and logits for topk > 1 and for topk plot ============================
def postprocess_logits_topk(layer_logits: Any, normalize_probs=False, top_n:int=5, return_scores:bool=True) -> Tuple[Any, Any, Any]:

    if layer_logits.dtype == np.float16:
        layer_logits = layer_logits.astype(np.float32)
        
    # Replace NaNs and infs with appropriate values
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    # Normalize the probabilities if needed
    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs

    #print(f"[DEBUG PROBS] {layer_probs} | ")

    # Get the index of the maximum logit (predicted token)
    layer_preds = layer_logits.argmax(axis=-1)

    # Compute the mean of the top-N probabilities for each token
    top_n_scores = np.mean(
        np.sort(layer_probs, axis=-1)[:, -top_n:], axis=-1
    )

    if return_scores:
        return layer_preds, layer_probs, top_n_scores
    else:
        return layer_preds, layer_probs

# ===================== Correctness ============================
def compute_correctness_metrics(layer_preds, target_ids, topk_indices=None):
    correct_1 = (layer_preds[-1] == target_ids).astype(int)

    if topk_indices is not None:
        # target_ids: [tokens], reshape to [1, tokens, 1] to broadcast against [layers, tokens, topk]
        target_ids_broadcasted = target_ids[None, :, None]
        correct_topk = np.any(topk_indices == target_ids_broadcasted, axis=-1).astype(int)
    else:
        correct_topk = None

    return correct_1, correct_topk

# ===================== Clipping and metric helpers ============================
def compute_kl_divergence(logits1, logits2):
    probs1 = scipy.special.softmax(logits1, axis=-1)
    probs2 = scipy.special.softmax(logits2, axis=-1)
    return np.sum(kl_div(probs1, probs2), axis=-1)

def compute_wasserstein_from_json(file_a, file_b, key="logit_mean"):
    with open(file_a) as f1, open(file_b) as f2:
        m1 = json.load(f1)[key]
        m2 = json.load(f2)[key]
    return wasserstein_distance(m1, m2)


def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)

def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark

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

def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)

def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)

def kl_div(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)

def js_divergence(p, q, axis=-1, clip=1e-16):
    """Computes Jensen-Shannon divergence between two probability distributions."""
    p, q = clipmin(p, clip), clipmin(q, clip)
    m = (p + q) / 2
    return (kl_div(p, m, axis=axis, clip=clip) + kl_div(q, m, axis=axis, clip=clip)) / 2

def nwd(p, q, axis=-1, clip=1e-6):  # <-- Increase clip
#def nwd(p, q, axis=-1, clip=1e-16):
    """Computes normalized Wasserstein distance between two probability distributions."""
    p, q = np.clip(p, clip, 1.0), np.clip(q, clip, 1.0)

    # Ensure both are 3D: (batch, seq_len, vocab)
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    if q.ndim == 2:
        q = np.expand_dims(q, 0)

    # Align batch sizes
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.repeat(p, q.shape[0], axis=0)
    elif q.shape[0] == 1 and p.shape[0] > 1:
        q = np.repeat(q, p.shape[0], axis=0)

    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch after fixing: p {p.shape}, q {q.shape}")

    vocab_size = p.shape[-1]
    indices = np.arange(vocab_size)

    distances = np.array([
        wasserstein_distance(indices, indices, p[b, i], q[b, i])
        for b in range(p.shape[0])
        for i in range(p.shape[1])
    ])

    # Reshape back to (batch, seq_len)
    distances = distances.reshape(p.shape[0], p.shape[1])
    #print(f"[Distances] {distances} | \n[NWD] {distances/vocab_size} |\n")
    return distances / vocab_size


def _plot_logit_lens_plotly(
    layer_logits,
    layer_preds,
    layer_probs,
    topk_scores,
    topk_indices,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    top_k=5,
    normalize=True,
    metric_type:str|None=None,
    map_color='Cividis',
    value_matrix=None,
    title:str|None=None,
    block_step:int=1,
    token_font_size:int=12
):
    
    num_layers, num_tokens, vocab_size = layer_logits.shape
    end_ix = start_ix + num_tokens

    # Decode input and next tokens
    input_token_ids = input_ids[0][start_ix:end_ix]
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_token_ids]

    full_token_ids = input_ids[0]
    next_token_ids = full_token_ids[start_ix + 1:end_ix + 1]
    next_token_text = [tokenizer.decode([tid]) for tid in next_token_ids]
    if len(next_token_text) < num_tokens:
        next_token_text.append("")

    
    # Decode predictions and top-k tokens
    pred_token_text = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)
    topk_tokens = np.vectorize(lambda idx: tokenizer.decode([idx]), otypes=[str])(topk_indices)

    # Default: use mean top-k probability if no divergence matrix provided
    if value_matrix is None:
        value_matrix = topk_scores.mean(axis=-1)

    # Select layers based on block_step
    keep_idxs = [0] + list(range(1, num_layers - 1, block_step)) + [num_layers - 1]
    value_matrix = value_matrix[keep_idxs]
    pred_token_text = pred_token_text[keep_idxs]
    topk_scores = topk_scores[keep_idxs]
    topk_tokens = topk_tokens[keep_idxs]
    layer_names = [layer_names[i] for i in keep_idxs]

    # Prepare hover text
    hovertext = []
    for i in range(len(layer_names)):
        row = []
        for j in range(num_tokens):
            val = value_matrix[i, j]
            if metric_type == "js":
                hover_val = f"<b>JS Divergence:</b> {val:.3f}<br>"
            elif metric_type == "nwd":
                hover_val = f"<b>NWD:</b> {val:.3f}<br>"
            else:
                hover_val = f"<b>Mean Top-{top_k} Prob:</b> {val:.3f}<br>"
            hover = hover_val + f"<b>Pred:</b> {pred_token_text[i, j]}<br><b>Top-{top_k}:</b><br>"
            for k in range(top_k):
                hover += f"&nbsp;&nbsp;{topk_tokens[i, j, k]}: {topk_scores[i, j, k]:.3f}<br>"
            row.append(hover)
        hovertext.append(row)

    # Normalize values for consistent color range
    if normalize:
        value_matrix = min_max_scale(value_matrix, 0, 1)
    
    # Select layers based on block_step — must come early before using keep_idxs
    keep_idxs = [0] + list(range(1, num_layers - 1, block_step)) + [num_layers - 1]

    # Decode predictions from selected layers
    pred_tokens_str = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds[keep_idxs])

    # Ensure next_token_text used for xaxis2 is what we compare against
    true_next_token_text = next_token_text[:num_tokens]  # true next tokens at t+1
    if len(true_next_token_text) < num_tokens:
        true_next_token_text += [""] * (num_tokens - len(true_next_token_text))  # pad if needed

    # Repeat next_token_text across the number of layers to match dimensions
    correct_tokens_matrix = np.tile(true_next_token_text, (pred_tokens_str.shape[0], 1))

    # Compare prediction tokens to correct next tokens
    is_correct = (pred_tokens_str == correct_tokens_matrix)

    # Flip for correct top-to-bottom layer visualization
    value_matrix = value_matrix[::-1]
    pred_token_text = pred_token_text[::-1]
    hovertext = hovertext[::-1]
    is_correct = is_correct[::-1]

    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        z=value_matrix,
        x=list(range(num_tokens)),
        y=list(range(len(layer_names))),
        text=pred_token_text,
        texttemplate="%{text}",
        textfont=dict(size=token_font_size),
        hovertext=hovertext,
        hoverinfo='text',
        colorscale=map_color,
        zmin=np.min(value_matrix),
        zmax=np.max(value_matrix),
    ))

    # Find the correct prediction positions
    correct_y, correct_x = np.where(is_correct[::-1, :] == 1)
    # Mark the predictions that match the tokens on xaxis2 (next_token_text)
    for y, x in zip(correct_y, correct_x):
        adjusted_y = len(layer_names) - 1 - y  # unflip layer index
        fig.add_shape(
            type="rect",
            x0=x - 0.5, x1=x + 0.5,
            y0=adjusted_y - 0.5, y1=adjusted_y + 0.5,
            line=dict(color="black", width=2),
            layer="above"
        )

    # Dummy trace to activate next-token axis
    fig.add_trace(go.Scatter(
        x=list(range(num_tokens)),
        y=[None] * num_tokens,
        xaxis='x2',
        mode='markers',
        marker=dict(opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        font=dict(family="DejaVu Sans", size=14),
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
            position=1.0,
            showline=True,
            ticks='outside'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(layer_names))),
            ticktext=layer_names[::-1],
            title='Layer',
            autorange='reversed',
        ),
        width=max(1200, 100 * num_tokens),
        height=max(600, 35 * len(layer_names)),
        margin=dict(l=20, r=10, t=40, b=10),
    )

    fig.show()


def plot_topk_comparing_lens(
    models:Tuple[Any, Any],
    #tokenizer:Any,
    tokenizers:Tuple[Any, Any],
    inputs:Union[str, List[str], None],
    start_ix:int,
    end_ix:int,
    topk:int=5,
    topk_mean:bool=True,
    js:bool=False,
    block_step:int=1,
    token_font_size:int=12,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:List=['norm', 'lm_head'],
    top_down:bool=False,
    verbose:bool=False,
    pad_to_max_length:bool=False
):
    
    metric_type = None

    # Handle input format and ensure it's a list of prompts
    if isinstance(inputs, str):
        inputs = [inputs]
    elif inputs is None:
        inputs = ["What is y if y=2*2-4+(3*2)"]  # Default prompt

    def multiple_layer_names(model_1:Any, model_2:Any) -> Tuple[List[str], List[str]]:
        layer_names_1 = make_layer_names(
            model_1,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            decoder_layer_names=decoder_layer_names
        )

        layer_names_2 = make_layer_names(
            model_2,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            decoder_layer_names=decoder_layer_names
        )

        return layer_names_1, layer_names_2
    

    model_1, model_2 = models # first is true distribution and second is e.g., quantized model for comparison
    tokenizer_1, tokenizer_2 = tokenizers # first is true distribution and second is e.g., quantized model for comparison

    layer_names_1, layer_names_2 = multiple_layer_names(model_1=model_1, model_2=model_2)

    make_lens_hooks(
        model_1, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_1, decoder_layer_names=decoder_layer_names, verbose=verbose
    )
    make_lens_hooks(
        model_2, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_2, decoder_layer_names=decoder_layer_names, verbose=verbose
    )

    # Tokenize inputs with padding control
    input_ids_1 = text_to_input_ids(tokenizer_1, inputs, model_1, pad_to_max_length=pad_to_max_length)
    input_ids_1 = input_ids_1.to(next(model_1.parameters()).device)
    input_ids_2 = text_to_input_ids(tokenizer_2, inputs, model_2, pad_to_max_length=pad_to_max_length)
    input_ids_2 = input_ids_1.to(next(model_2.parameters()).device)
    #input_ids_2 = input_ids
    # Extract ground truth next-token IDs (shifted input_ids)

    # Get logits
    layer_logits_1, layer_names_1 = collect_logits(model_1, input_ids_1, layer_names_1, decoder_layer_names)
    layer_logits_2, _ = collect_logits(model_2, input_ids_2, layer_names_2, decoder_layer_names)
    layer_logits_1 = safe_cast_logits(torch.tensor(layer_logits_1)).numpy()
    layer_logits_2 = safe_cast_logits(torch.tensor(layer_logits_2)).numpy()
    # Get predictions/probabilities
    layer_preds_1, layer_probs_1, _ = postprocess_logits_topk(layer_logits_1, top_n=topk)
    layer_preds_2, layer_probs_2, _ = postprocess_logits_topk(layer_logits_2, top_n=topk)

    # Clean up NaNs/Infs
    layer_probs_1 = np.nan_to_num(layer_probs_1, nan=1e-10, posinf=1.0, neginf=0.0)
    layer_probs_2 = np.nan_to_num(layer_probs_2, nan=1e-10, posinf=1.0, neginf=0.0)

    topk_indices_1 = np.argsort(layer_probs_1, axis=-1)[..., -topk:][..., ::-1]
    topk_scores_1 = np.take_along_axis(layer_probs_1, topk_indices_1, axis=-1)
    
    topk_indices_2 = np.argsort(layer_probs_2, axis=-1)[..., -topk:][..., ::-1]
    topk_scores_2 = np.take_along_axis(layer_probs_2, topk_indices_2, axis=-1)


    # Compute divergence matrix
    if js:
        metric_type = 'js'
        map_color = 'Blues'
        title = f"JS ({'mean topk' if topk_mean else 'full dist'})"

        if topk_mean:
            clipped_probs_1 = np.take_along_axis(layer_probs_1, topk_indices_2, axis=-1)
            clipped_probs_2 = np.take_along_axis(layer_probs_2, topk_indices_2, axis=-1)
            
            norm_probs_1 = clipped_probs_1 / np.clip(clipped_probs_1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
            norm_probs_2 = clipped_probs_2 / np.clip(clipped_probs_2.sum(axis=-1, keepdims=True), 1e-10, 1.0)
            value_matrix = js_divergence(norm_probs_1, norm_probs_2)
        else:
            value_matrix = js_divergence(layer_probs_1, layer_probs_2)
    else:
        metric_type = 'nwd'
        map_color = 'Blues'
        title = f"NWD ({'mean topk' if topk_mean else 'top-1'})"

        if topk_mean:
            # Take top-k probabilities for NWD calculation
            topk_probs_1 = np.take_along_axis(layer_probs_1, topk_indices_2, axis=-1)
            topk_probs_2 = np.take_along_axis(layer_probs_2, topk_indices_2, axis=-1)

            # Create sparse probability distributions for both models
            batch, seq_len, _ = topk_indices_2.shape
            vocab_size = layer_probs_1.shape[-1]
            sparse_probs_1 = np.zeros((batch, seq_len, vocab_size))
            sparse_probs_2 = np.zeros((batch, seq_len, vocab_size))

            for b in range(batch):
                for t in range(seq_len):
                    sparse_probs_1[b, t, topk_indices_2[b, t]] = topk_probs_1[b, t]
                    sparse_probs_2[b, t, topk_indices_2[b, t]] = topk_probs_2[b, t]

            sparse_probs_1 /= np.clip(sparse_probs_1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
            sparse_probs_2 /= np.clip(sparse_probs_2.sum(axis=-1, keepdims=True), 1e-10, 1.0)

            value_matrix = nwd(sparse_probs_1, sparse_probs_2)
        else:
            value_matrix = nwd(layer_probs_1, layer_probs_2)

    # Plot using model 1’s annotations and divergence matrix
    _plot_logit_lens_plotly(
        layer_logits=layer_logits_2,  # Use model 2's logits
        layer_preds=layer_preds_2,    # Use model 2's predictions
        layer_probs=layer_probs_2,   # Use model 2's probabilities
        topk_scores=topk_scores_2,   # Use model 2's top-k scores
        topk_indices=topk_indices_2, # Use model 2's top-k indices
        tokenizer=tokenizer_2,       # Use model 2's tokenizer
        input_ids=input_ids_2,       # Use model 2's input IDs
        start_ix=start_ix,
        layer_names=layer_names_2,   # Use model 2's layer names
        top_k=topk,
        normalize=True,
        metric_type=metric_type,
        map_color=map_color,
        value_matrix=value_matrix,
        title=title,
        block_step=block_step,
        token_font_size=token_font_size
    )

    clear_cuda_cache()
