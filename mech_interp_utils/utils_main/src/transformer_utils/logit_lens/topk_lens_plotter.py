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


# ===================== Misc ============================
def clear_cuda_cache() -> None:
    """Clear GPU cache to avoid memory errors during operations"""
    torch.cuda.empty_cache()

def save_metrics_to_json(metrics:Dict, save_path:str) -> None:

    def convert_ndarray(o):
        if isinstance(o, np.ndarray):
            return o.tolist()  # Convert ndarray to a list
        elif isinstance(o, dict):
            return {k: convert_ndarray(v) for k, v in o.items()}  # Recursively convert dict
        elif isinstance(o, list):
            return [convert_ndarray(i) for i in o]  # Recursively convert list
        return o  # Return other objects as is

    # Convert the entire metrics dictionary to a JSON serializable format
    serializable_metrics = convert_ndarray(metrics)

    # Save the metrics to the specified path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)


def clean_prompt_list(prompts:List) -> List:
    # Ensure all entries are plain strings (non-empty after stripping)
    return [p for p in prompts if isinstance(p, str) and p.strip() and not isinstance(p, tuple)]

def clean_and_join_prompts(list_of_line_lists):
    cleaned_prompts = []
    for i, lines in enumerate(list_of_line_lists):
        try:
            prompt = " ".join(line.strip() for line in lines if isinstance(line, str) and line.strip())
            if prompt:
                cleaned_prompts.append(prompt)
        except Exception as e:
            print(f"[ERROR] Failed to clean sample {i}: {e}")
    return cleaned_prompts

# ===================== Topk Lens Hooks ============================
def topk_make_lens_hooks(model:Any, layer_names:Any, verbose=False) -> List:
    hook_handles = []
    
    # Print the layer names for clarity
    print(f"[Debug] Layer names being passed: {layer_names}")

    for layer_name in layer_names:
        try:
            # Debug: Trying to access the layer by name
            print(f"[Debug] Trying to access layer: {layer_name}")
            layer = dict(model.named_modules())[layer_name]  # Get the layer by name
            print(f"[Debug] Successfully found layer: {layer_name}")

            # Register the hook for the layer
            handle = layer.register_forward_hook(my_hook)
            hook_handles.append(handle)
        
        except KeyError:
            print(f"[Error] Layer {layer_name} not found in model.")  # If the layer is not found
        except Exception as e:
            print(f"[Error] Failed to register hook for {layer_name}: {str(e)}")

    if verbose:
        print(f"[Debug] Hook handles: {hook_handles}")

    return hook_handles


def my_hook(module, input, output) -> Any:
    # Your custom hook logic here
    print(f"[Hook] Layer {module} received input {input} and output {output}")
    return output  # Pass the output back as usual


def make_layer_names_topk(model):
    # Generate layer names for LLaMA
    layer_names = []

    # Access the layers from the model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # The model is structured with a 'model' submodule that contains 'layers'
        for i in range(len(model.model.layers)):
            layer_names.append(f"model.layers.{i}")
        layer_names.append("model.embed_tokens")  # Add the embedding layer

    elif hasattr(model, 'layers'):
        # For models without the 'model' submodule
        for i in range(len(model.layers)):
            layer_names.append(f"layers.{i}")

    else:
        print("[Error] Cannot find layers in the model.")
    
    return layer_names


# ===================== Tokenize input texts ============================
def text_to_input_ids(tokenizer:Any, text:Union[str, List[str]], model:Optional[torch.nn.Module]=None, add_special_tokens:bool=True, pad_to_max_length=False) -> torch.Tensor:
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
def collect_logits(model, input_ids, layer_names, decoder_layer_names) -> Tuple:
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
def postprocess_logits_tokp(layer_logits: Any, normalize_probs=False, top_n:int=5, return_scores:bool=True) -> Tuple[Any, Any, Any]:
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
    
# ===================== Avg Stability ============================
def calculate_avg_stability(stability_top1, stability_topk) -> Tuple[Any, Any]:
    """
    Calculate the average stability score based on top-1 and top-k metrics.
    
    Args:
        stability_top1 (np.ndarray): Stability metric for top-1 predictions, shape = [tokens]
        stability_topk (np.ndarray): Stability metric for top-k predictions, shape = [tokens]
    
    Returns:
        tuple: average_stability_top1, average_stability_topk
    """
    avg_stability_top1 = np.mean(stability_top1[stability_top1 != -1])  # Average stability for top-1 predictions
    avg_stability_topk = np.mean(stability_topk[stability_topk != -1])  # Average stability for top-k predictions

    return avg_stability_top1, avg_stability_topk


def compute_stability_metrics(layer_preds, topk_indices, target_ids) -> Tuple:
    """
    Compute the stability metrics: how quickly the model predicts the correct token 
    in top-1 and top-k predictions relative to the depth (layers).

    Args:
        layer_preds (np.ndarray): Predicted tokens for each layer, shape = [layers, tokens]
        topk_indices (np.ndarray): Indices of the top-k predicted tokens for each layer, shape = [layers, tokens, topk]
        target_ids (np.ndarray): Ground truth token ids, shape = [tokens]

    Returns:
        tuple: stability_top1, stability_topk
    """
    
    # 1. Stability in terms of top-1 prediction: Find the first layer where the model's top-1 prediction is correct
    stability_top1 = np.argmax(layer_preds == target_ids, axis=0)  # First layer where correct token is top-1
    stability_top1 = np.where(stability_top1 == 0, -1, stability_top1)  # If no layer is correct, return -1
    
    # 2. Stability in terms of top-k prediction: Find the first layer where the correct token is in the top-k
    stability_topk = np.argmax(np.any(topk_indices == target_ids[None, :, None], axis=-1), axis=0)  # First layer where correct token is in top-k
    stability_topk = np.where(stability_topk == 0, -1, stability_topk)  # If no layer is correct, return -1
    
    return stability_top1, stability_topk


# ===================== Correctness ============================
def compute_correctness_metrics(layer_preds, target_ids, topk_indices=None) -> Tuple:
    correct_1 = (layer_preds[-1] == target_ids).astype(int)

    if topk_indices is not None:
        # target_ids: [tokens], reshape to [1, tokens, 1] to broadcast against [layers, tokens, topk]
        target_ids_broadcasted = target_ids[None, :, None]
        correct_topk = np.any(topk_indices == target_ids_broadcasted, axis=-1).astype(int)
    else:
        correct_topk = None

    return correct_1, correct_topk

# ===================== Collect layer logits ============================
def collect_batch_logits(model, input_ids, layer_names, outputs) -> Tuple:
    collected_logits = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # Get hidden states only
        collected_logits.append(output.detach().cpu().numpy())

    handles = []
    for name in layer_names:
        layer = dict([*model.named_modules()])[name]
        handles.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_ids)

        for h in handles:
            h.remove()

    # Stack into shape [num_layers, batch, seq_len, hidden_size]
    logits = np.stack(collected_logits, axis=0)

    return logits, layer_names


# ===================== Clipping and metric helpers ============================
def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)

def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark

def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)

def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)

def kl_div(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)

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

def compute_kl_divergence(logits1, logits2):
    probs1 = scipy.special.softmax(logits1, axis=-1)
    probs2 = scipy.special.softmax(logits2, axis=-1)
    return np.sum(kl_div(probs1, probs2), axis=-1)

def compute_wasserstein_from_json(file_a, file_b, key="logit_mean"):
    with open(file_a) as f1, open(file_b) as f2:
        m1 = json.load(f1)[key]
        m2 = json.load(f2)[key]
    return wasserstein_distance(m1, m2)


# ===================== Topk-N Analysis ============================
def collect_logit_lens_metrics_batch(
    model:Any,
    tokenizer:Any,
    prompts:List[str],
    start_ix:int,
    end_ix:int,
    topk:int=5,
    prompt_type:str="text",
    max_prompts:int=50,
) -> List:
    assert isinstance(prompts, list), "prompts should be a list of strings"
    prompts = prompts[:max_prompts]

    results = []

    for idx, prompt in enumerate(prompts):
        input_ids_tensor = text_to_input_ids(tokenizer, prompt, model)
        input_ids_list = input_ids_tensor[0].tolist()

        layer_names = make_layer_names_topk(model)

        hook_handles = topk_make_lens_hooks(model, layer_names=layer_names)
        if hook_handles is None:
            print(f"[Error] No hooks were registered for prompt {idx}. Skipping.")
            continue

        try:
            layer_logits, _ = collect_batch_logits(model, input_ids_tensor, layer_names, [])

            # Handle shape: [layers, batch, seq_len, hidden] → [layers, seq_len, hidden]
            if isinstance(layer_logits, list):
                layer_logits = np.stack(layer_logits, axis=0)
            if layer_logits.ndim == 4 and layer_logits.shape[1] == 1:
                layer_logits = layer_logits[:, 0, :, :]
            elif layer_logits.ndim != 3:
                raise ValueError(f"Expected layer_logits to be 3D but got shape {layer_logits.shape}")

            # Project to vocab if it's still in hidden state space
            if layer_logits.shape[-1] == model.config.hidden_size:
                hidden_states = torch.tensor(layer_logits, dtype=torch.float32).to(model.device)
                with torch.no_grad():
                    logits = model.lm_head(hidden_states)
                layer_logits = logits.cpu().numpy()
                # Clean logits immediately
                layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            # Slice to match the token prediction window
            layer_logits = layer_logits[:, start_ix + 1:end_ix + 1, :]

            # Top-k prediction postprocessing
            layer_preds, layer_probs, _ = postprocess_logits_tokp(layer_logits, top_n=topk)
            topk_indices = np.argsort(layer_probs, axis=-1)[..., -topk:][..., ::-1]

            # Ground truth target token IDs
            target_ids = input_ids_tensor[0, start_ix + 1:end_ix + 1].cpu().numpy()

            # Metrics: entropy, correctness
            entropy = compute_entropy(layer_probs)
            prob_correct = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1).squeeze(-1)
            logit_correct = np.take_along_axis(layer_logits, layer_preds[..., None], axis=-1).squeeze(-1)

            # Broadcast target IDs for top-k correctness
            target_ids_broadcasted = target_ids[None, :, None]
            correct_1 = (layer_preds == target_ids[None, :]).astype(int)
            correct_topk = np.any(topk_indices == target_ids_broadcasted, axis=-1).astype(int)

            # Stability metrics
            stability_top1, stability_topk = compute_stability_metrics(layer_preds, topk_indices, target_ids)

            # Aggregated stats
            correct_1_std = np.std(correct_1, axis=1).tolist()
            correct_topk_std = np.std(correct_topk, axis=1).tolist()
            vocab_size = tokenizer.vocab_size
            norm_entropy = (entropy / np.log(vocab_size)).tolist()

            # KL divergence between layers
            layer_kl_divergences = [
                compute_kl_divergence(layer_logits[i], layer_logits[i + 1])
                for i in range(len(layer_logits) - 1)
            ]

            # Store results
            metrics = {
                "prompt": input_ids_tensor.tolist(),
                "decoded_prompt_str": tokenizer.decode(input_ids_list),
                "tokens": tokenizer.convert_ids_to_tokens(input_ids_list),
                "prompt_type": prompt_type,
                "target_ids": target_ids.tolist(),
                "target_tokens": tokenizer.convert_ids_to_tokens(target_ids.tolist()),
                "layer_names": layer_names,
                "correct_1": correct_1.mean(axis=1).tolist(),
                "correct_topk": correct_topk.mean(axis=1).tolist(),
                "correct_1_std": correct_1_std,
                "correct_topk_std": correct_topk_std,
                "correct_1_by_position": correct_1.T.tolist(),
                "correct_topk_by_position": correct_topk.T.tolist(),
                "entropy": entropy.mean(axis=1).tolist(),
                "normalized_entropy": norm_entropy,
                "logit_mean": logit_correct.mean(axis=1).tolist(),
                "prob_mean": prob_correct.mean(axis=1).tolist(),
                "stability_top1": stability_top1.tolist(),
                "stability_topk": stability_topk.tolist(),
                "layer_kl_divergences": layer_kl_divergences,
            }

            results.append(metrics)

        finally:
            if hook_handles:
                for handle in hook_handles:
                    handle.remove()
            for var in ['input_ids_tensor', 'layer_logits', 'layer_preds', 'layer_probs']:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()

    return results


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
    topk_mean:bool=True,
    normalize=True,
    metric_type:str|None=None,
    map_color='Cividis',
    value_matrix=None,
    rank_matrix_raw:Optional[Any|None]=None,
    pred_ranks:Optional[Any|None]=None,    
    title:str|None=None,
    block_step:int=1,
    token_font_size:int=12
) -> None:
    num_layers, num_tokens, vocab_size = layer_logits.shape
    end_ix = start_ix + num_tokens

    input_token_ids = input_ids[0][start_ix:end_ix]
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_token_ids]

    # Top-axis label logic
    full_token_ids = input_ids[0]
    next_token_ids = full_token_ids[start_ix + 1:end_ix + 1]
    next_token_text = [tokenizer.decode([tid]) for tid in next_token_ids]
    if len(next_token_text) < num_tokens:
        next_token_text.append("")

    # Predicted token text for each layer/token
    pred_token_text = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)

    topk_tokens = np.vectorize(lambda idx: tokenizer.decode([idx]), otypes=[str])(topk_indices)

    # Use mean top-k probability as default value_matrix if not provided
    if value_matrix is None:
        value_matrix = topk_scores.mean(axis=-1)

    # Layer filtering using block_step
    keep_idxs = [0] + list(range(1, num_layers - 1, block_step)) + [num_layers - 1]
    value_matrix = value_matrix[keep_idxs]
    pred_token_text = pred_token_text[keep_idxs]
    layer_preds = layer_preds[keep_idxs]
    topk_scores = topk_scores[keep_idxs]
    topk_tokens = topk_tokens[keep_idxs]
    layer_names = [layer_names[i] for i in keep_idxs]

    if metric_type == "ranks":
        if pred_ranks is None:
            raise ValueError("pred_ranks must be provided for metric_type='ranks'")
        pred_ranks = pred_ranks[keep_idxs]

    # ── Hover-text and normalization logic ─────────────────────
    hovertext = []

    for i in range(len(layer_names)):
        row = []
        for j in range(num_tokens):
            if metric_type == "ranks":
                raw_val = pred_ranks[i, j]
            else:
                raw_val = value_matrix[i, j]

            # Hover display logic
            try:
                if metric_type == "entropy":
                    hover_val = f"<b>Entropy:</b> {raw_val:.3f}<br>"
                elif metric_type == "logits":
                    hover_val = f"<b>Logit:</b> {raw_val:.3f}<br>"
                elif metric_type == "kl":
                    hover_val = f"<b>KL Divergence:</b> {raw_val:.3f}<br>"
                elif metric_type == "ranks":
                    hover_val = f"<b>Rank:</b> {raw_val}<br>"
                else:
                    hover_val = f"<b>Mean Top-{top_k} Prob:</b> {raw_val:.3f}<br>"
            except:
                hover_val = f"<b>{metric_type}:</b> N/A<br>"

            # ALWAYS include predicted token and top-k predictions
            hover = hover_val
            hover += f"<b>Pred:</b> {pred_token_text[i, j]}<br><b>Top-{top_k}:</b><br>"
            for k in range(top_k):
                hover += f"&nbsp;&nbsp;{topk_tokens[i, j, k]}: {topk_scores[i, j, k]:.3f}<br>"

            row.append(hover)
        hovertext.append(row)

    # ── Normalization of ranks for coloring ────────────────────
    if normalize:
        if metric_type == "ranks":
            # Normalize ranks for coloring (log scale)
            vmax = 2000  # Max value for the rank
            norm = mpl.colors.LogNorm(vmin=1, vmax=vmax)  # LogNorm for better color scaling

            # Apply log scale and normalization to the value matrix for proper coloring
            value_matrix = np.log10(pred_ranks)  # Use log scale for ranks
            value_matrix = min_max_scale(value_matrix, 0, 1)  # Scale to [0, 1] for proper coloring
        else:
            value_matrix = min_max_scale(value_matrix, 0, 1)

    # Decode predictions from selected layers
    pred_tokens_str = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)

    # Prepare true next tokens
    true_next_token_text = next_token_text[:num_tokens]
    if len(true_next_token_text) < num_tokens:
        true_next_token_text += [""] * (num_tokens - len(true_next_token_text))

    # Ground truth matrix
    correct_tokens_matrix = np.tile(true_next_token_text, (pred_tokens_str.shape[0], 1))

    # Compute match between predicted and true next tokens
    is_correct = (pred_tokens_str == correct_tokens_matrix)

    # Check which predictions match true next tokens
    is_correct = (pred_tokens_str == correct_tokens_matrix)
    value_matrix = value_matrix[::-1]
    pred_token_text = pred_token_text[::-1]
    hovertext = hovertext[::-1]
    is_correct = is_correct[::-1]
    
    fig = go.Figure()

    # Use rank values for text display if plotting ranks, else predicted tokens
    if metric_type == "ranks":
        cell_text = pred_ranks[::-1].astype(str)  # Match flipped value_matrix
    else:
        cell_text = pred_token_text

    # Heatmap trace
    fig.add_trace(go.Heatmap(
        z=value_matrix,
        x=list(range(num_tokens)),
        y=list(range(len(layer_names))),
        text=cell_text,
        #text=pred_token_text,
        texttemplate="%{text}",
        textfont=dict(size=token_font_size),
        hovertext=hovertext,
        hoverinfo='text',
        colorscale=map_color,
        zmin=np.min(value_matrix),
        zmax=np.max(value_matrix),
        colorbar=dict(
            title="log₁₀(rank)" if metric_type == "ranks" else None
        ),
    ))

    # Plot shapes
    correct_y, correct_x = np.where(is_correct == 1)
    for y, x in zip(correct_y, correct_x):
        fig.add_shape(
            type="rect",
            x0=x - 0.5, x1=x + 0.5,
            y0=y - 0.5, y1=y + 0.5,
            line=dict(color="black", width=2),
            layer="above"
        )
    # Dummy scatter trace to activate xaxis2 (next-token labels)
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
        font=dict(family="DejaVu Sans", size=14), # or 'Noto Sans'
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


def plot_topk_logit_lens(
    model:Any,
    tokenizer:Any,
    inputs:Union[str, List[str], None],
    start_ix:int,
    end_ix:int,
    topk:int=5,
    plot_topk_lens:bool=True,
    topk_mean:bool=True,
    save_fig_path:str|None=None,
    json_log_path:str|None=None,
    probs:bool=False,
    entropy:bool=False,
    kl:bool=False,
    ranks:bool=False,
    block_step:int=1,
    token_font_size:int=12,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:List=['norm', 'lm_head'],
    top_down:bool=False,
    verbose:bool=False,
    pad_to_max_length:bool=False,
    model_precision:Optional[torch.dtype|None]=None
) -> None:
    
    rank_matrix_raw = None
    pred_ranks = None

    metric_type = None
    if model_precision:
        model = model.to(model_precision) 
        
    # Handle input format and ensure it's a list of prompts
    if isinstance(inputs, str):
        inputs = [inputs]
    elif inputs is None:
        inputs = ["What is y if y=2*2-4+(3*2)"]  # Default prompt

    if plot_topk_lens:
        # Create layer names for the model layers
        layer_names = make_layer_names(
            model,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            decoder_layer_names=decoder_layer_names
        )

        # Register hooks
        #hook_handles = make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names, decoder_layer_names=decoder_layer_names, verbose=verbose)
        make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names, decoder_layer_names=decoder_layer_names, verbose=verbose)
        # Tokenize inputs with padding control
        input_ids = text_to_input_ids(tokenizer, inputs, model, pad_to_max_length=pad_to_max_length)

        # Collect logits from the model
        layer_logits, layer_names = collect_logits(model, input_ids, layer_names, decoder_layer_names)

        # Process logits to get top-k predictions and probabilities
        layer_preds, layer_probs, _ = postprocess_logits_tokp(layer_logits, top_n=topk)

        # Clean up probabilities before sorting
        layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

        # Compute top-k indices and scores from cleaned probs
        topk_indices = np.argsort(layer_probs, axis=-1)[..., -topk:][..., ::-1]
        topk_scores = np.take_along_axis(layer_probs, topk_indices, axis=-1)

        # Entropy (mean over top-k only)
        if entropy:
            map_color = 'RdBu_r'
            if topk_mean:
                clipped_probs = np.take_along_axis(layer_probs, topk_indices, axis=-1)
                log_probs = np.log(np.clip(clipped_probs, 1e-10, 1.0))
                value_matrix = -np.sum(clipped_probs * log_probs, axis=-1)
            else:
                value_matrix = compute_entropy(layer_probs)
            metric_type = 'entropy'
            title = f"Entropy ({'mean topk' if topk_mean else 'full dist'})"

        # Probabilities
        elif probs:
            map_color = 'Blues'
            if topk_mean:
                value_matrix = topk_scores.mean(axis=-1)
            else:
                value_matrix = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1).squeeze(-1)
            metric_type = 'probs'
            title = f"Probabilities ({'mean topk' if topk_mean else 'top-1'})"

        # KL-Divergence block
        elif kl:  # Ensure we’re in the KL Divergence block
            map_color = 'Cividis'

            if topk_mean:
                # Clip and normalize
                clipped_probs = np.take_along_axis(layer_probs, topk_indices, axis=-1)  # (L, T, K)
                log_probs = np.log(np.clip(clipped_probs, 1e-10, 1.0))
                q_probs = np.exp(log_probs)  # Just to be safe: ensure proper probs (though they already are)

                # Scatter q_probs into full vocab-size tensor
                q_full_probs = np.zeros_like(layer_probs)
                for k in range(topk_indices.shape[-1]):
                    np.put_along_axis(
                        q_full_probs,
                        topk_indices[:, :, k:k+1],
                        q_probs[:, :, k:k+1],
                        axis=-1
                    )

                # Compute KL divergence between full probs and top-k projected
                value_matrix = kl_div(layer_probs, q_full_probs)
            
            else:
                # Just compare full predicted distribution to one-hot of top-1 pred
                pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)
                pred_probs_full = np.zeros_like(layer_probs)
                np.put_along_axis(pred_probs_full, layer_preds[..., None], pred_probs, axis=-1)
                value_matrix = kl_div(layer_probs, pred_probs_full)

            metric_type = 'kl'
            title = f"KL Divergence ({'mean topk' if topk_mean else 'top-1'})"

        # Ranks
        elif ranks:
            map_color = 'Blues'

            # ── Raw rank matrix ────────────────────────────────────
            if topk_mean:
                # Get ranks for top-k tokens and average over them
                topk_probs = np.take_along_axis(layer_probs, topk_indices, axis=-1)  # shape (L, T, k)
                ranks_matrix = (layer_probs[..., None] >= topk_probs[:, :, None, :]).sum(axis=-2)  # (L, T, k)
                value_matrix = ranks_matrix.mean(axis=-1)  # Average over top-k (L, T)
            else:
                pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)  # shape (L, T, 1)
                value_matrix = (layer_probs >= pred_probs).sum(axis=-1)  # Compare and sum (L, T)

            # Calculate ranks based on probabilities, sorting in descending order
            rank_matrix_raw = np.argsort(-layer_probs, axis=-1)  # (L, T, Vocab)

            # Now we need to calculate the rank of the true predicted token for each layer/token
            # For each token, find its position in the sorted list of probabilities
            pred_ranks = np.take_along_axis(rank_matrix_raw, layer_preds[..., None], axis=-1).squeeze(-1) + 1  # +1 to make rank start from 1

            metric_type = "ranks"
            title = f"Prediction Rank ({'mean topk' if topk_mean else 'top-1'})"

        # Logits
        else:
            map_color = 'thermal'
            topk_logits = np.take_along_axis(layer_logits, topk_indices, axis=-1)
            if topk_mean:
                value_matrix = topk_logits.mean(axis=-1)
            else:
                value_matrix = np.take_along_axis(layer_logits, layer_preds[..., None], axis=-1).squeeze(-1)
            metric_type = 'logits'
            title = f"Logits ({'mean topk' if topk_mean else 'top-1'})"

    else:
        # Collect metrics if needed
        if json_log_path is not None:
            metrics = collect_logit_lens_metrics_batch(
                model, tokenizer, prompts=inputs, start_ix=start_ix, end_ix=end_ix, topk=topk, prompt_type="text", max_prompts=50
            )
            if json_log_path:
                save_metrics_to_json(metrics, json_log_path)

    # Plot logit lens if requested
    if plot_topk_lens:
        _plot_logit_lens_plotly(
            layer_logits=layer_logits,
            layer_preds=layer_preds,
            layer_probs=layer_probs,
            topk_scores=topk_scores,
            topk_indices=topk_indices,
            tokenizer=tokenizer,
            input_ids=input_ids,
            start_ix=start_ix,
            layer_names=layer_names,
            top_k=topk,
            topk_mean=topk_mean,
            normalize=True,
            metric_type=metric_type,
            map_color=map_color,
            value_matrix=value_matrix,
            rank_matrix_raw=rank_matrix_raw,
            pred_ranks=pred_ranks,  
            title=title,
            block_step=block_step,
            token_font_size=token_font_size
        )

    # Clean up GPU memory to avoid memory overflow after operations
    clear_cuda_cache()
