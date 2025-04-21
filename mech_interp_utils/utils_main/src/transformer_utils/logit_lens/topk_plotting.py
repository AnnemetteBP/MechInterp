from __future__ import annotations
from typing import Tuple, List, Dict, Literal, Optional, Any

from functools import partial
import torch
import numpy as np

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from scipy.stats import entropy

import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa

from ..util.python_utils import make_print_if_verbose

from .hooks import make_lens_hooks
from .layer_names import make_layer_names


# ===================== Encode input ============================
def text_to_input_ids(tokenizer: Any, text: str) -> torch.Tensor:
    toks = tokenizer.encode(text, return_tensors="pt")
    # Ensure batch dimension is present (batch size 1)
    return toks.cpu()

# ===================== Extract logits from hooks ============================
def collect_logits(model, input_ids, layer_names, decoder_layer_names):
    model._last_resid = None
    with torch.no_grad():
        _ = model(input_ids)
    model._last_resid = None

    # Use np.stack if model._layer_logits[name] are arrays of equal shape.
    layer_logits = np.stack([model._layer_logits[name] for name in layer_names], axis=0)
    return layer_logits, layer_names

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

"""def postprocess_logits(layer_logits):
    layer_preds = layer_logits.argmax(axis=-1)

    layer_probs = scipy.special.softmax(layer_logits, axis=-1)

    return layer_preds, layer_probs"""

# ===================== Post-process logits ============================
def postprocess_logits(layer_logits: Any, normalize_probs: bool = False) -> Tuple[Any, Any]:
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs

    layer_preds = layer_logits.argmax(axis=-1)
    return layer_preds, layer_probs

def postprocess_logits_topk(layer_logits: Any, tokenizer:Any, normalize_probs: bool = False, top_n: int = 5) -> Tuple[Any, Any, Any]:
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs

    top_n_idx = np.argsort(layer_probs, axis=-1)[:, :, -top_n:]
    top_n_vals = np.take_along_axis(layer_probs, top_n_idx, axis=-1)
    top_n_labels = np.array([[num2tok(idx, tokenizer, "'") for idx in row] for row in top_n_idx.reshape(-1, top_n)]).reshape(top_n_idx.shape)

    return top_n_idx, top_n_vals, top_n_labels


"""def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)"""
def get_value_at_preds(values, preds):
    """
    values: [seq_len, vocab_size]
    preds: [seq_len, top_k]
    returns: [seq_len, top_k] – top-k probabilities per token
    """
    # Ensure preds is 2D
    if preds.ndim == 1:
        preds = preds[:, None]  # Make it a 2D array if it's 1D
    
    seq_len, top_k = preds.shape
    return values[np.arange(seq_len)[:, None], preds]


def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark


def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)


def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)


def kl_div(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)


"""def compute_entropy(probs, axis=-1, clip=1e-16):
    probs = clipmin(probs, clip)
    return entropy(probs, axis=axis)"""

def compute_entropy(probs):
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_probs, axis=-1)

def js_divergence(p, q, axis=-1, clip=1e-16):
    """Computes Jensen-Shannon divergence between two probability distributions."""
    p, q = clipmin(p, clip), clipmin(q, clip)
    m = (p + q) / 2
    return (kl_div(p, m, axis=axis, clip=clip) + kl_div(q, m, axis=axis, clip=clip)) / 2


def maybe_batchify(p):
    """ Normalize shape for e.g., wasserstein """
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    return p


def min_max_scale(arr, new_min, new_max):
    """Scales an array to a new range [new_min, new_max] for plotting and comparison of normalized values across models."""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr

# ===================== Top-k plotting function ============================
def _plot_topk_lens(
    topk_idx:np.ndarray,
    topk_val:np.ndarray,
    topk_labels:np.ndarray,
    layer_logits:np.ndarray,
    tokenizer:Any,
    input_ids:torch.Tensor,
    start_ix:int,
    end_ix:int,
    layer_names:list[str],
    save_fig_path:str|None=None,
    probs:bool=False,
    kl:bool=False,
    entropy:bool=False,
    top_down:bool=True,
    selected_layers:Optional[List[int]]=[5, 10, 15],
):
    """
    Plot a single heatmap for selected layers × top-k ranks.
    Each row = one top-k rank from a specific layer.
    Columns = tokens.
    """

    if entropy or kl:
        num_layers, total_tokens, vocab_size = layer_logits.shape
    else:
        num_layers, total_tokens, top_k = topk_val.shape

    end_ix = min(end_ix, total_tokens)
    input_tokens = [tokenizer.decode([tok]) for tok in input_ids[0, start_ix:end_ix]]

    # Handle layer selection
    layer_indices = list(range(num_layers)) if selected_layers is None else selected_layers
    ylabels = []
    heat_data = []
    label_data = []

    for layer in (reversed(layer_indices) if top_down else layer_indices):
        if entropy:
            probs = scipy.special.softmax(layer_logits[layer], axis=-1)
            top_k = topk_idx.shape[-1]
            for k in range(top_k):
                preds = topk_idx[layer, start_ix:end_ix, k]
                pred_probs = get_value_at_preds(probs[start_ix:end_ix], preds)
                entropy_vals = -pred_probs * np.log(np.clip(pred_probs, 1e-10, 1.0))
                
                # Flatten entropy_vals and add to heat_data and label_data
                entropy_vals_flattened = entropy_vals.flatten()
                
                heat_data.append(entropy_vals_flattened)
                #label_data.append([f"{val:.2f}" for val in entropy_vals_flattened])
                tokens = topk_labels[layer, start_ix:end_ix, k].flatten()
                label_data.append([f"{tok}\n{val:.2f}" for tok, val in zip(tokens, entropy_vals_flattened)])

                ylabels.append(f"{layer_names[layer]} – Entropy@Top-{k+1}")

        elif kl:
            top_k = topk_idx.shape[-1]
            ref_probs = scipy.special.softmax(layer_logits[layer_indices[-1]], axis=-1)
            this_probs = scipy.special.softmax(layer_logits[layer], axis=-1)
            for k in range(top_k):
                preds = topk_idx[layer, start_ix:end_ix, k]
                p = get_value_at_preds(this_probs[start_ix:end_ix], preds)
                q = get_value_at_preds(ref_probs[start_ix:end_ix], preds)
                kl_vals = kl_summand(p, q)
                
                # Flatten KL values and add to heat_data and label_data
                kl_vals_flattened = kl_vals.flatten()
                
                heat_data.append(kl_vals_flattened)
                #label_data.append([f"{val:.2f}" for val in kl_vals_flattened])
                tokens = topk_labels[layer, start_ix:end_ix, k].flatten()
                label_data.append([f"{tok}\n{val:.2f}" for tok, val in zip(tokens, kl_vals_flattened)])

                ylabels.append(f"{layer_names[layer]} – KL@Top-{k+1}")

        else:
            for k in range(top_k):
                heat_row = topk_val[layer, start_ix:end_ix, k]
                label_row = topk_labels[layer, start_ix:end_ix, k]
                heat_data.append(heat_row.flatten())  # Flatten the data for plotting
                label_data.append(label_row.flatten())  # Flatten the labels
                ylabels.append(f"{layer_names[layer]} – Top-{k+1}")

    heat_data = np.array(heat_data)
    label_data = np.array(label_data)
    heat_data = min_max_scale(heat_data, 0, 1)  # Normalize across all rows

    # Plot setup
    plot_kwargs = {
        "annot": label_data,
        "fmt": "",
        "xticklabels": input_tokens,
        "yticklabels": ylabels,
    }

    if kl:
        title = "KL Divergence"
        plot_kwargs.update({"cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r", "vmin": 0, "vmax": 10})
    elif entropy:
        title = "Entropy"
        plot_kwargs.update({"cmap": "coolwarm", "vmin": 0, "vmax": 10})
    elif probs:
        title = "Top-1 Probability"
        plot_kwargs.update({"cmap": "Blues_r", "vmin": 0, "vmax": 1})
    else:
        title = "Top-1 Logit"
        plot_kwargs.update({"cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40", "vmin": 0, "vmax": 1})

    plt.figure(figsize=(1.6 * (end_ix - start_ix), 0.6 * len(ylabels)))
    plt.rcParams['font.family'] = 'Times New Roman'
    ax = sns.heatmap(heat_data, **plot_kwargs)

    ax.set_yticklabels(ylabels, rotation=0)

    # Add top-aligned token labels
    ax_top = ax.twiny()
    padw = 0.5 / heat_data.shape[1]
    ax_top.set_xticks(np.linspace(padw, 1 - padw, heat_data.shape[1]))
    ax_top.set_xticklabels(input_tokens, rotation=90 if len(input_tokens) > 20 else 0)

    if top_down:
        ax.invert_yaxis()

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_fig_path:
        plt.savefig(save_fig_path)
    else:
        plt.show()


def plot_topk_lens(
    model: Any,
    tokenizer: Any,
    input_ids: str,
    start_ix: int,
    end_ix: int,
    topk_n: int,
    save_fig_path: str | None = None,
    probs: bool = False,
    kl: bool = False,
    entropy: bool = False,
    block_step: int = 1,
    include_input: bool = True,
    force_include_output: bool = True,
    include_subblocks: bool = False,
    decoder_layer_names: list = ['norm', 'lm_head'],
    top_down: bool = True,
    verbose: bool = False
):
    """
    Draws "topk lens" plots, and generalizations thereof.

    `model`, `tokenizer` and `input_ids` should be familiar from the transformers library.  Other args are
     documented below.

    Here `model` should be a `transformers.PreTrainedModel` with an LLaMA / OLMo architecture
        - to work for e.g., GPT2 see orgiginal nostalgebraist: https://github.com/nostalgebraist/transformer-utils

    Note that using `start_ix` and `end_ix` is not equivalent to passed an `input_ids` sliced like `input_ids[start_ix:end_ix]`. 
    The LM will see the entire input you pass in as `input_ids`, no matter how you set `start_ix` and `end_ix`.  These "ix" arguments only control what is _displayed_.

    The boolean arguments `probs`, `kl`, `entropy` control the type of plot.  The options are:

        - Logits (the default plot type, if `probs`, `ranks` and `kl` are all False):
            - cell color: logit assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer

        - Probabilities:
            - cell color: probability assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer

        - KL:
            - cell color: KL divergence of each layer's probability distribtion w/r/t the final layer's
            - cell text:  same as cell color / top-1 token prediction at each layer

        - Entropy:
            - cell color: entropy of each layer's probability distribtion w/r/t the final layer's
            - cell text:  same as cell color / top-1 token prediction at each layer

    `include_subblocks` and `decoder_layer_names` allow the creation of plots that go beyond what was done
    in the original blog post.  See below for details

    Arguments:

        topk_n:
            topk to plot for each selected layer.
        probs:
            draw a "Probabilities" plot for only the selected layers for chosen topk tokens.
            each selected layer will occur in the heatmap corrosponding to topk_n.
        kl:
            draw a "KL" plot (overrides `probs`)
        entropy:
            draw a "Entropy" plot (overrides `probs`, `entropy`)
        save_fig_path:
            save a plot to path if not None: str | None
        layer_names:
            List[int] where int is the layer. Plots each selected layer topk-times.
        block_step:
            stride when choosing blocks to plot, e.g. block_step=2 skips every other block
        include_input:
            whether to treat the input embeddings (before any blocks have been applied) as a "layer"
        force_include_output:
            whether to include the final layer in the plot, even if the passed `block_step` would otherwise skip it
        include_subblocks:
            if True, includes predictions after the only the attention part of each block, along with those after the
            full block
        decoder_layer_names:
            defines the subset of the model used to "decode" hidden states.

            The default value `['final_layernorm', 'lm_head']` corresponds to the ordinary "logit lens," where
            we decode each layer's output as though it were the output of the final block.

            Prepending one or more of the last layers of the model, e.g. `['h11', 'final_layernorm', 'lm_head']`
            for a 12-layer model, will treat these layers as part of the decoder.  In the general case, this is equivalent
            to dropping different subsets of interior layers and watching how the output varies.
    """
        
    layer_names = make_layer_names(
        model,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names
    )

    make_lens_hooks(
        model,
        start_ix=start_ix,
        end_ix=end_ix,
        layer_names=layer_names,
        decoder_layer_names=decoder_layer_names,
        verbose=verbose
    )

    input_ids = text_to_input_ids(tokenizer, input_ids)
    layer_logits, layer_names = collect_logits(model, input_ids, layer_names, decoder_layer_names)
    norm_vals = scipy.special.softmax(layer_logits, axis=-1) if (probs or entropy or kl) else layer_logits

    topk_idx, topk_val, topk_labels = postprocess_logits_topk(
        norm_vals,
        tokenizer=tokenizer,
        normalize_probs=probs,
        top_n=topk_n
    )

    if kl or entropy:
        topk_val = None

    _plot_topk_lens(
        topk_idx=topk_idx,
        topk_val=topk_val,
        topk_labels=topk_labels,
        layer_logits=layer_logits,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        end_ix=end_ix,
        layer_names=layer_names,
        save_fig_path=save_fig_path,
        probs=probs,
        kl=kl,
        entropy=entropy,
        top_down=top_down
    )