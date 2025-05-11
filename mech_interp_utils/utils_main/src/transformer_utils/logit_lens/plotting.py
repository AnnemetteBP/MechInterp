from functools import partial
from typing import Tuple, List, Dict, Any, Optional
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


# ===================== Probs and logits for topk = 1 ============================
"""def text_to_input_ids(tokenizer:Any, text:str) -> torch.Tensor:

    toks = tokenizer.encode(text, return_tensors="pt")
    return torch.as_tensor(toks).view(1, -1).cpu()"""

def text_to_input_ids(tokenizer:Any, text:str, model:Optional[torch.nn.Module]=None) -> torch.Tensor:
    """ Encode inputs and move them to the model's device """
    toks = tokenizer.encode(text, return_tensors="pt")

    if model is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        toks = toks.to(device)
    
    return toks

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


def postprocess_logits(layer_logits:Any, normalize_probs:bool=False) -> Tuple[Any,Any]:
    """
    Process logits into stable probabilities and predictions.
    Works across float16, FP32, 8-bit, 4-bit, and low-bit quantized models.
    Mostly, this is an issue for low-bit models with all linear layers as QuantLinear.
    Only quantizing: layers_to_quant=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'] avoids this.
    However, this ONLY resolves not plotting a mostly empty logits plot!
    """
    if layer_logits.dtype == np.float16:
        layer_logits = layer_logits.astype(np.float32)
    
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    # Compute softmax for probabilities
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    # Clean probabilities
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    # Normalize if the sum is way off
    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs
        #layer_probs += np.random.normal(0, 1e-6, size=layer_probs.shape)

    layer_preds = layer_logits.argmax(axis=-1)

    return layer_preds, layer_probs


# ===================== Probs and logits for topk > 1 and for topk plot ============================
def postprocess_logits_tokp(layer_logits:Any, normalize_probs=False, top_n:int=5) -> Tuple[Any,Any,Any]:
        
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

    return layer_preds, layer_probs, top_n_scores


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


# ===================== The logit lens Plotter ============================
def _plot_logit_lens(
    layer_logits,
    layer_preds,
    layer_probs,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    save_fig_path:str|None=None,
    probs=False,
    ranks=False,
    kl=False,
    entropy=False,       
    top_down=False,
):
    end_ix = start_ix + layer_logits.shape[1]

    final_preds = layer_preds[-1]

    aligned_preds = layer_preds

    final_probs = layer_probs[-1]
    
    if kl:
        clip = 1 / (10 * layer_probs.shape[-1])
        #final_probs = layer_probs[-1]
        to_show = kl_div(final_probs, layer_probs, clip=clip)
        to_show = min_max_scale(to_show, 0, 10) # Normalize scale for plotting and comparison across models
    elif entropy:
        to_show = compute_entropy(layer_probs)
        to_show = min_max_scale(to_show, 0, 10)
    else:
        numeric_input = layer_probs if probs else layer_logits

        to_show = get_value_at_preds(numeric_input, final_preds)

        if not ranks:
            to_show = min_max_scale(to_show, 0, 1)

        if ranks:
            to_show = (numeric_input >= to_show[:, :, np.newaxis]).sum(axis=-1)

    _num2tok = np.vectorize(
        partial(num2tok, tokenizer=tokenizer, quotemark="'"), otypes=[str]
    )
    aligned_texts = _num2tok(aligned_preds)

    to_show = to_show[::-1]

    aligned_texts = aligned_texts[::-1]

    fig = plt.figure(figsize=(1.5 * to_show.shape[1], 0.375 * to_show.shape[0]))
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_kwargs = {"annot": aligned_texts, "fmt": ""}
    if kl:
        #vmin, vmax = None, None
        vmin, vmax = 0, 10
        title = "KL Divergence"

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r",
                "vmin": vmin,
                "vmax": vmax,
                #"annot": True,
                #"fmt": ".1f",
            }
        )
    elif entropy:
        vmin, vmax = None, None
        #vmin, vmax = 0, 10
        title = "Entropy"

        plot_kwargs.update(
            {
                "cmap": "coolwarm",
                "vmin": vmin,
                "vmax": vmax,
            }
        )
    elif ranks:
        vmax = 2000
        title = "Ranks"

        plot_kwargs.update(
            {
                "cmap": "Blues",
                "norm": mpl.colors.LogNorm(vmin=1, vmax=vmax),
                "annot": True,
            }
        )
    elif probs:
        title = "Probability"

        plot_kwargs.update({"cmap": "Blues", "vmin": 0, "vmax": 1})
    else:
        title = "Logits"
        vmin = np.percentile(to_show.reshape(-1), 5)
        vmax = np.percentile(to_show.reshape(-1), 95)
        #vmin = 0
        #vmax = 1

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40",
                "vmin": vmin,
                "vmax": vmax,
            }
        )

    sns.heatmap(to_show, **plot_kwargs)

    ax = plt.gca()
    input_tokens_str = _num2tok(input_ids[0].cpu())

    if layer_names is None:
        layer_names = ["Layer {}".format(n) for n in range(to_show.shape[0])]
    ylabels = layer_names[::-1]
    ax.set_yticklabels(ylabels, rotation=0)

    ax_top = ax.twiny()

    padw = 0.5 / to_show.shape[1]
    ax_top.set_xticks(np.linspace(padw, 1 - padw, to_show.shape[1]))

    ax_inputs = ax
    ax_targets = ax_top

    if top_down:
        ax.invert_yaxis()
        ax_inputs = ax_top
        ax_targets = ax

    ax_inputs.set_xticklabels(input_tokens_str[start_ix:end_ix], rotation=0)

    starred = [
        "* " + true if pred == true else " " + true
        for pred, true in zip(
            aligned_texts[0], input_tokens_str[start_ix + 1 : end_ix + 1]
        )
    ]
    ax_targets.set_xticklabels(starred, rotation=0)
    
    plt.title(title, fontsize=12, fontweight="bold", pad=10)

    if save_fig_path is not None:
        plt.savefig(save_fig_path)


def plot_logit_lens(
    model:Any,
    tokenizer:Any,
    input_ids:Any,
    start_ix:int,
    end_ix:int,
    save_fig_path:str|None=None,
    probs:bool=False,
    ranks:bool=False,
    kl:bool=False,
    entropy:bool=False,  
    block_step:int=1,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:list=['norm', 'lm_head'],
    top_down:bool=False,
    verbose:bool=False
):
    """
    Draws "logit lens" plots, and generalizations thereof.

    For background, see
        https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
        https://jalammar.github.io/hidden-states/

    `model`, `tokenizer` and `input_ids` should be familiar from the transformers library.  Other args are
     documented below.

    `model` should be a `transformers.PreTrainedModel` with an `lm_head`, e.g. `AutoModelForCausalLM`. This implementation works for LLaMAs, OLMos

    Note that using `start_ix` and `end_ix` is not equivalent to passed an `input_ids` sliced like `input_ids[start_ix:end_ix]`.  The LM will see the entire input you pass in as `input_ids`, no matter how you set `start_ix` and `end_ix`.  These "ix" arguments only control what is _displayed_.

    The boolean arguments `probs`, `ranks`, `entropy` and `kl` control the type of plot.  The options are:

        - Logits (the default plot type, if `probs`, `ranks` and `kl` are all False):
            - cell color: logit assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer

        - Probabilities:
            - cell color: probability assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer
        
        - Entropy:
            - cell color: entropy of probability assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer

        - Ranks:
            - cell color: ranking over the vocab assigned by each layer to the final layer's top-1 token prediction
            - cell text:  same as cell color

        - KL:
            - cell color: KL divergence of each layer's probability distribtion w/r/t the final layer's
            - cell text:  same as cell color

    `include_subblocks` and `decoder_layer_names` allow the creation of plots that go beyond what was done
    in the original blog post.  See below for details

    Arguments:

        probs:
            draw a "Probabilities" plot
        ranks:
            draw a "Ranks" plot (overrides `probs`)
        kl:
            draw a "KL" plot (overrides `probs`, `ranks`)
        entropy:
            draw a "Entropy" plot (overrides `probs`, `ranks`, `kl`)
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

            Prepending one or more of the last layers of the model, e.g. `['h11', 'final_layernorm', 'lm_head']` for GPT2, LLaMA: ['norm', 'lm_head']
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

    make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names,
                    decoder_layer_names=decoder_layer_names,
                    verbose=verbose)

    input_ids = text_to_input_ids(tokenizer, input_ids, model)

    layer_logits, layer_names = collect_logits(
        model, input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names,
        )
    
    layer_preds, layer_probs = postprocess_logits(layer_logits)


    _plot_logit_lens(
        layer_logits=layer_logits,
        layer_preds=layer_preds,
        layer_probs=layer_probs,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        save_fig_path=save_fig_path,
        probs=probs,
        ranks=ranks,
        kl=kl,
        entropy=entropy,
        layer_names=layer_names,
        top_down=top_down,
    )
