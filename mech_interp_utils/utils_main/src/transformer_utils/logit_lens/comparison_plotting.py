from __future__ import annotations
from typing import Tuple, List, Dict, Literal, Optional, Any, Union

from functools import partial
from typing import Tuple, List, Dict, Any
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

"""def text_to_input_ids(tokenizer:Any, text:str, model:Optional[torch.nn.Module]=None) -> torch.Tensor:
    #Encode inputs and move them to the model's device
    toks = tokenizer.encode(text, return_tensors="pt")

    if model is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        toks = toks.to(device)
    
    return toks"""

def text_to_input_ids(tokenizer:Any, text:Union[str, List[str]], model:Optional[torch.nn.Module]=None, add_special_tokens:bool=True, pad_to_max_length=False) -> torch.Tensor: # NEW
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
    
    # Replace NaNs/Infs before softmax
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    #print(layer_logits)

    # Clip logits layer-wise by z-score thresholds
    # This keeps relative variation but avoids extreme outliers ruining softmax
    """mean = np.mean(layer_logits, axis=-1, keepdims=True)
    std = np.std(layer_logits, axis=-1, keepdims=True)
    layer_logits = np.clip(layer_logits, mean - 5 * std, mean + 5 * std)"""

    # Clip extreme values (final safety net)
    #layer_logits = np.clip(layer_logits, a_min=-100, a_max=100)

    # Compute softmax for probabilities
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)

    # Clean probabilities
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    # Normalize if the sum is way off
    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        # Prevent divide-by-zero
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs
        #layer_probs += np.random.normal(0, 1e-6, size=layer_probs.shape)

    # Step 7: Argmax over logits (not probs) for prediction
    layer_preds = layer_logits.argmax(axis=-1)
    #print(layer_preds)

    return layer_preds, layer_probs


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

"""def kl_div(p, q, clip=1e-3):
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    p = torch.clamp(p, min=clip)
    q = torch.clamp(q, min=clip)
    return (p * (p / q).log()).sum(dim=-1).cpu().numpy()"""

def js_divergence(p, q, axis=-1, clip=1e-16):
    """Computes Jensen-Shannon divergence between two probability distributions."""
    p, q = clipmin(p, clip), clipmin(q, clip)
    m = (p + q) / 2
    return (kl_div(p, m, axis=axis, clip=clip) + kl_div(q, m, axis=axis, clip=clip)) / 2

def compute_entropy(probs):
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_probs, axis=-1)

def maybe_batchify(p):
    """ Normalize shape for e.g., wasserstein """
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    return p


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
    print(f"[Distances] {distances} | \n[NWD] {distances/vocab_size} |\n")
    return distances / vocab_size


def min_max_scale(arr, new_min, new_max):
    """Scales an array to a new range [new_min, new_max] for plotting and comparison of normalized values across models."""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr


# ===================== The Comparing Logit Lens ============================
def _plot_comparing_lens(
    layer_logits_1,
    layer_logits_2,
    layer_preds_1,
    layer_preds_2,
    layer_probs_1,
    layer_probs_2,
    tokenizer:Any,
    input_ids:Any,
    start_ix:int,
    layer_names_1,
    layer_names_2,
    save_fig_path,
    kl:bool=False,     
    js:bool=False,           
    wasserstein:bool=False,
    top_down:bool=False,
):
    if not kl and not js and not wasserstein:
        wasserstein = True

    end_ix = start_ix + layer_logits_2.shape[1]

    final_preds_1, final_preds_2 = layer_preds_1[-1], layer_preds_2[-1]
    aligned_preds_2 = layer_preds_2
    final_probs_1, final_probs_2 = layer_probs_1[-1], layer_probs_2[-1]

    if kl or js or wasserstein:
        clip = 1 / (10 * layer_probs_2.shape[-1])
        metric_fn = (
            lambda p1, p2: kl_div(p1, p2, clip=clip) if kl else
            js_divergence(p1, p2) if js else
            nwd(p1, p2)
        )

        to_show = np.stack([
            metric_fn(p1, p2)
            for p1, p2 in zip(layer_probs_1, layer_probs_2)
        ], axis=0)  # shape: (num_layers, batch, seq_len) if using nwd

        # Squeeze middle dim if needed (e.g., batch=1)
        if to_show.ndim == 3 and to_show.shape[1] == 1:
            to_show = to_show.squeeze(1)

        # Final shape: (num_layers, seq_len)
        to_show = min_max_scale(to_show, 0, 1 if not kl else 10)

    _num2tok = np.vectorize(
        partial(num2tok, tokenizer=tokenizer, quotemark="'"), otypes=[str]
    )

    aligned_texts = _num2tok(aligned_preds_2)
    input_tokens_str = _num2tok(input_ids[0].cpu())

    to_show = to_show[::-1]
    aligned_texts = aligned_texts[::-1]

    fig = plt.figure(figsize=(1.5 * to_show.shape[1], 0.375 * to_show.shape[0]))
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_kwargs = {"annot": aligned_texts, "fmt": ""}

    if kl:
        vmin, vmax = 0, 10
        title = "KL Divergence"
        plot_kwargs.update({
            "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r",
            "vmin": vmin,
            "vmax": vmax,
        })
    elif js or wasserstein:
        vmin, vmax = 0, 1
        title = "Jensen–Shannon Divergence" if js else "Normalized Wasserstein Distance"
        plot_kwargs.update({
            #"cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r",
            "cmap": "Blues",
            "vmin": vmin,
            "vmax": vmax,
        })

    sns.heatmap(to_show, **plot_kwargs)

    ax = plt.gca()
    if layer_names_2 is None:
        layer_names_2 = [f"Layer {n}" for n in range(to_show.shape[0])]
    ax.set_yticklabels(layer_names_2[::-1], rotation=0)

    ax_top = ax.twiny()
    padw = 0.5 / to_show.shape[1]
    ax_top.set_xticks(np.linspace(padw, 1 - padw, to_show.shape[1]))

    ax_inputs, ax_targets = ax, ax_top
    if top_down:
        ax.invert_yaxis()
        ax_inputs, ax_targets = ax_top, ax

    ax_inputs.set_xticklabels(input_tokens_str[start_ix:end_ix], rotation=0)

    starred = [
        "* " + true if pred == true else " " + true
        for pred, true in zip(
            aligned_texts[0], input_tokens_str[start_ix + 1 : end_ix + 1]
        )
    ]
    ax_targets.set_xticklabels(starred, rotation=0)
    
    #plt.title(title, fontsize=12, fontweight='bold', pad=10)

    if save_fig_path is not None:
        plt.savefig(save_fig_path)


def plot_comparing_lens(
    models:Tuple[Any,Any],
    tokenizer:Any,
    input_ids:Any,
    start_ix: int,
    end_ix: int,
    save_fig_path:Optional[str]|None=None,
    kl:bool=False,
    js:bool=False,            
    wasserstein:bool=False,
    block_step:int=1,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:list = ['norm', 'lm_head'],
    top_down:bool=False,
    verbose:bool=False,
    topk:Optional[int|None]=None
):

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

    layer_names_1, layer_names_2 = multiple_layer_names(model_1=model_1, model_2=model_2)

    make_lens_hooks(
        model_1, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_1,
        decoder_layer_names=decoder_layer_names,
        verbose=verbose
    )
                
    make_lens_hooks(
        model_2, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_2,
        decoder_layer_names=decoder_layer_names,
        verbose=verbose
    )

    input_ids = text_to_input_ids(tokenizer, input_ids, model_2)
    input_ids_1 = input_ids.to(next(model_1.parameters()).device)
    input_ids_2 = input_ids
    
    if topk:
        layer_logits_1, layer_names_1 = collect_logits(
            model_1, input_ids_1, layer_names=layer_names_1, decoder_layer_names=decoder_layer_names,
        )

        layer_logits_2, layer_names_2 = collect_logits(
            model_2, input_ids_2, layer_names=layer_names_2, decoder_layer_names=decoder_layer_names,
        )
            
        layer_preds_1, layer_probs_1 = postprocess_logits(layer_logits_1)
        layer_preds_2, layer_probs_2 = postprocess_logits(layer_logits_2)

    else:
        layer_logits_1, layer_names_1 = collect_logits(
            model_1, input_ids_1, layer_names=layer_names_1, decoder_layer_names=decoder_layer_names,
        )

        layer_logits_2, layer_names_2 = collect_logits(
            model_2, input_ids_2, layer_names=layer_names_2, decoder_layer_names=decoder_layer_names,
        )
            
        layer_preds_1, layer_probs_1 = postprocess_logits(layer_logits_1)
        layer_preds_2, layer_probs_2 = postprocess_logits(layer_logits_2)


    _plot_comparing_lens(
            layer_logits_1=layer_logits_1,
            layer_logits_2=layer_logits_2,
            layer_preds_1=layer_preds_1,
            layer_preds_2=layer_preds_2,
            layer_probs_1=layer_probs_1,
            layer_probs_2=layer_probs_2,
            tokenizer=tokenizer,
            input_ids=input_ids,
            start_ix=start_ix,
            save_fig_path=save_fig_path,
            kl=kl,
            js=js,
            wasserstein=wasserstein,
            layer_names_1=layer_names_1,
            layer_names_2=layer_names_2,
            top_down=top_down,
        )