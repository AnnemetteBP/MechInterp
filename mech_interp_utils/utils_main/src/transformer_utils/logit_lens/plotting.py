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
from ..lm_tasks import lm_task_manager


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


def postprocess_logits(layer_logits):
    layer_preds = layer_logits.argmax(axis=-1)

    layer_probs = scipy.special.softmax(layer_logits, axis=-1)

    return layer_preds, layer_probs


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


def compute_entropy(probs, axis=-1, clip=1e-16):
    """Computes entropy over token probabilities."""
    probs = clipmin(probs, clip)
    return entropy(probs, axis=axis)


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


def normalized_wasserstein_distance(p, q, axis=-1, clip=1e-16):
    """Computes normalized Wasserstein distance between two probability distributions."""
    p, q = np.clip(p, clip, 1.0), np.clip(q, clip, 1.0)  # Avoid negative values
    vocab_size = p.shape[-1]
    indices = np.arange(vocab_size)  # Vocab indices

    # Ensure p and q are properly shaped
    if p.ndim == 2:  # If `p` lacks a batch dimension
        p = np.expand_dims(p, axis=0)  # Shape (1, 15, 50304)
    
    if p.shape[0] == 1 and q.shape[0] > 1:  # If `p` is singleton batch
        p = np.repeat(p, q.shape[0], axis=0)  # Expand `p` to match `q`
    
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch after fixing: p {p.shape}, q {q.shape}")

    # Compute Wasserstein distances batch-wise
    distances = np.array([
        wasserstein_distance(indices, indices, p[b, i], q[b, i])
        for b in range(p.shape[0])  # Iterate over batch
        for i in range(p.shape[1])  # Iterate over sequence
    ])

    distances = distances.reshape(p.shape[0], p.shape[1])  # Ensure 2D output
    return distances / vocab_size  # Normalize

def nwd(p, q, axis=-1, clip=1e-16):
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
    return distances / vocab_size


def min_max_scale(arr, new_min, new_max):
    """Scales an array to a new range [new_min, new_max] for plotting and comparison of normalized values across models."""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr


""" LOGIT LENS FOR MODEL CAMPARISON ANALYSIS """
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
    probs:bool=False,
    kl:bool=False,     
    js:bool=False,           
    wasserstein:bool=False,
    top_down:bool=False,
):
    end_ix = start_ix + layer_logits_2.shape[1]

    final_preds_1, final_preds_2 = layer_preds_1[-1], layer_preds_2[-1]

    aligned_preds_1, aligned_preds_2 = layer_preds_1, layer_preds_2

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
    else:
        numeric_input = layer_probs_2 if probs else layer_logits_2
        to_show = get_value_at_preds(numeric_input, final_preds_2)
        to_show = min_max_scale(to_show, 0, 1)

    _num2tok = np.vectorize(
        partial(num2tok, tokenizer=tokenizer, quotemark="'"), otypes=[str]
    )

    aligned_texts = _num2tok(aligned_preds_2)

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
    elif js or wasserstein:
        vmin, vmax = 0, 1

        if js:
            title = "Jensen–Shannon Divergence"
        else:
            title = "Normalized Wasserstein Distance"

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r",
                "vmin": vmin,
                "vmax": vmax,
                #"annot": True,
                #"fmt": ".1f",
            }
        )
    elif probs:
        title = "Probability"

        plot_kwargs.update({"cmap": "Blues_r", "vmin": 0, "vmax": 1})
    else:
        title = "Logits"
        #vmin = np.percentile(to_show.reshape(-1), 5)
        #vmax = np.percentile(to_show.reshape(-1), 95)
        vmin = 0
        vmax = 1

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

    if layer_names_2 is None:
        layer_names_2 = ["Layer {}".format(n) for n in range(to_show.shape[0])]
    ylabels = layer_names_2[::-1]
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


""" LOGIT LENS FOR SINGLE MODEL ANALYSIS """
def _plot_logit_lens(
    layer_logits,
    layer_preds,
    layer_probs,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    probs=False,
    ranks=False,
    kl=False,
    entropy=False,      
    js=False,           
    wasserstein=False,  
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
    elif js:
        to_show = js_divergence(final_probs, layer_probs)
        to_show = min_max_scale(to_show, 0, 1)
    elif wasserstein:
        to_show = normalized_wasserstein_distance(final_probs, layer_probs)
        to_show = min_max_scale(to_show, 0, 1)
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
        #vmin, vmax = None, None
        vmin, vmax = 0, 10
        title = "Entropy"

        plot_kwargs.update(
            {
                "cmap": "coolwarm",
                "vmin": vmin,
                "vmax": vmax,
            }
        )
    elif js or wasserstein:
        vmin, vmax = 0, 1

        if js:
            title = "Jensen–Shannon Divergence"
        else:
            title = "Normalized Wasserstein Distance"

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r",
                "vmin": vmin,
                "vmax": vmax,
                #"annot": True,
                #"fmt": ".1f",
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

        plot_kwargs.update({"cmap": "Blues_r", "vmin": 0, "vmax": 1})
    else:
        title = "Logits"
        #vmin = np.percentile(to_show.reshape(-1), 5)
        #vmax = np.percentile(to_show.reshape(-1), 95)
        vmin = 0
        vmax = 1

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

def plot_logit_lens(
    model:Any|List[Any],
    tokenizer:Any,
    input_ids:Any,
    start_ix: int,
    end_ix: int,
    lm_task:str,
    probs:bool=False,
    ranks:bool=False,
    kl:bool=False,
    entropy:bool=False,  
    js:bool=False,            
    wasserstein:bool=False,  
    block_step:int=1,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:list = ['norm', 'lm_head'],
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

    `model` should be a `transformers.PreTrainedModel` with an `lm_head`, e.g. `GPTNeoForCausalLM`.

    Note that using `start_ix` and `end_ix` is not equivalent to passed an `input_ids` sliced like `input_ids[start_ix:end_ix]`.  The LM will see the entire input you pass in as `input_ids`, no matter how you set `start_ix` and `end_ix`.  These "ix" arguments only control what is _displayed_.

    The boolean arguments `probs`, `ranks` and `kl` control the type of plot.  The options are:

        - Logits (the default plot type, if `probs`, `ranks` and `kl` are all False):
            - cell color: logit assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer

        - Probabilities:
            - cell color: probability assigned by each layer to the final layer's top-1 token prediction
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
    """layer_names = make_layer_names(
        model,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names
    )

    make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names,
                    decoder_layer_names=decoder_layer_names,
                    verbose=verbose)"""
    
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
    

    match lm_task:
        case 'cloze':
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
            
            layer_logits, layer_names = lm_task_manager.cloze_task(
                model=model, tokenizer=tokenizer, context=input_ids, top_k=1, max_new_tokens=50, layer_names=layer_names, decoder_layer_names=decoder_layer_names
                )

            layer_preds, layer_probs = postprocess_logits(layer_logits=layer_logits)

        case 'template':
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
    
            layer_logits, layer_names = lm_task_manager.template_to_input_ids(
                model=model, tokenizer=tokenizer, text=input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names
            )

            layer_preds, layer_probs = postprocess_logits(layer_logits=layer_logits)

        case 'compare':
            if len(model) != 2:
                raise ValueError("Error: 'compare' requires exactly two models!")
            else:
                model_1, model_2 = model[0], model[1]

            layer_names_1, layer_names_2 = multiple_layer_names(model_1=model_1, model_2=model_2)

            make_lens_hooks(model_1, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_1,
                    decoder_layer_names=decoder_layer_names,
                    verbose=verbose)
                    
            make_lens_hooks(model_2, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_2,
                    decoder_layer_names=decoder_layer_names,
                    verbose=verbose)
                    
            input_ids = text_to_input_ids(tokenizer, input_ids)

            layer_logits_1, layer_names_1 = collect_logits(
                model_1, input_ids, layer_names=layer_names_1, decoder_layer_names=decoder_layer_names,
                )
                    
            layer_logits_2, layer_names_2 = collect_logits(
                model_2, input_ids, layer_names=layer_names_2, decoder_layer_names=decoder_layer_names,
                )
            
            layer_preds_1, layer_probs_1 = postprocess_logits(layer_logits_1)
            layer_preds_2, layer_probs_2 = postprocess_logits(layer_logits_2)

        case 'base':
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
    
            input_ids = text_to_input_ids(tokenizer, input_ids)

            layer_logits, layer_names = collect_logits(
                model, input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names,
                )
            
            layer_preds, layer_probs = postprocess_logits(layer_logits)

        case _:
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
    
            input_ids = text_to_input_ids(tokenizer, input_ids)

            layer_logits, layer_names = collect_logits(
                model, input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names,
                )
            
            layer_preds, layer_probs = postprocess_logits(layer_logits)

    """layer_logits, layer_names = collect_logits(
        model, input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names,
    )

    layer_preds, layer_probs = postprocess_logits(layer_logits)"""

    if lm_task == 'compare':
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
            probs=probs,
            kl=kl,
            js=js,
            wasserstein=wasserstein,
            layer_names_1=layer_names_1,
            layer_names_2=layer_names_2,
            top_down=top_down,
        )
    
    else:
        _plot_logit_lens(
            layer_logits=layer_logits,
            layer_preds=layer_preds,
            layer_probs=layer_probs,
            tokenizer=tokenizer,
            input_ids=input_ids,
            start_ix=start_ix,
            probs=probs,
            ranks=ranks,
            kl=kl,
            entropy=entropy,
            js=js,
            wasserstein=wasserstein,
            layer_names=layer_names,
            top_down=top_down,
        )
