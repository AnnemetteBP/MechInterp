from __future__ import annotations
from typing import Tuple, List, Dict, Literal, Optional, Any

from functools import partial
from typing import Tuple, List, Dict, Union, Any
import torch
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa

from ..util.python_utils import make_print_if_verbose
from .activation_lens_hooks import make_lens_hooks
from ..logit_lens.layer_names import make_layer_names


"""def text_to_input_ids(tokenizer: Any, text: str) -> torch.Tensor:
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

# ========== Collect Logits and Post-process ========== 
def collect_logits(model, tokenizer, input_ids, layer_names, decoder_layer_names=None):
    """
    Collect logits from the specified layers of the model.

    Args:
        model: The model to run the forward pass.
        tokenizer: Tokenizer used for tokenizing inputs.
        input_ids: The tokenized input IDs for the model.
        layer_names: List of layer names for which we want the logits.
        decoder_layer_names: Optional names for decoder layers (if using decoder).

    Returns:
        layer_logits (np.array): Logits collected from the specified layers.
        outputs (dict): A dictionary of all outputs, including layer logits.
    """
    model._last_resid = None
    outputs = {}

    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach().cpu()

        return fn

    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    with torch.no_grad():
        _ = model(input_ids) 

    for hook in hooks:
        hook.remove()

    layer_logits = np.concatenate(
        [outputs[name].numpy() for name in layer_names if name in outputs],
        axis=0,
    )

    if len(layer_logits) == 0:
        raise ValueError("No logits were collected. Please check the model or layer names.")

    print(f"Collected logits shape: {[logits.shape for logits in layer_logits]}") 

    return layer_logits, outputs

def postprocess_logits(layer_logits: Any, normalize_probs: bool = False):
    if len(layer_logits) == 0 or layer_logits[0].shape[0] == 0:
        raise ValueError("Layer logits are empty or have zero size. Check the model output.")

    print(f"Layer logits shape before softmax: {layer_logits.shape}") 
    
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)

    if np.any(np.isnan(layer_probs)) or np.any(np.isinf(layer_probs)):
        raise ValueError("Softmax resulted in invalid values (NaN or Inf). Please check the logits.")

    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs

    layer_preds = layer_logits.argmax(axis=-1)
    return layer_preds, layer_probs

# ===================== Plot Helpers =====================
def get_value_at_preds(tensor, preds):
    return torch.gather(tensor, dim=2, index=preds.unsqueeze(-1)).squeeze(-1).cpu().numpy()

def min_max_scale(x, min_val=0.0, max_val=1.0):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x * (max_val - min_val) + min_val

def compute_entropy(tensor):
    probs = torch.softmax(tensor, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1).cpu().numpy()

def kl_div(p, q, clip=1e-3):
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    p = torch.clamp(p, min=clip)
    q = torch.clamp(q, min=clip)
    return (p * (p / q).log()).sum(dim=-1).cpu().numpy()

def num2tok(i, tokenizer, quotemark="'"):
    try:
        tok = tokenizer.decode([i])
        if tok.strip() == "":
            return "␣"
        if tok in [quotemark, "``", "''"]:
            return quotemark
        return tok
    except Exception:
        return "<?>"

def min_max_scale(arr, new_min, new_max):
    """Scales an array to a new range [new_min, new_max] for plotting and comparison of normalized values across models."""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr

# ===================== Activations =====================
def collect_activations(model, input_ids, layer_names):
    """
    Runs a forward pass and collects activations for the specified layers.
    """
    # Reset last_resid if present
    model._last_resid = 0

    with torch.no_grad():
        if input_ids.device != next(model.parameters()).device:
            input_ids = input_ids.to(next(model.parameters()).device)
        model(input_ids)

    activations = {name: model._activations[name].cpu().numpy() for name in layer_names}
    return activations, layer_names

def compute_activation_metrics(activations, metric="norm", normalize=True, clip_percentile=99.9):
    def safe_entropy(x):
        # Softmax over last dim to turn activations into pseudo-probs
        probs = np.exp(x - np.max(x, axis=-1, keepdims=True))
        #temperature = 1.0  # Lower values (e.g., 0.5) make distributions sharper
        #probs = np.exp(x / temperature - np.max(x / temperature, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        return -np.sum(probs * np.log(probs + 1e-9), axis=-1)

    metric_fn = {
        "norm": lambda x: np.linalg.norm(x, axis=-1),
        "var": lambda x: np.var(x, axis=-1),
        "entropy": safe_entropy
    }.get(metric)

    if metric_fn is None:
        raise ValueError(f"Unsupported metric: {metric}")

    result = {}
    for name, act in activations.items():
        # normalization to avoid overflow in float16
        if normalize:
            act = act.astype(np.float32)  # temp upcast for safety
            scale = np.percentile(np.abs(act), clip_percentile) + 1e-6
            act = np.clip(act, -scale, scale)
            act = act / scale

        result[name] = metric_fn(act).squeeze()

    return result


def _plot_comparing_act_lens(
    acts_dict_true:Dict[str, np.ndarray],
    acts_dict_compare:Dict[str, np.ndarray],
    tokenizer:Any,
    input_ids:str|Any,
    start_ix:int,
    end_ix:int,
    save_fig_path:str|None=None,
    metric_name:str='abs',
    metric:str='norm',
    top_down:bool=False
):
    layer_names = list(acts_dict_true.keys())
    num_layers = len(layer_names)

    # Stack values into arrays of shape [layers, tokens]
    vals_true = np.stack(list(acts_dict_true.values()))[:, start_ix:end_ix]  # [layers, tokens]
    vals_compare = np.stack(list(acts_dict_compare.values()))[:, start_ix:end_ix]  # [layers, tokens]

    # Transpose for easier token-wise comparison: [tokens, layers]
    vals_true = vals_true.T
    vals_compare = vals_compare.T

    # Apply comparison metric
    if metric_name == 'abs':
        diff_vals = np.abs(vals_true - vals_compare)
    elif metric_name == 'l2':
        diff_vals = np.linalg.norm(vals_true - vals_compare, axis=-1, keepdims=True)
        diff_vals = np.repeat(diff_vals, vals_true.shape[1], axis=1)  # replicate to match heatmap shape
    elif metric_name == 'cosine':
        numerator = np.sum(vals_true * vals_compare, axis=-1)
        denominator = (
            np.linalg.norm(vals_true, axis=-1) * np.linalg.norm(vals_compare, axis=-1) + 1e-9
        )
        cosine_sim = numerator / denominator
        diff_vals = 1 - cosine_sim[:, None]  # shape [tokens, 1]
        diff_vals = np.repeat(diff_vals, vals_true.shape[1], axis=1)  # replicate
    else:
        raise ValueError(f"Unsupported metric_name: {metric_name}")

    num_tokens, num_layers = diff_vals.shape

    # Token labels
    input_ids_slice = input_ids[0, start_ix:end_ix] if isinstance(input_ids, torch.Tensor) else input_ids[start_ix:end_ix]
    token_labels = [tokenizer.decode([i]).replace("\n", "⏎").strip() or "␣" for i in input_ids_slice]

    print(f"Shape of diff_vals: {diff_vals.shape}")

    # === Plotting
    fig_width = max(6, 0.4 * num_layers)
    fig_height = max(4, 0.8 * num_tokens)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    
    mask = np.zeros_like(diff_vals, dtype=bool)
    print(f"Shape of mask: {mask.shape}")

    sns.heatmap(
        diff_vals[::-1] if top_down else diff_vals,
        cmap='coolwarm',
        ax=ax,
        vmin=np.min(diff_vals),
        vmax=np.max(diff_vals),
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor='white',
        mask=mask
    )

    ax.set_xticks(np.arange(num_layers) + 0.5)
    ax.set_xticklabels(layer_names[::-1] if top_down else layer_names, rotation=90)

    ax.set_yticks(np.arange(num_tokens) + 0.5)
    ax.set_yticklabels(token_labels[::-1] if top_down else token_labels)

    plt.xlabel("Hooked layers / modules")
    plt.ylabel("Tokens")
    plt.title(f"Activation Comparison ({metric} & {metric_name})")

    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")

    plt.show()

# ===================== Main Function to Compare Models =====================
def plot_comparing_act_lens(
    models:Tuple[Any,Any],
    tokenizer:Any,
    input_ids:str|Any,
    start_ix:int,
    end_ix:int,
    metric:str='norm',  # Can be 'norm', 'var', or 'entropy'
    metric_name='l2',  # Can be 'abs', 'l2', 'cosine'
    save_fig_path:str|None=None,
    verbose:bool=False,
    block_step=1,
    include_input=True,
    force_include_output=True,
    include_subblocks=True,
    decoder_layer_names=['norm'],
):
    """
    Draws "comparing activation lens" plots using first model in list as true distribution, and generalizations thereof.
    Here `model` should be a tuple of two `transformers.PreTrainedModel` with an LLaMA / OLMo architecture
        - The first model in tuple `model[0]` is used as true distribution for `model[1]`.

    Note that using `start_ix` and `end_ix` is not equivalent to passed an `input_ids` sliced like `input_ids[start_ix:end_ix]`. 
    The LM will see the entire input you pass in as `input_ids`, no matter how you set `start_ix` and `end_ix`.  These "ix" arguments only control what is _displayed_.

    The metric arguments `norm`, `var` and `entropy` control the type of plot.  The options are:

        - abs:
            - cell color: absolute distance between base model and comparison model for: `norm`, `var` and `entropy` 
            - cell text:  same as cell color and annotated w. scalar value

        - l2:
            - cell color: euclidian (l2) distance between base model and comparison model for: `norm`, `var` and `entropy` 
            - cell text:  same as cell color and annotated w. scalar value

        - cosine:
            - cell color: cosine similarity between base model and comparison model for: `norm`, `var` and `entropy` 
            - cell text:  same as cell color and annotated w. scalar value

    `include_subblocks` and `decoder_layer_names` allow the creation of plots that go beyond what was done
    in the original blog post.  See below for details

    Arguments:
        metric:
            norm:
                collect activation `norm` 
            var:
                collect activation `var`
            cosine:
                collect activation `entropy` over pseudo-probs
        entropy:
            abs:
                draw a "abs" comparison plot of either: `norm`, `var` or `entropy` 
            l2:
                draw a "l2" comparison plot of either: `norm`, `var` or `entropy` 
            cosine:
                draw a "cosine" comparison plot of either: `norm`, `var` or `entropy` 
        save_fig_path:
            save a plot to path if not None: str | None
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
    """
       
    base_model, comparison_model = models  # True distribution model (FP) and Comparison model (Quantized)

    layer_names_base = make_layer_names(
        base_model,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names,
    )
    layer_names_comparison = make_layer_names(
        comparison_model,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names,
    )

    make_lens_hooks(
        base_model,
        start_ix=start_ix,
        end_ix=end_ix,
        layer_names=layer_names_base,
        decoder_layer_names=decoder_layer_names,
        verbose=verbose,
        record_activations=True,
    )
    make_lens_hooks(
        comparison_model,
        start_ix=start_ix,
        end_ix=end_ix,
        layer_names=layer_names_base,
        decoder_layer_names=decoder_layer_names,
        verbose=verbose,
        record_activations=True,
    )

    if isinstance(input_ids, str):
        input_ids = text_to_input_ids(tokenizer, input_ids, comparison_model)
    
    activations_true, _ = collect_activations(base_model, input_ids, layer_names_base)
    activations_compare, _ = collect_activations(comparison_model, input_ids, layer_names_comparison)
    
    metric_vals_true = compute_activation_metrics(activations_true, metric)
    metric_vals_compare = compute_activation_metrics(activations_compare, metric)

    _plot_comparing_act_lens(
        acts_dict_true=metric_vals_true,
        acts_dict_compare=metric_vals_compare,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        end_ix=end_ix,
        save_fig_path=save_fig_path,
        metric_name=metric_name,
        metric=metric,
        top_down=False
    )