from __future__ import annotations
from typing import Tuple, List, Dict, Literal, Optional, Any

from typing import Tuple, List, Dict, Union, Any
import torch
import numpy as np
from scipy.stats import entropy
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
# ===================== Plot Helpers =====================
def get_value_at_preds(tensor, preds):
    return torch.gather(tensor, dim=2, index=preds.unsqueeze(-1)).squeeze(-1).cpu().numpy()

def min_max_scale(x, min_val=0.0, max_val=1.0):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x * (max_val - min_val) + min_val
    
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

"""def compute_activation_metrics(activations, metric="norm"):
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

    return {
        name: metric_fn(act).squeeze()
        for name, act in activations.items()
    }"""

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


# ===================== Plot Activation Lens =====================
def _plot_activation_lens(
    activations_dict:Dict,
    tokenizer:Any,
    input_ids:str|Any,
    start_ix:int,
    end_ix:int,
    save_fig_path:str|None=None,
    metric_name:str='norm',
    top_down:bool=False
):
    layer_names = list(activations_dict.keys())
    num_layers = len(layer_names)

    # Stack and squeeze activations
    activations_array = np.stack(list(activations_dict.values()))  # shape: [num_layers, 1, seq_len] or [num_layers, seq_len]
    if activations_array.ndim == 3 and activations_array.shape[1] == 1:
        activations_array = activations_array[:, 0, :]  # Remove batch dim

    metric_vals = activations_array[:, start_ix:end_ix].T  # shape: [seq_len, num_layers]

    token_labels = []
    input_ids_slice = input_ids[0, start_ix:end_ix] if isinstance(input_ids, torch.Tensor) else input_ids[start_ix:end_ix]
    for i in input_ids_slice:
        tok = tokenizer.decode([i])
        token_labels.append(tok.replace("\n", "⏎").strip() or "␣")
    
    fig_width = max(6, 0.4 * num_layers)
    #fig_height = max(3, 0.6 * (end_ix - start_ix))
    fig_height = max(4, 0.8 * (end_ix - start_ix)) 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.heatmap(
        metric_vals[::-1] if top_down else metric_vals,
        cmap='coolwarm',
        ax=ax,
        vmin=np.min(metric_vals),
        vmax=np.max(metric_vals),
        annot=True,
        fmt=".1f",
        linewidths=0.5,  
        linecolor='white'  
    )

    ax.set_xticks(np.arange(num_layers) + 0.5)
    ax.set_xticklabels(layer_names[::-1] if top_down else layer_names, rotation=90)

    ax.set_yticks(np.arange(end_ix - start_ix) + 0.5)
    ax.set_yticklabels(token_labels[::-1] if top_down else token_labels)

    plt.xlabel("Hooked layers / modules")
    plt.ylabel("Tokens")
    plt.title(f"Activation ({metric_name})")

    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_activation_lens(
    model:Any,
    tokenizer:Any,
    input_ids:str|Any,
    start_ix:int,
    end_ix:int,
    metric:str='norm',  # or "var", "entropy"
    save_fig_path:str|None=None,
    block_step:int=1,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=True,
    decoder_layer_names:List[str]=['norm'],
    top_down:bool=False,
    verbose:bool=False,
):
    """
    Draws "activation lens" plots and generalizations thereof.
    Here `model` should be a `transformers.PreTrainedModel` with an LLaMA / OLMo architecture

    Note that using `start_ix` and `end_ix` is not equivalent to passed an `input_ids` sliced like `input_ids[start_ix:end_ix]`. 
    The LM will see the entire input you pass in as `input_ids`, no matter how you set `start_ix` and `end_ix`.  These "ix" arguments only control what is _displayed_.

    The metric arguments `norm`, `var` and `entropy` control the type of plot.  The options are:

        - norm:
            - cell color: activation `norm`
            - cell text:  same as cell color and annotated w. scalar value

        - var:
            - cell color: activation `var`
            - cell text:  same as cell color and annotated w. scalar value

        - cosine:
            - cell color: activation `entropy` over pseudo-probs
            - cell text:  same as cell color and annotated w. scalar value

    `include_subblocks` and `decoder_layer_names` allow the creation of plots that go beyond what was done
    in the original blog post.  See below for details

    Arguments:
        metric:
            norm:
                draw a "norm" plot of activations
            var:
                draw a "var" plot of activations
            entropy:
                draw a "entropy" plot of activation pseudo-probs
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
    
    layer_names = make_layer_names(
        model,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names,
    )

    make_lens_hooks(
        model,
        start_ix=start_ix,
        end_ix=end_ix,
        layer_names=layer_names,
        decoder_layer_names=decoder_layer_names,
        verbose=verbose,
        record_activations=True,
    )

    if isinstance(input_ids, str):
        input_ids = text_to_input_ids(tokenizer, input_ids, model)

    activations, layer_names = collect_activations(model, input_ids, layer_names)
    metric_vals = compute_activation_metrics(activations, metric)

    _plot_activation_lens(
        activations_dict=metric_vals,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        end_ix=end_ix,
        save_fig_path=save_fig_path,
        metric_name=metric,
        top_down=top_down
    )