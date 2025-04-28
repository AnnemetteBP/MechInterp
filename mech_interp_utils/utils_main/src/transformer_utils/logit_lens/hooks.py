import torch
import torch.nn as nn

from ..util.python_utils import make_print_if_verbose
from ..util.module_utils import get_child_module_by_names


_RESID_SUFFIXES = {".self_attn", ".mlp"}

def unquantize_tensor(tensor):
    """
    Unquantize a tensor if it is quantized, i.e., if it has a `dequantize` method.
    """
    if hasattr(tensor, "dequantize"):
        return tensor.dequantize()
    return tensor.float()


def blocks_input_locator(model: nn.Module):
    """
    Identifies the input to the transformer blocks.
    """
    return lambda: model.embed_tokens  # OLMo input embeddings

def final_layernorm_locator(model: nn.Module):
    """
    Identifies the final normalization layer.
    """
    if hasattr(model.base_model, "norm"):
        return lambda: model.base_model.norm
    else:
        raise ValueError("Could not identify final layer norm")

def _locate_special_modules(model):
    if not hasattr(model, "_blocks_input_getter"):
        model._blocks_input_getter = blocks_input_locator(model)

    if not hasattr(model, "_ln_f_getter"):
        model._ln_f_getter = final_layernorm_locator(model)

def _get_layer(model, name):
    if name == "input":
        return model._blocks_input_getter()
    if name == "norm":
        return model._ln_f_getter()

    model_with_module = model if name == "lm_head" else model.base_model
    return get_child_module_by_names(model_with_module, name.split("."))

def _sqz(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x[0]
    except:
        return x

def _get_layer_and_compose_with_ln(model, name):
    if name.endswith('.self_attn'):
        lname = name[:-len('.self_attn')] + '.input_layernorm'
        ln = _get_layer(model, lname)
    elif name.endswith('.mlp'):
        lname = name[:-len('.mlp')] + '.post_attention_layernorm'
        ln = _get_layer(model, lname)
    else:
        ln = lambda x: x
    return lambda x: _get_layer(model, name)(ln(x))

def make_decoder(model, decoder_layer_names=['norm', 'lm_head']):  # Adjusted for OLMo
    _locate_special_modules(model)

    decoder_layers = [_get_layer_and_compose_with_ln(model, name) for name in decoder_layer_names]

    def _decoder(x):
        for name, layer in zip(decoder_layer_names, decoder_layers):
            layer_out = _sqz(layer(_sqz(x)))

            is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])
            if is_resid:
                x = x + layer_out
            else:
                x = layer_out
        return x
    return _decoder

def make_lens_hooks(
    model,
    layer_names: list,
    decoder_layer_names: list = ['norm', 'lm_head'],  # Adjusted for OLMo
    verbose=True,
    start_ix=None,
    end_ix=None,
):
    vprint = make_print_if_verbose(verbose)

    clear_lens_hooks(model)

    def _opt_slice(x, start_ix, end_ix):
        if start_ix is None:
            start_ix = 0
        if end_ix is None:
            end_ix = x.shape[1]
        return x[:, start_ix:end_ix, :]

    _locate_special_modules(model)

    for attr in ["_layer_logits", "_layer_logits_handles"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    model._ordered_layer_names = layer_names
    model._lens_decoder = make_decoder(model, decoder_layer_names)

    def _make_record_logits_hook(name):
        model._layer_logits[name] = None

        is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])

        def _record_logits_hook(module, input, output) -> None:
            del model._layer_logits[name]
            ln_f = model._ln_f_getter()

            if is_resid:
                decoder_in = model._last_resid + _sqz(output)
            else:
                decoder_in = _sqz(output)

            decoder_out = model._lens_decoder(decoder_in)
            decoder_out = _opt_slice(decoder_out, start_ix, end_ix)

            model._layer_logits[name] = decoder_out.cpu().numpy()
            model._last_resid = decoder_in

        return _record_logits_hook

    def _hook_already_there(name):
        handle = model._layer_logits_handles.get(name)
        if not handle:
            return False
        layer = _get_layer(model, name)
        return handle.id in layer._forward_hooks

    for name in layer_names:
        if _hook_already_there(name):
            vprint(f"Skipping layer {name}, hook already exists")
            continue
        layer = _get_layer(model, name)
        handle = layer.register_forward_hook(_make_record_logits_hook(name))
        model._layer_logits_handles[name] = handle

def clear_lens_hooks(model):
    if hasattr(model, "_layer_logits_handles"):
        for k, v in model._layer_logits_handles.items():
            v.remove()

        ks = list(model._layer_logits_handles.keys())
        for k in ks:
            del model._layer_logits_handles[k]
