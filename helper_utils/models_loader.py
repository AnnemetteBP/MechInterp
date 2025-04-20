from __future__ import annotations
from typing import Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)



def load_fp_auto_tokenizer(KEY:str, hs:bool, r_dict:bool, precision:torch.dtype, mem:bool, dmap:str, sf:bool, trust_remote:bool, configs:bool) -> Tuple[AutoModelForCausalLM, AutoTokenizer]: 
    """ Load auto model and tokenizer
        KEY: model transformer endpoint
        hs: hidden states = True - important for snapper snalysis
        precision: torch_dtype=torch.float32 or torch_dtype=torch.float16 or torch.bfloat16
        mem: cpu memory usage = True - important for memory efficiency
        dmap: e.g., 'auto' """
    
    if configs is True:
        config = AutoConfig.from_pretrained(KEY)
        config.output_hidden_states = hs
        config.low_cpu_mem_usage = mem

        model = AutoModelForCausalLM.from_pretrained(
            KEY,
            config=config
        )
        
        tokenizer = AutoTokenizer.from_pretrained(KEY)

        return model, tokenizer
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            KEY,
            return_dict=r_dict,
            output_hidden_states=hs,
            torch_dtype=precision,
            low_cpu_mem_usage=mem,
            device_map=dmap,
            use_safetensors=sf,
            #trust_remote_code=trust_remote 
        )
        
        tokenizer = AutoTokenizer.from_pretrained(KEY)

        return model, tokenizer


def load_fp_auto(KEY:str, hs:bool, r_dict:bool, precision:torch.dtype, mem:bool, dmap:str, sf:bool, trust_remote:bool, configs:bool) -> AutoModelForCausalLM:
    """ Load auto model
        KEY: model transformer endpoint
        hs: hidden states = True - important for snapper snalysis
        precision: torch_dtype=torch.float32 or torch_dtype=torch.float16 or torch.bfloat16
        mem: cpu memory usage = True - important for memory efficiency
        dmap: e.g., 'auto' """

    if configs is True:
        config = AutoConfig.from_pretrained(KEY)
        config.output_hidden_states = hs
        config.low_cpu_mem_usage = mem

        model = AutoModelForCausalLM.from_pretrained(
            KEY,
            config=config
        )

        return model
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            KEY,
            return_dict=r_dict,
            output_hidden_states=hs,
            torch_dtype=precision,
            low_cpu_mem_usage=mem,
            device_map=dmap,
            use_safetensors=sf,
            #trust_remote_code=trust_remote 
        )
        
        return model


def load_8bit_auto(KEY:str, hs:bool, r_dict:bool, precision:torch.dtype, bnb_precision:torch.dtype, dmap:str, sf:bool, trust_remote:bool) -> AutoModelForCausalLM:
    """ Load auto model in 8-bit precision
        KEY: model transformer endpoint
        hs: hidden states = True - important for snapper snalysis
        dmap: e.g., 'auto' """

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Native 8-bit quantization
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_has_fp16_weight=False,
        bnb_8bit_compute_dtype=bnb_precision,  # Ensure compute dtype is float16
        bnb_8bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        KEY,
        return_dict=r_dict,
        output_hidden_states=hs,
        torch_dtype=precision,
        quantization_config=bnb_config,
        device_map=dmap,
        use_safetensors=sf,
        #trust_remote_code=trust_remote 
    )
    
    return model


def load_4bit_auto(KEY:str, hs:bool, r_dict:bool, precision:torch.dtype, dmap:str, sf:bool, trust_remote:bool) -> AutoModelForCausalLM:
    """ Load auto model in 4-bit precision
        KEY: model transformer endpoint
        hs: hidden states = True - important for snapper snalysis
        precision: torch_dtype=torch.float32 or torch_dtype=torch.float16 or torch.bfloat16
        dmap: e.g., 'auto' """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=precision,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', # e.g., int4 for further efficiency
    )

    model = AutoModelForCausalLM.from_pretrained(
        KEY,
        return_dict=r_dict,
        output_hidden_states=hs,
        quantization_config=bnb_config,
        device_map=dmap,
        use_safetensors=sf,
        #trust_remote_code=trust_remote 
    )
    
    return model