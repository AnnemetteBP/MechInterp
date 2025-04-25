from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os
import time
import psutil
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import scipy.special
import colorcet  # noqa


""" ******************** FOR BASE TEXT INPUT PROMPT ******************** """

def text_to_input_ids(tokenizer:Any, text:str) -> torch.Tensor:
    """ Encode inputs """

    toks = tokenizer.encode(text, return_tensors='pt')
    return torch.as_tensor(toks).view(1, -1).cpu()


def collect_logits(model, input_ids, layer_names, decoder_layer_names):
    """ Collect logits for Logit Lens """
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
    """ Prepare layer logits """

    layer_preds = layer_logits.argmax(axis=-1)

    layer_probs = scipy.special.softmax(layer_logits, axis=-1)

    return layer_preds, layer_probs


def template_to_input_ids(model:Any, tokenizer:Any, text:str, layer_names, decoder_layer_names) -> Tuple[Any,Any]:
    """ Encode inputs using e.g., Deep Hermes (long chain) chat format from:
        https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-3B-Preview / https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview """

    messages = [
        {"role": "user", "content": text}  # Wrap input in chat format
    ]

    # Tokenize using chat template
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
    )
    
    toks = torch.as_tensor(input_ids).view(1, -1).cpu()
    layer_logits, layer_names = collect_logits(model=model, input_ids=toks, layer_names=layer_names, decoder_layer_names=decoder_layer_names)

    return layer_logits, layer_names



""" ******************** LM TASKS ******************** """

def cloze_task(model:Any, tokenizer:Any, context:str, top_k:int, max_new_tokens:int, layer_names:Any, decoder_layer_names:Any) -> Tuple[np.ndarray[Any], Any]:
    """ Simulates a fill-in-the-blank task for decoder-only models like OLMo by using auto-regressive text generation. """

    masked_text = context.replace("____", tokenizer.mask_token) if tokenizer.mask_token else context
    input_ids = text_to_input_ids(tokenizer=tokenizer, text=masked_text)
    layer_logits, layer_names = collect_logits(model, input_ids, layer_names, decoder_layer_names)

    return layer_logits, layer_names


def next_token_pred(model:Any, tokenizer:Any, text:str) -> Tuple[np.ndarray, np.array, Any]:
    """ Next Token Prediction (Language Modeling):
            Goal: Measure how well OLMo predicts the next token in a passage.
            How: Compute log probabilities of correct next tokens.
            Use case: Measures how well the model follows linguistic patterns.
            Interpretation: Higher log probability means the model is better at predicting the next token. """
    
    input_ids = text_to_input_ids(tokenizer, text)
    with torch.no_grad():
        logits = model(input_ids).logits  # Get logits from the model
    
    probs = scipy.special.softmax(logits.numpy(), axis=-1)  # Convert to probabilities
    next_token_id = input_ids[0, 1:].numpy()  # Shifted input (targets)
    predicted_probs = np.array([probs[i, token] for i, token in enumerate(next_token_id)])

    avg_log_prob = np.mean(np.log(predicted_probs))
    print(f"Average Log Probability: {avg_log_prob:.4f}")
    
    return (
        probs,
        predicted_probs,
        avg_log_prob
    )


def text_generation_task(model:Any, tokenizer:Any, content1:str, content2:str, max_new_tokens:int, temp:float, rep_penalty:float, sample:bool):
    """ Generating text (chatbots, QA, completion) for testing the model's output quality
        Examples:
        "content1": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
        "content2": "What is y if y=2*2-4+(3*2)" """

    messages:Dict[str] = [
    {"role": "system", "content": content1},
    {"role": "user", "content": content2}
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cpu')

    generated_ids = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        temperature=temp, 
        repetition_penalty=rep_penalty, 
        do_sample=sample, 
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Response: {response}")



""" ******************** COST AND METRICS ******************** """

def measure_time(func, *args, **kwargs):
    """ Measures execution time of any function """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    print(f"Execution Time for {func.__name__}: {elapsed_time:.4f} seconds")
    
    return result, elapsed_time


def energy_of_union(layer_1, layer_2):
    """ Compute energy of union between two layers """

    layer_1 = torch.tensor(layer_1)
    layer_2 = torch.tensor(layer_2)

    # Normalize activations
    norm1 = F.normalize(layer_1, dim=-1)
    norm2 = F.normalize(layer_2, dim=-1)

    # Cosine similarity
    similarity = torch.matmul(norm1, norm2.T)

    return similarity.mean().item()



""" ******************** FULL ANALYSIS OF EXTRINSIC AND EXTRINSIC PERFORMANCE ******************** """

def start_time() -> float:
    """ Start time """
    start_time = time.perf_counter()
    return start_time

def end_time(start:float) -> float:
    """ End time to get total latency w. 2 decimals """
    end = time.perf_counter()
    latency = end - start
    return round(latency, 2)


def run_full_analysis(models:Dict[Any], tokenizer:Any, analysis_params:Dict[Any], device:str|None, full_path:str) -> pd.DataFrame:
    """ Runs full analysis on all models w. SAME TOKENIZER and returns a DataFrame with results """

    results = []
    for model_name, model in models.items():
        result = full_analysis(
            model=model, tokenizer=tokenizer, model_name=model_name, device=device,
            content1=analysis_params.get('content1'), content2=analysis_params.get('content2'), save_path=full_path,
            max_new_tokens=analysis_params.get('max_new_tokens'), temp=analysis_params.get('temp'), rep_penalty=analysis_params.get('rep_penalty')
        )
        results.append(result)

    return pd.DataFrame(results) 
    

def full_analysis(model:Any, tokenizer:Any, model_name, device:str|None, content1:str, content2:str, save_path:str, max_new_tokens:int, temp:float, rep_penalty:float) -> Dict:
    """ Evaluate model across multiple dimensions: text generation, perplexity, hardware usage, and hidden state analysis """

    # Special handling for bnb quantized models (8-bit or 4-bit)
    if hasattr(model, 'bnb_config') or getattr(model, 'is_loaded_in_8bit', False):
        print("The model has a bnb_config or is_loaded_in_8bit attribute - likely quantized using bitsandbytes.")
    elif device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Since device is None, the model was moved to default device: {device}...")
    else:  
        model.to(device)
        print(f"The model was moved to selected device: {device}...")

    # --- TEXT GENERATION ---
    messages = [
        {"role": "system", "content": content1},
        {"role": "user", "content": content2}
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)

    for param in model.parameters():
        param.data = param.data.to(model.device)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(model.device)

    gen_start = start_time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            temperature=temp, 
            repetition_penalty=rep_penalty, 
            do_sample=True, 
            eos_token_id=tokenizer.eos_token_id
        ).to(model.device)

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
    #response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    latency = end_time(start=gen_start)

    print(f"Generated Tokens: {generated_ids.shape[-1:]}\nResponse: {response}")

    # --- PERPLEXITY ---
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.to(model.device)  # Cross-entropy loss
        perplexity = torch.exp(loss).item()  # PPL = e^loss

    # --- HARDWARE USAGE ---
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # --- ENERGY OF UNION (Between First & Last Layer) ---
    try:
        model.config.output_hidden_states = True  

        with torch.no_grad():
            outputs = model(input_ids, return_dict=True, output_hidden_states=True)

            if outputs.hidden_states is None:
                raise ValueError(f"Hidden states not returned by {model_name}.")

            activations = outputs.hidden_states

            first_layer = activations[0].to(model.device).mean(dim=1)
            last_layer = activations[-1].to(model.device).mean(dim=1) 

            # Take the mean across the sequence length before normalization
            first_layer = first_layer.mean(dim=1)  # Shape: (batch_size, hidden_dim)
            last_layer = last_layer.mean(dim=1)  # Shape: (batch_size, hidden_dim)

            norm1 = F.normalize(first_layer, dim=-1)
            norm2 = F.normalize(last_layer, dim=-1)

            # Compute similarity properly
            similarity = torch.matmul(norm1, norm2.T).mean().item()

    except Exception as e:
        print(f"Warning: Failed to compute Energy of Union for {model_name}: {e}")
        similarity = None  # If hidden states fail, set similarity to None

    # --- STORE RESULTS ---
    results = {
        'Model': model_name,
        'Perplexity': perplexity,
        'CPU Usage (%)': cpu_usage,
        'RAM Usage (%)': ram_usage,
        'GPU Memory (MB)': gpu_memory,
        'Energy of Union': similarity,
        'Sample Response': response,
        'Latency': latency
    }

    # Save to JSON
    json_path = f"{save_path}/{model_name}.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {json_path}")
        
    return results
