from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os, json, time, psutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd


def _start_time() -> float:
    """ Start time """
    start_time = time.perf_counter()
    return start_time

def _end_time(start:float) -> float:
    """ End time to get total latency w. 2 decimals """
    end = time.perf_counter()
    latency = end - start
    return round(latency, 2)

def _run_chatbot_analysis(
        model:Any,
        tokenizer:Any,
        model_name:str,
        context:str,
        prompt:str,
        max_new_tokens:int,
        temperature:float,
        repetition_penalty:float,
        sample:bool,
        device:str|None,
        save_path:str
) -> Dict:
    """Evaluate model across multiple dimensions: text generation, perplexity, hardware usage, and hidden state analysis"""

    # --- DEVICE HANDLING ---
    quantized_model = hasattr(model, 'bnb_config') or getattr(model, 'is_loaded_in_8bit', False)
    if quantized_model:
        print(f"Quantized model detected ({model_name}), skipping .to(device)...")
    elif device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device not specified. Defaulting to: {device}")
        model.to(device)
    else:
        model.to(device)
        print(f"Moving model {model_name} to {device}...")

    model_device = next(model.parameters()).device

    # --- TEXT GENERATION ---
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model_device)

    gen_start = _start_time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=sample,
            eos_token_id=tokenizer.eos_token_id
        )
    latency = _end_time(gen_start)

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Generated Tokens: {generated_ids.shape[-1:]}\nResponse: {response}")

    # --- PERPLEXITY ---
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is NaN or Inf for {model_name}. Setting Perplexity to Infinity.")
            perplexity = float('inf')
        else:
            perplexity = torch.exp(loss).item()

    # --- HARDWARE USAGE ---
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # --- ACTIVATION SIMILARITY (formerly "Energy of Union") ---
    similarity = None
    try:
        model.config.output_hidden_states = True
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True, output_hidden_states=True)

        if outputs.hidden_states is None:
            raise ValueError(f"No hidden states returned by {model_name}.")

        activations = outputs.hidden_states
        first_layer = activations[0].mean(dim=1).mean(dim=1)  # mean across sequence and batch
        last_layer = activations[-1].mean(dim=1).mean(dim=1)

        norm1 = F.normalize(first_layer + 1e-8, dim=-1)
        norm2 = F.normalize(last_layer + 1e-8, dim=-1)

        if torch.any(torch.isnan(norm1)) or torch.any(torch.isnan(norm2)):
            print(f"Warning: NaN detected in normalized activations for {model_name}.")
            similarity = float('nan')
        else:
            similarity = torch.matmul(norm1, norm2.T).mean().item()

    except Exception as e:
        print(f"Warning: Failed to compute Activation Similarity for {model_name}: {e}")
        similarity = None

    # --- SAVE RESULTS ---
    results = {
        'Model': model_name,
        'Timestamp': datetime.now().isoformat(),
        'Precision': 'float32' if 'fp32' in model_name.lower() else 'float16',
        'Perplexity': perplexity,
        'CPU Usage (%)': cpu_usage,
        'RAM Usage (%)': ram_usage,
        'GPU Memory (MB)': gpu_memory,
        'Activation Similarity': similarity,
        'Last Layer Mean Activation': activations[-1].mean().item(),
        'Last Layer Activation Std': activations[-1].std().item(),
        'Mean Logits': outputs.logits.mean().item() if hasattr(outputs, 'logits') else None,
        'Logit Std': outputs.logits.std().item() if hasattr(outputs, 'logits') else None,
        'Token Count': generated_ids.shape[-1],
        'Sample Response': response,
        'Latency (s)': latency
    }

    safe_name = model_name.replace('/', '_').replace(':', '_').replace(' ', '_')
    json_path = os.path.join(save_path, f"{safe_name}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {json_path}")

    return results


def run_chatbot_analysis(
        models:Dict[Any],
        tokenizer:Any,
        context:str|None=None,
        prompt:str|None=None,
        max_new_tokens:int|None=None,
        temperature:float|None=None,
        repetition_penalty:float|None=None,
        sample:bool=True,
        device:str|None=None,
        full_path:str|None=None
) -> pd.DataFrame:

    """ Runs full analysis on all models w. SAME TOKENIZER and returns a DataFrame with results """

    default_params = {
        'context': "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.",
        'prompt': "Who was the US president during the Apollo 11 moon landing?",
        'max_new_tokens': 250,
        'temperature': 0.8,
        'repetition_penalty': 1.0,
        'sample': True,
        'device': None
    }

    params = {
        'context': context if context else default_params.get('context'),
        'prompt': prompt if prompt else default_params.get('prompt'),
        'max_new_tokens': max_new_tokens if max_new_tokens else default_params.get('max_new_tokens'),
        'temperature': temperature if temperature else default_params.get('temperature'),
        'repetition_penalty': repetition_penalty if repetition_penalty else default_params.get('repetition_penalty'),
        'sample': sample if sample else default_params.get('sample'),
        'device': device if device else default_params.get('device')
    }

    results = []
    for model_name, model in models.items():
        result = _run_chatbot_analysis(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            context=params.get('context'),
            prompt=params.get('prompt'),
            max_new_tokens=params.get('max_new_tokens'),
            temperature=params.get('temperature'),
            repetition_penalty=params.get('repetition_penalty'),
            sample=params.get('sample'),
            device=params.get('device'),
            save_path=full_path
        )

        results.append(result)

    return pd.DataFrame(results)
