import os
import json
import time
import psutil
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Any, Dict
from datetime import datetime

def start_time() -> float:
    """Start time"""
    return time.perf_counter()

def end_time(start: float) -> float:
    """End time to get total latency rounded to 2 decimals"""
    end = time.perf_counter()
    latency = end - start
    return round(latency, 2)

def run_full_analysis(models: Dict[Any], tokenizer: Any, analysis_params: Dict[Any], device: str | None, full_path: str) -> pd.DataFrame:
    """Runs full analysis on all models using the same tokenizer and returns a DataFrame with results"""

    results = []
    for model_name, model in models.items():
        result = full_analysis(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            content1=analysis_params.get('content1'),
            content2=analysis_params.get('content2'),
            save_path=full_path,
            max_new_tokens=analysis_params.get('max_new_tokens'),
            temp=analysis_params.get('temp'),
            rep_penalty=analysis_params.get('rep_penalty')
        )
        results.append(result)

    return pd.DataFrame(results)

def full_analysis(model: Any, tokenizer: Any, model_name: str, device: str | None, content1: str, content2: str, save_path: str, max_new_tokens: int, temp: float, rep_penalty: float) -> Dict:
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
        {"role": "system", "content": content1},
        {"role": "user", "content": content2}
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model_device)

    gen_start = start_time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            repetition_penalty=rep_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    latency = end_time(gen_start)

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
