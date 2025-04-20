import time
import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import psutil
from typing import Dict, Any
from datetime import datetime

def start_time() -> float:
    """Start timing."""
    return time.perf_counter()

def end_time(start: float) -> float:
    """End timing and return latency rounded to 2 decimals."""
    return round(time.perf_counter() - start, 2)

def run_full_analysis(models: Dict[Any], tokenizer: Any, analysis_params: Dict[Any], device: str | None, full_path: str) -> pd.DataFrame:
    """Runs full analysis on all models with SAME tokenizer and returns a DataFrame with results."""
    results = []
    for model_name, model in models.items():
        result = full_analysis(
            model=model, tokenizer=tokenizer, model_name=model_name, device=device,
            content1=analysis_params.get('content1'), content2=analysis_params.get('content2'), save_path=full_path,
            max_new_tokens=analysis_params.get('max_new_tokens'), temp=analysis_params.get('temp'), rep_penalty=analysis_params.get('rep_penalty')
        )
        results.append(result)
    return pd.DataFrame(results)

def safe_filename(name: str) -> str:
    """Make filename safe by removing slashes and colons."""
    return name.replace('/', '_').replace(':', '_')

def full_analysis(model: Any, tokenizer: Any, model_name: str, device: str | None, 
                  content1: str, content2: str, save_path: str, 
                  max_new_tokens: int, temp: float, rep_penalty: float) -> Dict:
    """Evaluate model across text generation, perplexity, hardware usage, and hidden state analysis."""

    # --- DEVICE HANDLING ---
    if hasattr(model, 'bnb_config') or getattr(model, 'is_loaded_in_8bit', False):
        print("Quantized model detected, skipping .to(device).")
    elif device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device not specified. Using default device: {device}...")
        model.to(device)
    else:
        model.to(device)
        print(f"Model moved to selected device: {device}...")

    # --- TEXT GENERATION ---
    messages = [
        {"role": "system", "content": content1},
        {"role": "user", "content": content2}
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    input_ids = input_ids.to(next(model.parameters()).device)  # Match model's device

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
    print(f"Generated Tokens: {generated_ids.shape[-1]}\nResponse: {response}")

    # --- PERPLEXITY ---
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            perplexity = float('inf')
            print(f"Warning: NaN or Inf loss detected in {model_name}, setting perplexity to Inf.")
        else:
            perplexity = torch.exp(loss).item()

    # --- HARDWARE USAGE ---
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # --- HIDDEN STATE SIMILARITY (formerly Energy of Union) ---
    similarity = None
    try:
        model.config.output_hidden_states = True

        with torch.no_grad():
            outputs = model(input_ids, return_dict=True, output_hidden_states=True)

            if outputs.hidden_states is None:
                raise ValueError(f"Hidden states not returned by {model_name}.")

            activations = outputs.hidden_states
            first_layer = activations[0].mean(dim=1).mean(dim=1)  # Mean over sequence, then over batch
            last_layer = activations[-1].mean(dim=1).mean(dim=1)

            norm1 = F.normalize(first_layer.unsqueeze(0) + 1e-8, dim=-1)  # Add epsilon for stability
            norm2 = F.normalize(last_layer.unsqueeze(0) + 1e-8, dim=-1)

            if torch.any(torch.isnan(norm1)) or torch.any(torch.isnan(norm2)):
                print(f"Warning: NaNs detected in hidden states normalization for {model_name}.")
                similarity = None
            else:
                similarity = torch.matmul(norm1, norm2.T).item()
    except Exception as e:
        print(f"Warning: Failed to compute hidden state similarity for {model_name}: {e}")

    # --- STORE RESULTS ---
    results = {
        'Model': model_name,
        'Perplexity': perplexity,
        'CPU Usage (%)': cpu_usage,
        'RAM Usage (%)': ram_usage,
        'GPU Memory (MB)': gpu_memory,
        'First-Last Hidden Similarity': similarity,
        'Sample Response': response,
        'Latency (s)': latency,
        'Timestamp': datetime.now().isoformat()
    }

    safe_name = safe_filename(model_name)
    json_path = os.path.join(save_path, f"{safe_name}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {json_path}")

    return results
