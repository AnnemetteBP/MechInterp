from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os, json, time, psutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import random
import numpy as np
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

def set_deterministic_backend(seed:int=42) -> None:
        """ 
        Forces PyTorch to use only deterministic operations (e.g., disables non-deterministic GPU kernels).
        This is crucial for reproducibility: given the same inputs and model state, to get the same outputs every time.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def _run_nq_analysis(
        model: Any,
        tokenizer: Any,
        model_name: str,
        context: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        sample: bool,
        device: str | None,
        save_path: str,
        deterministic_backend: bool = True
) -> Dict:
    """Simplified evaluation pipeline without chat templates."""

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

    # --- PROMPT PREP ---
    full_prompt = f"{context.strip()}\n{prompt.strip()}\nAnswer:"

    # --- Ensure special tokens are set properly ---
    if tokenizer.pad_token is None:
        print(f"[DEBUG] tokenizer type: {type(tokenizer)}")
        assert hasattr(tokenizer, "pad_token"), f"Expected tokenizer-like object, got: {type(tokenizer)}"

        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("[INFO] pad_token was None; set to eos_token.")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("[WARNING] pad_token and eos_token were None; added '[PAD]' as pad_token.")

    # Update model config with pad_token_id if not already set
    if getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- Tokenize input ---
    input_dict = tokenizer(full_prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = input_dict['input_ids'].to(model_device)
    attention_mask = input_dict['attention_mask'].to(model_device)

    # --- Deterministic setup ---
    if deterministic_backend:
        set_deterministic_backend()

    # --- GENERATION ---
    gen_start = _start_time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=sample,
            pad_token_id=tokenizer.pad_token_id,  # Needed for padding in generation
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
    latency = _end_time(gen_start)

    # --- Decode output ---
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Generated Tokens: {generated_ids.shape[-1:]}\nResponse: {response}")

    # --- PERPLEXITY ---
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = float('inf') if torch.isnan(loss) or torch.isinf(loss) else torch.exp(loss).item()

    # --- HARDWARE ---
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # --- ACTIVATION SIMILARITY ---
    similarity = None
    try:
        model.config.output_hidden_states = True
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True, output_hidden_states=True)

        activations = outputs.hidden_states
        first_layer = activations[0].mean(dim=1).mean(dim=1)
        last_layer = activations[-1].mean(dim=1).mean(dim=1)

        norm1 = F.normalize(first_layer + 1e-8, dim=-1)
        norm2 = F.normalize(last_layer + 1e-8, dim=-1)

        similarity = torch.matmul(norm1, norm2.T).mean().item()
    except Exception as e:
        print(f"Warning: Failed to compute Activation Similarity for {model_name}: {e}")

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


def run_nq_analysis(
        model,
        tokenizer,
        model_name,
        dataset,
        save_path,
        max_new_tokens=50,
        num_samples=100,
        device=None,
        temperature=0.7,
        repetition_penalty=1.0,
        sample=False,
        deterministic_backend=True
):
    results = []

    for idx, sample in enumerate(dataset.select(range(num_samples))):
        question = sample["query"]
        true_answer = sample["answer"].split("####")[-1].strip()
        print(f"\n[{idx+1}/{num_samples}] Question: {question.strip()}")

        result = _run_nq_analysis(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            context="You are a helpful assistant that answers questions.",
            prompt=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            sample=sample,
            device=device,
            save_path=save_path,
            deterministic_backend=deterministic_backend
        )

        # Extract and compare numeric answer
        predicted_answer = result["Sample Response"]
        pred = ''.join(filter(str.isdigit, predicted_answer.split()[-1]))
        target = ''.join(filter(str.isdigit, true_answer))
        exact_match = int(pred == target)

        result.update({
            "True Answer": true_answer,
            "Predicted Answer": predicted_answer,
            "Exact Match": exact_match,
            "Sample Index": idx
        })

        results.append(result)

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_path, f"{model_name.replace('/', '_')}_nq_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")
    return df