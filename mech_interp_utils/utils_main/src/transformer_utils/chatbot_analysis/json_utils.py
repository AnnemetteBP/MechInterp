from typing import Tuple, List, Any, Dict, Optional
import os, json, time, psutil
from datetime import datetime
import pandas as pd


def load_json_results(json_dir:str) -> pd.DataFrame:
    """Load all .json result files into a single DataFrame"""
    all_data = []
    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            with open(os.path.join(json_dir, file), 'r') as f:
                data = json.load(f)
                all_data.append(data)
    return pd.DataFrame(all_data)


def preprocess_df(df:pd.DataFrame) -> pd.DataFrame:
    # Convert to numeric where needed
    df['Model Size (B)'] = pd.to_numeric(df['Model Size (B)'], errors='coerce')
    df['GPU Memory (MB)'] = pd.to_numeric(df['GPU Memory (MB)'], errors='coerce')
    df['Activation Similarity'] = pd.to_numeric(df['Activation Similarity'], errors='coerce')
    df['Perplexity'] = pd.to_numeric(df['Perplexity'], errors='coerce')
    df['Latency (s)'] = pd.to_numeric(df['Latency (s)'], errors='coerce')
    df['Last Layer Activation Std'] = pd.to_numeric(df['Last Layer Activation Std'], errors='coerce')
    return df