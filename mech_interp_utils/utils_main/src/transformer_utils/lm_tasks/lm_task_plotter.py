from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from functools import partial

import os
import time
import psutil
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from scipy.stats import entropy

import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa


def plot_full_analysis(df:pd.DataFrame, save_name:str, dir:str) -> None:
    """ Generate comparison plots for different models """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plt.rcParams['font.family'] = 'Times New Roman'
    # Perplexity
    sns.barplot(x='Model', y='Perplexity', data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Perplexity") # (Lower is Better)
    
    # Hardware Usage
    sns.barplot(x='Model', y='CPU Usage (%)', data=df, ax=axes[0, 1])
    axes[0, 1].set_title("CPU Usage")

    sns.barplot(x='Model', y='RAM Usage (%)', data=df, ax=axes[1, 0])
    axes[1, 0].set_title("RAM Usage")

    # Check if Energy of Union values exist
    if df['Energy of Union'].notna().any():
        sns.barplot(x='Model', y='Energy of Union', data=df, ax=axes[1, 1])
        axes[1, 1].set_title("Energy of Union Between Layers")
    else:
        axes[1, 1].axis('off')  # Hide subplot if no data

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(dir), exist_ok=True) 
    plt.savefig(f"{dir}/{save_name}.jpg")
    plt.show()


def plot_full_analysis_grouped(json_dir:str, save_name:str, title:str, dir:str) -> None:
    """ Load results from JSON and plot metrics grouped on the x-axis with scaling. """

    # Load all JSON files into a DataFrame
    json_files = glob.glob(f"{json_dir}/*.json")
    if not json_files:
        print("No JSON files found. Run full_analysis first.")
        return

    all_results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)

    df = pd.DataFrame(all_results)

    # Check if essential columns exist
    required_cols = ['Model', 'Perplexity', 'CPU Usage (%)', 'RAM Usage (%)', 'GPU Memory (MB)', 'Energy of Union']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} missing, filling with NaN.")
            df[col] = None  # Add column if missing

    # Handle missing Perplexity values by replacing NaN with median
    df['Perplexity'] = pd.to_numeric(df['Perplexity'], errors='coerce')  # Convert to numeric
    df['Energy of Union'] = pd.to_numeric(df['Energy of Union'], errors='coerce')

    df['Perplexity'].fillna(df['Perplexity'].median(), inplace=True)
    df['Energy of Union'].fillna(df['Energy of Union'].median(), inplace=True)

    # Scale Perplexity & Energy of Union to a comparable range
    scaler = MinMaxScaler()
    df[['Perplexity', 'Energy of Union']] = scaler.fit_transform(df[['Perplexity', 'Energy of Union']])

    # Melt the dataframe to have "Metric" on the X-axis
    df_melted = df.melt(id_vars=['Model'], 
                         value_vars=['Perplexity', 'CPU Usage (%)', 'RAM Usage (%)', 'GPU Memory (MB)', 'Energy of Union'],
                         var_name='Metric', value_name='Value')

    # Plot using seaborn
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model')

    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.xticks(rotation=45)
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(os.path.dirname(dir), exist_ok=True) 
    plt.savefig(f"{dir}/{save_name}.jpg")
    plt.show()