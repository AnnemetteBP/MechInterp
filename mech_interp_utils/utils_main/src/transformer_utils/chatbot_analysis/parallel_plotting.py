from typing import Tuple, List, Any, Dict, Optional, Union
import os, json, time, psutil
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pathlib import Path
import re
import numpy as np

from .json_utils import load_json_results, preprocess_df

def extract_think_content(response:str) -> str:
    # Find all occurrences of <think>...</think> or <think>... (open only)
    think_blocks = re.findall(r"<think>(.*?)(?:</think>|$)", response, re.DOTALL)
    # Skip the first one (intro) and join the rest
    return " ".join(block.strip() for block in think_blocks[1:]) if len(think_blocks) > 1 else ""

def _preprocess_df(df: pd.DataFrame, reference_file: str) -> pd.DataFrame:
    # Load reference text
    with open(reference_file, 'r') as f:
        ref_data = json.load(f)
    
    # Extract reference text from response
    reference_text = extract_answer(ref_data['Sample Response'])
    
    # Compute 'Think Content' and Precision
    df = df.copy()
    df['Think Content'] = df['Sample Response'].apply(extract_answer)
    df['Precision'] = df['Think Content'].apply(lambda x: compute_precision(reference_text, x))

    return df


def extract_answer(response: str) -> str:
    match = re.search(r"\nAnswer:\s*(.*)", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def compute_precision(ref_text: str, pred_text: str) -> float:
    # Calculate Jaccard similarity as precision
    return jaccard_similarity(ref_text, pred_text)


def tokenize(text: str) -> set:
    return set(re.findall(r"\w+|\S", text.lower()))


def jaccard_similarity(ref_text: str, pred_text: str) -> float:
    ref_tokens = tokenize(ref_text)
    pred_tokens = tokenize(pred_text)
    union = ref_tokens.union(pred_tokens)
    if not union:
        return 0.0
    intersection = ref_tokens.intersection(pred_tokens)
    return len(intersection) / len(union)


def _metric_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    reference_file: str,
    title: str = "Model Metric Heatmap",
    fig_path: Union[str, None] = None,
) -> None:
    df = df.copy()
    df = _preprocess_df(df, reference_file=reference_file)

    if "Model" not in df.columns:
        raise ValueError("'Model' column is required in the DataFrame for heatmap rows.")

    df = df.set_index("Model")

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"The following metrics are missing in the DataFrame: {missing}")

    df_numeric = df[metrics]

    # Debugging: Check if values are NaN or missing
    #print("Checking if there are NaN values in the selected metrics:")
    #print(df_numeric.isna().sum())

    # True values for annotations
    z_text = [[f"{val:.2f}" if pd.notnull(val) else "" for val in row] for row in df_numeric.values]

    # Normalize each column separately (min-max to [0, 1])
    df_normalized = df_numeric.copy()
    for col in df_normalized.columns:
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()
        if col_max != col_min:  # Avoid division by zero in case all values are the same
            df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)

    # Check if normalization works
    #print("Data after normalization:\n", df_normalized.head())

    z = df_normalized.values
    x = df_normalized.columns.tolist()
    y = df_normalized.index.tolist()

    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=z_text,
        colorscale='Blackbody_r',  # Options: 'Cividis', 'Turbo', 'Rainbow'
        showscale=True,
        hoverinfo='z',
        zmin=0,
        zmax=1
    )
    fig.data[0].colorbar.tickvals = [0, 1]
    fig.data[0].colorbar.ticktext = ["min", "max"]
    fig.data[0].colorbar.ticks = "outside"
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickvals=[0, 1],
            ticktext=["min", "max"],
            ticks="outside",
            lenmode="pixels",
            len=200
        ),
        title=title,
        xaxis=dict(title="Metrics"),
        yaxis=dict(title="Models"),
        font=dict(family="Times New Roman", size=12),
        margin=dict(l=100, r=100, t=100, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    if fig_path:
        fig.write_html(fig_path)
        print(f"Saved heatmap to: {fig_path}")
    else:
        fig.show()



def _parallel_plot(
    df: pd.DataFrame,
    reference_file: str,
    dimensions: List[str] | None = None,
    color_metric: str | None = None,
    colormap: Union[str, List, Dict, Any, None] = None,
    title: str | None = None,
    fig_path: str | None = None,
) -> None:
    with open(reference_file, 'r') as f:
        ref_data = json.load(f)
    reference_response = ref_data['Sample Response']
    reference_text = extract_answer(reference_response)

    df = df.copy()
    df['Think Content'] = df['Sample Response'].apply(extract_answer)
    df['Precision'] = df['Think Content'].apply(lambda x: compute_precision(reference_text, x))

    reference_model_name = ref_data.get('Model', None)
    if reference_model_name:
        df = df[df['Model'] != reference_model_name]

    df.columns = df.columns.astype(str).str.strip()

    for col in ['Token Count', 'Activation Similarity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if dimensions is None:
        dimensions = [
            'Model', 'Token Count', 'Perplexity',
            'Latency (s)', 'CPU Usage (%)', 'RAM Usage (%)',
            'GPU Memory (MB)', 'Logit Std', 'Last Layer Activation Std',
            'Activation Similarity',
        ]

    if 'Model' not in dimensions:
        dimensions = ['Model'] + dimensions

    df['Model'] = df['Model'].astype(str)
    models = sorted(df['Model'].unique())
    model_indices = {model: i for i, model in enumerate(models)}
    df['Model Index'] = df['Model'].map(model_indices)

    for col in dimensions:
        if col != "Model" and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype('category').cat.codes

    df = df.dropna(subset=dimensions)

    default_colors = px.colors.qualitative.Bold + px.colors.qualitative.Set2 + px.colors.qualitative.Dark24
    colors = colormap if colormap else default_colors[:len(models)]
    model_to_color = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    df['Color'] = df['Model'].map(model_to_color)

    parcoords_dims = []
    for col in dimensions:
        if col != "Model":
            parcoords_dims.append(dict(label=col, values=df[col]))

    fig = go.Figure()
    fig.add_trace(
        go.Parcoords(
            line=dict(
                color=df['Model Index'],
                colorscale=[[i / (len(models) - 1), model_to_color[model]] for i, model in enumerate(models)],
                showscale=False
            ),
            dimensions=parcoords_dims
        )
    )

    for model in models:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=model_to_color[model]),
            name=model,
            showlegend=True
        ))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=100, t=50, b=50),
        font=dict(family="Times New Roman", size=12),
        legend=dict(title="Model", orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    if fig_path:
        fig.write_html(fig_path)
        print(f"Saved plot to: {fig_path}")
    else:
        fig.show()


def plot_chatbot_analysis(
    json_logs: str | None = None,
    df_columns: List[str] | None = None,
    deep_thinking: bool = False,
    parallel_plot: bool = False,
    color_metric: str | None = None,
    colormap: str | None = None,
    title: str | None = None,
    fig_path: str | None = None,
    reference_file: str | None = None
) -> None:
    reference_file = reference_file or "logs/chatbot_logs/llama.8b-instruct.fp32.json"

    if json_logs is None:
        raise ValueError("Logging data is needed...\nPlease try running the chatbot analysis first!")

    else:
        json_data = load_json_results(json_dir=json_logs)
        df = json_data

    dimensions = df_columns or [
        'Model', 'Token Count', 'Perplexity',
        'Latency (s)', 'CPU Usage (%)', 'RAM Usage (%)',
        'GPU Memory (MB)', 'Logit Std', 'Last Layer Activation Std',
        'Activation Similarity', 'Precision'
    ]

    # Ensuring Token Count and Activation Similarity are treated as numeric
    for col in ['Token Count', 'Activation Similarity']:
        if col in df.columns:
            # Explicitly convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Optionally, replace NaN with 0 or another default value if you want
            df[col].fillna(0, inplace=True)  # Replace NaN with 0

    if parallel_plot:
        _parallel_plot(
            df=df,
            reference_file=reference_file,
            dimensions=dimensions,
            color_metric=color_metric,
            colormap=colormap,
            title=title,
            fig_path=fig_path
        )
    else:
        _metric_heatmap(
            df=df,
            metrics=[col for col in dimensions if col != 'Model'],
            reference_file=reference_file,
            title=title if title else "Model Metrics Overview",
            fig_path=fig_path
        )