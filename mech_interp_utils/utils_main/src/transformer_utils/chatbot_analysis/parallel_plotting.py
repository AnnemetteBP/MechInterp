from typing import Tuple, List, Any, Dict, Optional, Union
import os, json, time, psutil
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from .json_utils import load_json_results, preprocess_df

def _parallel_plot(
        df: pd.DataFrame,
        dimensions: List[str] | None = None,
        color_metric: str | None = None,
        colormap: Union[str, List, Dict, Any, None] = None,
        title: str | None = None,
        fig_path: str | None = None
) -> None:

    # --- Default Dimensions ---
    if dimensions is None:
        dimensions = [
            'Model', 'Perplexity', 'Latency (s)',
            'CPU Usage (%)', 'RAM Usage (%)', 'GPU Memory (MB)',
            'Activation Similarity', 'Last Layer Activation Std', 'Logit Std',
            'Token Count', 'Precision'
        ]

    # --- 'Model' is First Dimension ---
    if 'Model' not in dimensions:
        dimensions = ['Model'] + dimensions

    df = df.copy()
    df['Model'] = df['Model'].astype(str)
    models = sorted(df['Model'].unique())
    model_indices = {model: i for i, model in enumerate(models)}
    df['Model Index'] = df['Model'].map(model_indices)

    # --- Colors ---
    default_colors = px.colors.qualitative.Bold + px.colors.qualitative.Set2 + px.colors.qualitative.Dark24
    colors = colormap if colormap else default_colors[:len(models)]
    model_to_color = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    df['Color'] = df['Model'].map(model_to_color)

    # --- Build Dimensions ---
    parcoords_dims = [
        dict(label="Model", values=df["Model Index"], tickvals=list(model_indices.values()), ticktext=list(model_indices.keys()))
    ]
    for col in dimensions:
        if col != "Model":
            parcoords_dims.append(dict(label=col, values=df[col]))

    # --- Plot ---
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

    # --- Add Custom Legend using Dummy Scatter Traces ---
    for model in models:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=model_to_color[model]),
            name=model,
            showlegend=True
        ))

    # --- Layout ---
    fig.update_layout(
        #title=title or "Model Comparison (Discrete Colors)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=12),
        legend=dict(title="Model", orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    # --- Save or Show ---
    if fig_path:
        fig.write_html(fig_path)
        print(f"Saved plot to: {fig_path}")
    else:
        fig.show()


def _plot_pairwise_matrix(
        df:pd.DataFrame,
        df_columns:List[str]|None=None,
        metric:str|None=None,
        hue:str|None=None,
        title:str|None=None,
        fig_path:str|None=None
) -> None:
    
    if df_columns is None:
        metrics = ['Perplexity', 'Latency (s)', 'Activation Similarity', 'Last Layer Activation Std', 'GPU Memory (MB)']
    
    sns.pairplot(df[metrics + [metric if metric is not None else 'Precision']], hue=hue if hue is not None else "Precision")
    plt.suptitle(title if title is not None else "Pairwise Metric Comparison", fontsize=12)

    if fig_path is not None:
        plt.savefig(fig_path)
    
    plt.show()


def plot_chatbot_analysis(
        json_logs:str|None=None,
        df_columns:List[str]|None=None,
        parallel_plot:bool=False,
        pairwise_matrix:bool=False,
        corr_heatmap:bool=False,
        metric:str|None=None, # 'Perplexity'
        color_metric:str|None=None, # 'Perplexity'
        colormap:str|None=None,
        title:str|None=None,
        fig_path:str|None=None,
) -> None:

    if json_logs is None:
        raise ValueError("Logging data is needed...\nPlease try running the chatbot analysis first!")
    
    else:
        json_data = load_json_results(json_dir=json_logs)
        #df = _preprocess_df(df=json_data)
        df = json_data
    
    if parallel_plot:
        _parallel_plot(
            df=df,
            dimensions=df_columns,
            color_metric=color_metric,
            colormap=colormap,
            title=title,
            fig_path=fig_path
        )

    elif pairwise_matrix:
        _plot_pairwise_matrix(
        df=df,
        df_columns=df_columns,
        metric=metric,
        hue=color_metric,
        title=title,
        fig_path=fig_path
    )
    