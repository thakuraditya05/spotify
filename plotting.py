from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

CLUSTER_PAL = [
    "#1db954",
    "#1ed760",
    "#17a349",
    "#57d9a3",
    "#00b3b3",
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
]


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_PAL[int(cluster_id) % len(CLUSTER_PAL)]


def _maybe_set_theme() -> None:
    # Avoid resetting the global theme repeatedly when Streamlit reruns.
    if not getattr(_maybe_set_theme, "_done", False):
        sns.set_theme(style="whitegrid", context="notebook")
        _maybe_set_theme._done = True


def _fig_ax(figsize: tuple[float, float] = (8, 4)):
    _maybe_set_theme()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def plot_genre_distribution(df: pd.DataFrame):
    fig, ax = _fig_ax(figsize=(8, 4.5))
    counts = df["playlist_genre"].value_counts().sort_values(ascending=False)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_xlabel("Songs")
    ax.set_ylabel("Genre")
    ax.set_title("Songs per Genre")
    return fig


def plot_feature_distribution_by_genre(df: pd.DataFrame, feature: str):
    fig, ax = _fig_ax(figsize=(8, 5))
    order = (
        df.groupby("playlist_genre")[feature]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=df,
        x=feature,
        y="playlist_genre",
        order=order,
        ax=ax,
        palette="Set3",
    )
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel("Genre")
    ax.set_title(f"{feature.replace('_', ' ').title()} by Genre")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str]):
    fig, ax = _fig_ax(figsize=(7, 5.5))
    cols = _ensure_columns(df, features)
    corr = df[cols].corr()
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, linewidths=0.5)
    ax.set_title("Feature Correlation")
    return fig


def plot_genre_feature_means(df: pd.DataFrame, features: list[str]):
    fig, ax = _fig_ax(figsize=(9, 5))
    cols = _ensure_columns(df, features)
    means = df.groupby("playlist_genre")[cols].mean()
    sns.heatmap(means, ax=ax, cmap="mako", cbar=True)
    ax.set_xlabel("Audio Feature")
    ax.set_ylabel("Genre")
    ax.set_title("Average Audio Features by Genre")
    return fig


def _get_pca_coords(df: pd.DataFrame, pca) -> pd.DataFrame:
    if {"pca_x", "pca_y"}.issubset(df.columns):
        return df[["pca_x", "pca_y"]].rename(columns={"pca_x": "x", "pca_y": "y"})

    cols = _ensure_columns(df, AUDIO_FEATURES)
    if not cols:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cols = [c for c in numeric_cols if c not in {"cluster", "pca_x", "pca_y"}]

    if cols and pca is not None:
        coords = pca.transform(df[cols].values)
        return pd.DataFrame(coords[:, :2], columns=["x", "y"], index=df.index)

    fallback = df.select_dtypes(include=[np.number]).head(0)
    raise ValueError(
        "Could not derive PCA coordinates. Expected 'pca_x' and 'pca_y' columns "
        "or compatible PCA inputs."
    )


def plot_pca_clusters(df: pd.DataFrame, pca, color_by: str, n_points: int):
    fig, ax = _fig_ax(figsize=(8, 5))
    if len(df) > n_points:
        sample = df.sample(n_points, random_state=42)
    else:
        sample = df.copy()

    coords = _get_pca_coords(sample, pca)
    sample = sample.join(coords)

    if "Cluster" in color_by or "cluster" in color_by:
        colors = [cluster_color(cid) for cid in sample["cluster"]]
        ax.scatter(sample["x"], sample["y"], c=colors, s=18, alpha=0.7)
        ax.set_title("PCA: Colored by Cluster")
    else:
        sns.scatterplot(
            data=sample,
            x="x",
            y="y",
            hue="playlist_genre",
            palette="tab10",
            s=18,
            alpha=0.7,
            ax=ax,
            legend=False,
        )
        ax.set_title("PCA: Colored by Genre")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return fig


def plot_cluster_sizes(df: pd.DataFrame):
    fig, ax = _fig_ax(figsize=(6.5, 4))
    counts = df["cluster"].value_counts().sort_index()
    colors = [cluster_color(c) for c in counts.index]
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette=colors, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Songs")
    ax.set_title("Cluster Sizes")
    return fig


def plot_cluster_genre_mix(df: pd.DataFrame, k: int):
    fig, ax = _fig_ax(figsize=(9, 5))
    counts = (
        df.groupby(["cluster", "playlist_genre"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(range(k)).fillna(0)
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Songs")
    ax.set_title("Genre Mix per Cluster")
    ax.legend(title="Genre", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_input_vs_cluster_center(
    features: list[str],
    scaler,
    kmeans,
    cluster_id: int,
    feature_values: list[float],
):
    fig, ax = _fig_ax(figsize=(9, 4.5))
    input_arr = np.asarray(feature_values, dtype=float).reshape(1, -1)

    centers = kmeans.cluster_centers_
    if scaler is not None:
        centers = scaler.inverse_transform(centers)
        input_arr = input_arr

    center = centers[int(cluster_id)]

    x = np.arange(len(features))
    width = 0.38
    ax.bar(x - width / 2, input_arr[0], width, label="Your input", color="#1db954")
    ax.bar(x + width / 2, center, width, label="Cluster avg", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ") for f in features], rotation=45, ha="right")
    ax.set_title("Your Features vs Cluster Center")
    ax.legend()
    fig.tight_layout()
    return fig
