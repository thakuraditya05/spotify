from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from typing import Any
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class ModelBundle:
    kmeans: Any
    scaler: Any
    pca: Any
    label_encoder: Any
    k: int
    audio_features: list[str]
    df: pd.DataFrame
    cluster_profiles: dict[int, dict[str, Any]]


RECOMMENDATION_COLUMNS = [
    "track_name",
    "track_artist",
    "playlist_genre",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "cluster",
]


def validate_bundle(raw: dict[str, Any]) -> ModelBundle:
    required = {
        "kmeans",
        "scaler",
        "pca",
        "label_encoder",
        "k",
        "audio_features",
        "df",
        "cluster_profiles",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise ValueError(f"Invalid model.pkl. Missing keys: {', '.join(missing)}")

    df = raw["df"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Invalid model.pkl. 'df' must be a pandas DataFrame.")

    features = list(raw["audio_features"])
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "Invalid model.pkl. DataFrame is missing audio feature columns: "
            + ", ".join(missing_features)
        )

    if "cluster" not in df.columns:
        raise ValueError("Invalid model.pkl. DataFrame must include a 'cluster' column.")

    return ModelBundle(
        kmeans=raw["kmeans"],
        scaler=raw["scaler"],
        pca=raw["pca"],
        label_encoder=raw["label_encoder"],
        k=int(raw["k"]),
        audio_features=features,
        df=df.copy(),
        cluster_profiles=dict(raw["cluster_profiles"]),
    )


@st.cache_resource(show_spinner=False)
def load_model_bundle(pkl_bytes: bytes) -> ModelBundle:
    with _string_dtype_pickle_compat():
        raw = pickle.load(io.BytesIO(pkl_bytes))
    return validate_bundle(raw)


@contextmanager
def _string_dtype_pickle_compat():
    """
    Compatibility shim for pickles created with pandas StringDtype signatures
    that include a second positional argument (for example, `na_value`).
    """
    original_init = pd.StringDtype.__init__

    def compat_init(self, storage=None, na_value=None):
        return original_init(self, storage=storage)

    pd.StringDtype.__init__ = compat_init
    try:
        yield
    finally:
        pd.StringDtype.__init__ = original_init


def predict_cluster(model: ModelBundle, feature_values: list[float]) -> int:
    arr = np.asarray(feature_values, dtype=float).reshape(1, -1)
    arr_scaled = model.scaler.transform(arr)
    return int(model.kmeans.predict(arr_scaled)[0])


def get_recommendations(
    df: pd.DataFrame,
    cluster_id: int,
    exclude_idx: int | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    mask = df["cluster"] == cluster_id
    if exclude_idx is not None:
        mask &= df.index != exclude_idx

    pool = df.loc[mask].copy()
    if pool.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    n = min(top_n, len(pool))
    sample = pool.sample(n=n, random_state=42)
    return sample[RECOMMENDATION_COLUMNS]


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return str(value)


def get_feature_defaults(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for feature in features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            defaults[feature] = float(df[feature].median())
        else:
            defaults[feature] = 0.0
    return defaults


def feature_slider_spec(feature: str, default_value: float) -> tuple[float, float, float, float]:
    name = feature.lower()
    config = {
        "danceability": (0.0, 1.0, default_value, 0.01),
        "energy": (0.0, 1.0, default_value, 0.01),
        "valence": (0.0, 1.0, default_value, 0.01),
        "speechiness": (0.0, 1.0, default_value, 0.01),
        "acousticness": (0.0, 1.0, default_value, 0.01),
        "instrumentalness": (0.0, 1.0, default_value, 0.01),
        "liveness": (0.0, 1.0, default_value, 0.01),
        "loudness": (-60.0, 0.0, min(max(default_value, -60.0), 0.0), 0.5),
        "tempo": (40.0, 240.0, min(max(default_value, 40.0), 240.0), 1.0),
    }
    return config.get(name, (0.0, 1.0, min(max(default_value, 0.0), 1.0), 0.01))


def feature_label(feature: str) -> str:
    labels = {
        "danceability": "💃 Danceability",
        "energy": "⚡ Energy",
        "loudness": "🔊 Loudness (dB)",
        "speechiness": "🗣️ Speechiness",
        "acousticness": "🎸 Acousticness",
        "instrumentalness": "🎹 Instrumentalness",
        "liveness": "🎤 Liveness",
        "valence": "😊 Valence",
        "tempo": "🥁 Tempo (BPM)",
    }
    return labels.get(feature, feature.replace("_", " ").title())


def build_manual_feature_input(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    defaults = get_feature_defaults(df, features)
    return {feature: defaults.get(feature, 0.0) for feature in features}
