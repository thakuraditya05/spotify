"""
train_and_save.py
──────────────────────────────────────────────────────────
Ek baar run karo yeh script → model.pkl generate hoga.
Phir Streamlit app usi pickle se predict karega.

Usage:
    python train_and_save.py --data prisha_datset.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Audio features used for clustering ──────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

# ── 1. Load & clean ──────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[1] Raw shape      : {df.shape}")

    # drop rows with missing track name (only 5 rows)
    df.dropna(subset=["track_name", "track_artist"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # drop non-informational columns
    drop_cols = [
        "track_id", "track_album_id", "playlist_id",
        "track_album_release_date", "playlist_name"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print(f"[2] After cleaning : {df.shape}")
    return df

# ── 2. Scale features ────────────────────────────────────────
def scale_features(df: pd.DataFrame):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[AUDIO_FEATURES])
    print(f"[3] Scaled features: {scaled.shape}")
    return scaled, scaler

# ── 3. Find best K via Silhouette ─────────────────────────────
def find_best_k(scaled: np.ndarray, k_max: int = 12) -> int:
    print("[4] Running Elbow + Silhouette to find optimal K ...")
    best_k, best_score = 2, -1
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        score = silhouette_score(scaled, labels, sample_size=3000, random_state=42)
        print(f"    K={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    print(f"[4] Best K = {best_k}  (silhouette={best_score:.4f})")
    return best_k

# ── 4. Train KMeans ──────────────────────────────────────────
def train_kmeans(scaled: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    print(f"[5] KMeans trained  : k={k}, inertia={km.inertia_:,.0f}")
    return km, labels

# ── 5. PCA (for visualisation only) ─────────────────────────
def fit_pca(scaled: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    print(f"[6] PCA variance    : {pca.explained_variance_ratio_}")
    return pca, coords

# ── 6. Label Encoder for genre ───────────────────────────────
def fit_label_encoder(df: pd.DataFrame):
    le = LabelEncoder()
    le.fit(df["playlist_genre"])
    return le

# ── 7. Build cluster profile lookup ─────────────────────────
def build_cluster_profiles(df: pd.DataFrame, labels: np.ndarray, k: int) -> dict:
    """
    Returns a dict:
      { cluster_id: { dominant_genre, avg_danceability, avg_energy,
                      avg_valence, avg_tempo, avg_popularity, size, vibe } }
    """
    df = df.copy()
    df["cluster"] = labels
    profiles = {}

    for c in range(k):
        sub = df[df["cluster"] == c]
        dom_genre = sub["playlist_genre"].value_counts().index[0]
        avg_e = sub["energy"].mean()
        avg_d = sub["danceability"].mean()
        avg_v = sub["valence"].mean()

        if avg_e > 0.75 and avg_d > 0.7:
            vibe = "⚡ High Energy Dance"
        elif avg_e > 0.75 and avg_d <= 0.6:
            vibe = "🤘 Intense & Powerful"
        elif avg_v > 0.65:
            vibe = "😊 Happy & Upbeat"
        elif avg_v < 0.35:
            vibe = "😔 Dark & Melancholic"
        elif avg_d > 0.7:
            vibe = "💃 Chill Dance"
        else:
            vibe = "🎵 Balanced Mix"

        profiles[int(c)] = {
            "dominant_genre":    dom_genre,
            "avg_danceability":  round(float(avg_d), 3),
            "avg_energy":        round(float(avg_e), 3),
            "avg_valence":       round(float(avg_v), 3),
            "avg_tempo":         round(float(sub["tempo"].mean()), 1),
            "avg_popularity":    round(float(sub["track_popularity"].mean()), 1),
            "size":              int(len(sub)),
            "vibe":              vibe,
        }
    return profiles


# ── MAIN ─────────────────────────────────────────────────────
def main(csv_path: str, output_pkl: str = "model.pkl"):
    print("=" * 55)
    print("  Spotify Genre Segmentation — Model Trainer")
    print("=" * 55)

    # steps
    df       = load_and_clean(csv_path)
    scaled, scaler = scale_features(df)
    best_k   = find_best_k(scaled)
    km, labels = train_kmeans(scaled, best_k)
    pca, pca_coords = fit_pca(scaled)
    le       = fit_label_encoder(df)
    profiles = build_cluster_profiles(df, labels, best_k)

    # attach cluster + pca to df for the recommendation engine
    df["cluster"] = labels
    df["pca_x"]   = pca_coords[:, 0]
    df["pca_y"]   = pca_coords[:, 1]

    # ── bundle everything into one pickle ────────────────────
    bundle = {
        # core ML objects
        "kmeans":        km,          # trained KMeans
        "scaler":        scaler,      # StandardScaler (fit on training data)
        "pca":           pca,         # PCA (2 components, for viz)
        "label_encoder": le,          # LabelEncoder for genre

        # metadata
        "k":             best_k,      # optimal number of clusters
        "audio_features": AUDIO_FEATURES,

        # data (used by recommender & viz in Streamlit)
        "df":            df,          # full cleaned df with cluster + pca cols

        # cluster profile lookup (for Insights tab)
        "cluster_profiles": profiles,
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Pickle saved → {output_pkl}")
    print(f"   Keys: {list(bundle.keys())}")
    print(f"   DataFrame shape: {df.shape}")
    print(f"   K = {best_k}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="prisha_datset.csv", help="Path to CSV")
    parser.add_argument("--output", default="model.pkl",          help="Output pickle path")
    args = parser.parse_args()
    main(args.data, args.output)
