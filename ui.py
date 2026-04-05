from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from plotting import (
    CLUSTER_PAL,
    cluster_color,
    plot_cluster_genre_mix,
    plot_cluster_sizes,
    plot_correlation_heatmap,
    plot_feature_distribution_by_genre,
    plot_genre_distribution,
    plot_genre_feature_means,
    plot_input_vs_cluster_center,
    plot_pca_clusters,
)
from utils import (
    ModelBundle,
    build_manual_feature_input,
    feature_label,
    feature_slider_spec,
    get_recommendations,
    predict_cluster,
    safe_text,
)

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif;}
.stApp {background:#0d0d1a; color:#e8e8f0;}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#121228,#0a0a18);
    border-right:1px solid #1db95430;
}
[data-testid="stSidebar"] * {color:#c8c8d8 !important;}
h1 {color:#1db954 !important; font-size:2.2rem !important;}
h2 {color:#1db954 !important;}
h3 {color:#1ed760 !important;}
[data-testid="stMetric"] {
    background:#161630;
    border:1px solid #1db95435;
    border-radius:14px;
    padding:16px 20px;
}
[data-testid="stMetricLabel"] {color:#1db954 !important; font-size:0.8rem;}
[data-testid="stMetricValue"] {color:#fff !important; font-size:1.8rem;}
.stTabs [data-baseweb="tab-list"] {
    background:#161630;
    border-radius:12px;
    gap:6px;
    padding:6px;
}
.stTabs [data-baseweb="tab"] {
    background:transparent;
    color:#888 !important;
    border-radius:8px;
    padding:8px 16px;
    font-weight:600;
    border:none;
}
.stTabs [aria-selected="true"] {
    background:#1db954 !important;
    color:#000 !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1db954,#17a349);
    color:#000 !important;
    font-weight:700;
    border:none;
    border-radius:25px;
    padding:10px 26px;
    transition:all .25s;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 6px 20px #1db95440;
}
.card {
    background:#161630;
    border:1px solid #1db95430;
    border-left:4px solid #1db954;
    border-radius:14px;
    padding:18px 22px;
    margin-bottom:14px;
}
.rec-row {
    background:#161630;
    border:1px solid #ffffff10;
    border-radius:12px;
    padding:14px 18px;
    margin:7px 0;
}
</style>
"""


def apply_app_styling() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_sidebar() -> object:
    with st.sidebar:
        st.markdown("## 🎵 Spotify Segmentation")
        st.markdown("---")
        pkl_file = st.file_uploader(
            "📦 Upload **model.pkl**",
            type=["pkl"],
            help="Run train_and_save.py first to generate this file",
        )
        st.markdown("---")
        st.caption("Generate pickle:\n```\npython train_and_save.py \\\n  --data prisha_datset.csv\n```")
    return pkl_file


def render_hero() -> None:
    st.markdown(
        """
        <div style='text-align:center;padding:28px 0 6px'>
          <h1>🎵 Spotify Genre Segmentation</h1>
          <p style='color:#888;margin-top:-8px;font-size:1rem;'>
            KMeans Clustering · PCA Visualization · Song Recommender
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_start() -> None:
    st.info("👈 Upload **model.pkl** from the sidebar to get started!", icon="🎵")
    st.markdown(
        """
        <div class="card">
        <h3>🚀 Quick Start</h3>
        <p><b>Step 1</b> — Train the model (run once):</p>
        <code>python train_and_save.py --data prisha_datset.csv</code>
        <br><br>
        <p><b>Step 2</b> — Upload the generated <code>model.pkl</code> using the sidebar.</p>
        <p><b>Step 3</b> — Explore EDA, clusters, and get recommendations instantly!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations(recs: pd.DataFrame, heading: str) -> None:
    st.markdown(f"### {heading}")
    if recs.empty:
        st.warning("No recommendations available for this cluster.")
        return

    for rank, (_, row) in enumerate(recs.iterrows(), start=1):
        color = cluster_color(int(row["cluster"]))
        st.markdown(
            f"""
            <div class="rec-row" style="border-left:4px solid {color};">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                <div>
                  <span style="color:#fff;font-weight:700;font-size:1rem;">#{rank} &nbsp; {safe_text(row['track_name'])}</span>
                  <br><span style="color:#888;font-size:.85rem;">by {safe_text(row['track_artist'])}</span>
                </div>
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                  <span style="background:#1db95420;color:#1db954;padding:3px 12px;border-radius:20px;font-size:.78rem;font-weight:700;">
                    {safe_text(row['playlist_genre'])}
                  </span>
                  <span style="color:#888;font-size:.82rem;">
                    💃{float(row['danceability']):.2f} &nbsp;⚡{float(row['energy']):.2f} &nbsp;😊{float(row['valence']):.2f}
                  </span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_overview_tab(model: ModelBundle) -> None:
    df = model.df
    features = model.audio_features
    st.markdown("## 📊 Dataset + Model Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎵 Songs", f"{len(df):,}")
    c2.metric("🔵 Clusters (K)", model.k)
    c3.metric("🎸 Genres", int(df["playlist_genre"].nunique()))
    c4.metric("🎼 Subgenres", int(df["playlist_subgenre"].nunique()))
    c5.metric("⚙️ Features", len(features))

    st.markdown("---")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("### 🎵 Sample Songs")
        st.dataframe(
            df[["track_name", "track_artist", "playlist_genre", "danceability", "energy", "valence", "cluster"]].head(12),
            use_container_width=True,
        )

    with right:
        st.markdown("### 🎸 Genre Distribution")
        fig = plot_genre_distribution(df)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("### 🔧 Pickle Bundle Contents")
    info = pd.DataFrame(
        {
            "Object": ["KMeans", "StandardScaler", "PCA", "LabelEncoder", "DataFrame"],
            "Type": [
                type(model.kmeans).__name__,
                type(model.scaler).__name__,
                type(model.pca).__name__,
                type(model.label_encoder).__name__,
                type(df).__name__,
            ],
            "Detail": [
                f"k={model.k}, inertia={getattr(model.kmeans, 'inertia_', 0):,.0f}",
                f"fit on {len(features)} features",
                f"2 components, variance {model.pca.explained_variance_ratio_.sum() * 100:.1f}%",
                f"classes: {list(model.label_encoder.classes_)}",
                f"{df.shape[0]:,} rows × {df.shape[1]} cols",
            ],
        }
    )
    st.dataframe(info, use_container_width=True, hide_index=True)


def render_eda_tab(model: ModelBundle) -> None:
    df = model.df
    features = model.audio_features
    st.markdown("## 📈 Exploratory Data Analysis")
    selected_feature = st.selectbox("Select feature to explore:", features, key="eda_feat")

    fig = plot_feature_distribution_by_genre(df, selected_feature)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### 🔥 Correlation Heatmap")
    fig = plot_correlation_heatmap(df, features)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### 🎸 Genre-wise Average Audio Features")
    fig = plot_genre_feature_means(df, features)
    st.pyplot(fig)
    plt.close(fig)


def render_clusters_tab(model: ModelBundle) -> None:
    df = model.df
    pca = model.pca
    st.markdown("## 🗺️ Cluster Visualization (PCA 2D)")

    v1, v2 = pca.explained_variance_ratio_[:2]
    m1, m2, m3 = st.columns(3)
    m1.metric("PC1 Variance", f"{v1 * 100:.1f}%")
    m2.metric("PC2 Variance", f"{v2 * 100:.1f}%")
    m3.metric("Total Explained", f"{(v1 + v2) * 100:.1f}%")

    color_by = st.radio("Color by:", ["🔵 Cluster", "🎸 Genre"], horizontal=True)
    max_points = max(500, min(15000, len(df)))
    default_points = min(5000, max_points)
    n_points = st.slider("Points to plot:", 500, max_points, default_points, 500)

    fig = plot_pca_clusters(df, pca, color_by, n_points)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### 📊 Cluster Sizes")
    fig = plot_cluster_sizes(df)
    st.pyplot(fig)
    plt.close(fig)


def render_predict_by_index_tab(model: ModelBundle) -> None:
    df = model.df
    features = model.audio_features
    profiles = model.cluster_profiles

    st.markdown("## 🔍 Predict Cluster by Song Index")
    st.markdown(
        """
        <div class="card">
        <b>How it works:</b><br>
        Enter a song index (row number from the dataset) →
        App extracts its audio features → Passes through <b>scaler → KMeans</b>
        (from pickle) → Returns cluster → Shows recommendations from same cluster.
        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "🔍 Search song name (optional — to find index):",
        placeholder="e.g. Blinding Lights",
    )
    if search:
        hits = (
            df[df["track_name"].str.contains(search, case=False, na=False)][
                ["track_name", "track_artist", "playlist_genre", "cluster"]
            ]
            .head(8)
            .reset_index()
        )
        if len(hits):
            st.dataframe(hits, use_container_width=True, hide_index=True)
        else:
            st.warning("No matches found.")

    st.markdown("---")
    song_idx = st.number_input(
        f"Enter Song Index (0 – {len(df) - 1:,}):",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    if st.button("🎯 Predict & Recommend", key="pred_idx"):
        row = df.iloc[int(song_idx)]
        feature_values = row[features].tolist()
        cluster_id = predict_cluster(model, feature_values)
        profile = profiles[cluster_id]
        badge = CLUSTER_PAL[cluster_id % len(CLUSTER_PAL)]

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#1db95415,#0d0d1a);border:2px solid #1db954;border-radius:16px;padding:20px 26px;margin:14px 0;">
              <h2 style="color:#1db954;margin:0 0 4px">🎵 {safe_text(row.get('track_name', '—'))}</h2>
              <p style="color:#888;margin:0 0 12px">by <b style="color:#ccc">{safe_text(row.get('track_artist', '—'))}</b></p>
              <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:.86rem;">
                <span style="background:#1db95425;color:#1db954;padding:3px 14px;border-radius:20px;font-weight:700;">🎸 {safe_text(row.get('playlist_genre', '—'))}</span>
                <span style="background:{badge}25;color:{badge};padding:3px 14px;border-radius:20px;font-weight:700;">🔵 Cluster {cluster_id}</span>
                <span style="color:#888;">💃 dance {row.get('danceability', 0):.2f} &nbsp;|&nbsp; ⚡ energy {row.get('energy', 0):.2f} &nbsp;|&nbsp; 😊 valence {row.get('valence', 0):.2f} &nbsp;|&nbsp; 🥁 tempo {row.get('tempo', 0):.0f} BPM</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card">
            <b>Cluster {cluster_id} Profile ({profile['vibe']})</b><br>
            Dominant genre: <b>{profile['dominant_genre']}</b> &nbsp;|&nbsp;
            Avg energy: <b>{profile['avg_energy']}</b> &nbsp;|&nbsp;
            Avg danceability: <b>{profile['avg_danceability']}</b> &nbsp;|&nbsp;
            Avg valence: <b>{profile['avg_valence']}</b> &nbsp;|&nbsp;
            Size: <b>{profile['size']:,} songs</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        recs = get_recommendations(df, cluster_id, exclude_idx=int(song_idx))
        render_recommendations(recs, "🎯 Top 5 Similar Songs")


def render_predict_by_features_tab(model: ModelBundle) -> None:
    df = model.df
    features = model.audio_features
    profiles = model.cluster_profiles

    st.markdown("## 🎛️ Predict Cluster by Custom Audio Features")
    st.markdown(
        """
        <div class="card">
        Manually set audio features using sliders → model will predict which cluster
        your hypothetical song belongs to → then show real similar songs from that cluster!
        </div>
        """,
        unsafe_allow_html=True,
    )

    defaults = build_manual_feature_input(df, features)
    columns = st.columns(3)
    feature_values: list[float] = []

    for i, feature in enumerate(features):
        min_value, max_value, default_value, step = feature_slider_spec(feature, defaults[feature])
        with columns[i % 3]:
            value = st.slider(
                feature_label(feature),
                min_value,
                max_value,
                float(default_value),
                float(step),
            )
        feature_values.append(value)

    if st.button("🤖 Predict Cluster", key="pred_manual"):
        cluster_id = predict_cluster(model, feature_values)
        profile = profiles[cluster_id]
        color = CLUSTER_PAL[cluster_id % len(CLUSTER_PAL)]

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{color}15,#0d0d1a);border:2px solid {color};border-radius:16px;padding:22px 28px;margin:16px 0;text-align:center;">
              <h2 style="color:{color};margin:0 0 6px;">🔵 Cluster {cluster_id} — {profile['vibe']}</h2>
              <p style="color:#888;margin:0;">
                Dominant Genre: <b style="color:#ccc">{profile['dominant_genre']}</b> &nbsp;|&nbsp;
                Avg Energy: <b style="color:#ccc">{profile['avg_energy']}</b> &nbsp;|&nbsp;
                Avg Popularity: <b style="color:#ccc">{profile['avg_popularity']}</b> &nbsp;|&nbsp;
                Cluster Size: <b style="color:#ccc">{profile['size']:,} songs</b>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📊 Your Input vs Cluster Average")
        fig = plot_input_vs_cluster_center(
            features,
            model.scaler,
            model.kmeans,
            cluster_id,
            feature_values,
        )
        st.pyplot(fig)
        plt.close(fig)

        recs = get_recommendations(df, cluster_id)
        render_recommendations(recs, "🎯 Songs Similar to Your Input")


def render_insights_tab(model: ModelBundle) -> None:
    df = model.df
    profiles = model.cluster_profiles
    st.markdown("## 💡 Cluster Insights")

    for cluster_id in range(model.k):
        profile = profiles[cluster_id]
        color = CLUSTER_PAL[cluster_id % len(CLUSTER_PAL)]
        genre_counts = df[df["cluster"] == cluster_id]["playlist_genre"].value_counts()
        top3 = ", ".join(
            f"{genre} ({count / genre_counts.sum() * 100:.0f}%)"
            for genre, count in genre_counts.head(3).items()
        )
        st.markdown(
            f"""
            <div style="background:#161630;border:1px solid {color}40;border-left:5px solid {color};border-radius:14px;padding:18px 24px;margin-bottom:14px;">
              <h3 style="color:{color};margin:0 0 8px;">🔵 Cluster {cluster_id} — {profile['vibe']}</h3>
              <div style="display:flex;gap:28px;flex-wrap:wrap;color:#c8c8d8;font-size:.9rem;">
                <span>📦 <b>Size:</b> {profile['size']:,} songs</span>
                <span>🎸 <b>Top genres:</b> {top3}</span>
                <span>💃 <b>Danceability:</b> {profile['avg_danceability']}</span>
                <span>⚡ <b>Energy:</b> {profile['avg_energy']}</span>
                <span>😊 <b>Valence:</b> {profile['avg_valence']}</span>
                <span>🥁 <b>Tempo:</b> {profile['avg_tempo']} BPM</span>
                <span>⭐ <b>Popularity:</b> {profile['avg_popularity']}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🎸 Genre Mix per Cluster")
    fig = plot_cluster_genre_mix(df, model.k)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### 📋 Full Dataset with Cluster Labels")
    show_cols = [
        "track_name",
        "track_artist",
        "playlist_genre",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "track_popularity",
        "cluster",
    ]
    st.dataframe(df[show_cols].head(200), use_container_width=True)
    csv_data = df[show_cols].to_csv(index=False)
    st.download_button(
        "📥 Download Full Dataset with Clusters",
        data=csv_data,
        file_name="spotify_clustered.csv",
        mime="text/csv",
    )