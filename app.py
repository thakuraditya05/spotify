from __future__ import annotations

import warnings
from pathlib import Path

import streamlit as st

from ui import (
    apply_app_styling,
    render_clusters_tab,
    render_eda_tab,
    render_hero,
    render_insights_tab,
    render_overview_tab,
    render_predict_by_features_tab,
    render_predict_by_index_tab,
    render_quick_start,
    render_sidebar,
)
from utils import load_model_bundle

warnings.filterwarnings("ignore")

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")

st.set_page_config(
    page_title="Spotify Segmentation",
    page_icon=":musical_note:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()
render_sidebar(MODEL_PATH)
render_hero()

if not MODEL_PATH.exists():
    render_quick_start(MODEL_PATH, file_missing=True)
    st.stop()

try:
    with st.spinner("Loading model..."):
        model = load_model_bundle(MODEL_PATH.read_bytes())
except Exception as exc:
    st.error(f"Failed to load model.pkl: {exc}")
    st.stop()

st.success(
    f"Model loaded from `{MODEL_PATH.name}` - "
    f"**K={model.k}** clusters, **{len(model.df):,} songs**"
)

tabs = st.tabs([
    "Overview",
    "EDA",
    "Clusters",
    "Predict by Index",
    "Predict by Features",
    "Insights",
])

with tabs[0]:
    render_overview_tab(model)
with tabs[1]:
    render_eda_tab(model)
with tabs[2]:
    render_clusters_tab(model)
with tabs[3]:
    render_predict_by_index_tab(model)
with tabs[4]:
    render_predict_by_features_tab(model)
with tabs[5]:
    render_insights_tab(model)
