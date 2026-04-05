from __future__ import annotations

import warnings

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

st.set_page_config(
    page_title="🎵 Spotify Segmentation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()
pkl_file = render_sidebar()
render_hero()

if pkl_file is None:
    render_quick_start()
    st.stop()

try:
    with st.spinner("Loading model…"):
        model = load_model_bundle(pkl_file.read())
except Exception as exc:
    st.error(f"Failed to load model.pkl: {exc}")
    st.stop()

st.success(f"✅ Model loaded — **K={model.k}** clusters · **{len(model.df):,} songs**")

tabs = st.tabs([
    "📊 Overview",
    "📈 EDA",
    "🗺️ Clusters",
    "🔍 Predict by Index",
    "🎛️ Predict by Features",
    "💡 Insights",
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