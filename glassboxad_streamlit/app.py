import streamlit as st
from pathlib import Path

st.set_page_config(page_title="GlassboxAD", layout="wide")

st.title("GlassboxAD")

st.info(
    "**Note.** HYDRA is still under review, so the results shown in this demo may not be up to date. "
    "This application is intended to showcase the system design and interactive analysis workflow."
)

st.subheader("Abstract")
st.markdown(
    """
Time-series anomaly detection (TSAD) is challenging in unsupervised settings because anomalies are heterogeneous and often manifest at different temporal scales.
HYDRA addresses this by constructing a multi-level hierarchy of representative subsequences, computing reference-driven anomaly scores at each level, and aggregating them into a final score.
In this paper, we present **GlassboxAD**, an interactive demo system that makes HYDRA’s hierarchical behavior observable and interpretable.
The system supports two entry points: instant browsing of benchmark results on **TSB-AD** and on-demand analysis of a selected time series.
Users can inspect how detections emerge across levels and assess robustness under different tolerance and threshold settings through linked views, including layered visualizations of representatives, per-layer evidence for selected subsequences, score-layer switching on the timeline, and TSAD-specific evaluation metrics.
"""
)

st.subheader("HYDRA")
st.markdown(
    """
HYDRA is a hierarchical detector for subsequence anomalies.
Instead of committing to a single resolution, it builds a multi-level hierarchy of representative subsequences, scores each window by its distance to level-specific references, and then aggregates evidence across levels.
This design helps users understand *where* anomalies come from (which level, which reference) and supports interactive inspection of level-wise behavior.
"""
)

ASSET_DIR = Path(__file__).resolve().parent / "assets"
PIPELINE_IMG = ASSET_DIR / "HiAD_framework.png"
left, mid, right = st.columns([2, 6, 2])
with mid:
    st.image(str(PIPELINE_IMG), use_container_width=True, caption="HYDRA pipeline overview")

st.subheader("Benchmark")
st.markdown(
    """
The benchmark tab is built on **TSB-AD**.
It supports dataset/domain filtering, distributional comparisons (boxplots), statistically grounded ranking summaries (critical diagram), multi-metric leaderboards, and runtime scaling with series length.
"""
)

st.subheader("Contributors")
st.markdown(
    """
- Mingyi Huang
- Qinghua Liu
- Paul Boniol
- John Paparrizos
"""
)
