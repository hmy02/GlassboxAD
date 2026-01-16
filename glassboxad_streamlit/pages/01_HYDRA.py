from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.hydra_backend import auto_window_size, compute_hydra_payload, load_ts_csv
from core.vuspr_backend import compute_vus_pr_surface
from viz.hydra_embed import build_hydra_viewer_html
from viz.vuspr_viz import make_vuspr_3d_figure, make_pr_slice_figure, make_score_timeline_figure

st.set_page_config(page_title="HYDRA", layout="wide")

st.title("HYDRA")

DATA_DIR = Path("data/preloaded_ts")

with st.sidebar:
    st.header("Input")
    if DATA_DIR.exists():
        csvs = sorted([p.name for p in DATA_DIR.glob("*.csv")])
    else:
        csvs = []

    source = st.radio("Choose source", ["preloaded_ts", "upload"], horizontal=True)

    df = None
    data_name = None

    if source == "preloaded_ts":
        if not csvs:
            st.warning("No CSV files found in data/preloaded_ts")
        fname = st.selectbox("CSV", csvs) if csvs else None
        if fname:
            data_name = fname
            df = pd.read_csv(DATA_DIR / fname)
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            data_name = up.name
            df = pd.read_csv(up)

    st.header("HYDRA params")
    default_win = 30
    auto_win = st.checkbox("Auto window size", value=False)
    win_size = st.number_input("Window size", min_value=4, max_value=2000, value=default_win, step=1)
    mode = st.selectbox("Mode", ["approx", "exact"], index=0)

    st.header("VUS-PR params")
    max_buffer = st.number_input("Max buffer size (surface)", min_value=1, max_value=5000, value=200, step=1)

run = st.button("Run HYDRA", type="primary", disabled=(df is None))

if df is None:
    st.info("Select a CSV from data/preloaded_ts or upload one. Supported columns: Data/Label (recommended), or any first numeric column.")
    st.stop()

# Flexible loading: prefer (Data, Label) but fall back to first numeric column
data_col = None
for c in ["Data", "data", "value", "Value", "signal", "Signal"]:
    if c in df.columns:
        data_col = c
        break
if data_col is None:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.error("No numeric column found in the selected CSV.")
        st.stop()
    data_col = numeric_cols[0]

label_col = None
for c in ["Label", "label", "y", "Y", "is_anomaly", "anomaly"]:
    if c in df.columns:
        label_col = c
        break

x = df[data_col].astype(float).to_numpy()
labels = df[label_col].astype(int).to_numpy() if label_col is not None else None

if auto_win:
    win_size = auto_window_size(x, default=int(win_size))

st.write(f"**Dataset:** {data_name}  |  **Length:** {len(x)}  |  **Window:** {int(win_size)}")

if run:
    try:
        payload = compute_hydra_payload(x=x, labels=labels, win_size=int(win_size), mode=mode)
        st.session_state["hydra_payload"] = payload
        # also store ensemble score for VUS-PR
        st.session_state["hydra_score"] = np.asarray(payload["ts_score_ens"], dtype=float)
    except Exception as e:
        st.exception(e)
        st.stop()

if "hydra_payload" in st.session_state:
    payload = st.session_state["hydra_payload"]
    st.subheader("HYDRA Hierarchy Viewer")
    html = build_hydra_viewer_html(payload)
    st.components.v1.html(html, height=820, scrolling=False)

    st.divider()
    st.subheader("Scores")
    score = st.session_state.get("hydra_score")
    if score is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=score, mode="lines"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # VUS-PR dialog
    if labels is not None and score is not None:
        @st.dialog("VUS-PR Explorer", width="large")
        def vuspr_dialog():
            st.caption("Adjust threshold/buffer inside this dialog. Surface is computed once per max-buffer setting.")

            # Controls live inside the dialog
            ctrl1, ctrl2 = st.columns([1, 1])
            with ctrl1:
                thr = st.slider("Threshold (0–1)", 0.0, 1.0, 0.40, 0.01, key="vuspr_thr")
            with ctrl2:
                buffer_size = st.slider(
                    "Buffer size (slice)",
                    0,
                    int(max_buffer),
                    min(100, int(max_buffer)),
                    1,
                    key="vuspr_buffer",
                )

            # Cache surface across interactions inside the dialog
            cache_key = f"vuspr_surface::{data_name or 'data'}::{int(max_buffer)}"
            need_recompute = (
                cache_key not in st.session_state
                or st.session_state.get(cache_key + "::n") != int(len(score))
            )
            if need_recompute:
                with st.spinner("Computing VUS-PR surface..."):
                    def minmax_01(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
                        a = np.asarray(a, dtype=float)
                        mn = np.nanmin(a)
                        mx = np.nanmax(a)
                        denom = mx - mn
                        if not np.isfinite(denom) or denom < eps:
                            return np.zeros_like(a, dtype=float)
                        return (a - mn) / (denom + eps)
                    scores_01 = minmax_01(score)
                    surface = compute_vus_pr_surface(labels, scores_01, max_buffer_size=int(max_buffer), thre=250, version="opt")
                st.session_state[cache_key] = surface
                st.session_state[cache_key + "::n"] = int(len(score))
            else:
                surface = st.session_state[cache_key]

            c1, c2 = st.columns([1.1, 1.0], gap="large")
            with c1:
                st.plotly_chart(make_vuspr_3d_figure(surface, int(buffer_size), float(thr)), use_container_width=True)
            with c2:
                st.plotly_chart(make_pr_slice_figure(surface, int(buffer_size), float(thr)), use_container_width=True)

            st.plotly_chart(make_score_timeline_figure(score, labels, float(thr), int(buffer_size)), use_container_width=True)

        if st.button("Open VUS-PR Explorer"):
            vuspr_dialog()

    else:
        st.info("Label column not found; VUS-PR explorer requires ground-truth labels.")
else:
    st.info("Click **Run HYDRA** to compute the hierarchy and scores.")
