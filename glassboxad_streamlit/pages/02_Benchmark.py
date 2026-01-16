from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from core.benchmark_loader import load_tsb_ad_benchmark

try:
    from scipy.stats import friedmanchisquare
    import scikit_posthocs as sp
    HAS_CD = True
except Exception:
    HAS_CD = False


st.set_page_config(page_title="Benchmark", layout="wide")

st.title("Benchmark")

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@st.cache_data(show_spinner=False)
def _load(mode: str):
    return load_tsb_ad_benchmark(DATA_ROOT, mode=mode)


@st.cache_data(show_spinner=False)
def _load_lengths(data_root: Path) -> pd.DataFrame:
    """Load per-series length mapping.

    Expected file (preferred):
      - data/series_length.csv

    Accepted schemas:
      (A) columns: [series_id, length]
      (B) index is series_id and a column named length
    """
    candidates = [
        data_root / "series_length.csv",
        data_root / "series_lengths.csv",
        data_root / "length.csv",
        data_root / "lengths.csv",
        data_root / "ts_length.csv",
    ]
    p = None
    for c in candidates:
        if c.exists():
            p = c
            break
    if p is None:
        # try to auto-discover
        for c in data_root.glob("*.csv"):
            if "length" in c.stem.lower():
                p = c
                break
    if p is None:
        return pd.DataFrame(columns=["series_id", "length"])

    df = pd.read_csv(p)
    # schema A
    if "series_id" in df.columns and "length" in df.columns:
        out = df[["series_id", "length"]].copy()
    else:
        # schema B
        if "length" in df.columns:
            # assume first unnamed/index-like column is series_id
            sid_col = "series_id" if "series_id" in df.columns else df.columns[0]
            out = df[[sid_col, "length"]].copy()
            out = out.rename(columns={sid_col: "series_id"})
        else:
            out = pd.DataFrame(columns=["series_id", "length"])

    out["series_id"] = out["series_id"].astype(str)
    out["length"] = pd.to_numeric(out["length"], errors="coerce")
    return out.dropna(subset=["length"])


with st.sidebar:
    st.header("Benchmark")
    mode = st.radio("Data", ["uni", "multi"], index=0, horizontal=True)

    try:
        eval_long, runtime_df = _load(mode)
    except Exception as e:
        st.error(str(e))
        st.stop()

    metrics = sorted(eval_long["metric"].dropna().unique().tolist())
    default_metric = "VUS-PR" if "VUS-PR" in metrics else (metrics[0] if metrics else None)
    metric = st.selectbox("Metric", metrics, index=metrics.index(default_metric) if default_metric in metrics else 0)

    datasets = sorted(eval_long["dataset"].dropna().unique().tolist())
    domains = sorted(eval_long["domain"].dropna().unique().tolist())
    methods = sorted(eval_long["method"].dropna().unique().tolist())

    sel_datasets = st.multiselect("Datasets", datasets, default=datasets)
    sel_domains = st.multiselect("Domains", domains, default=domains)
    sel_methods = st.multiselect("Algorithms", methods, default=methods)


flt = eval_long[
    eval_long["dataset"].isin(sel_datasets)
    & eval_long["domain"].isin(sel_domains)
    & eval_long["method"].isin(sel_methods)
    & (eval_long["metric"] == metric)
].copy()

if flt.empty:
    st.warning("No rows after filtering.")
    st.stop()

# Each series_id is treated as a test case for ranking.
wide = flt.pivot_table(index="series_id", columns="method", values="value", aggfunc="mean")
wide = wide.dropna(axis=1, how="all")

st.subheader("Runtime")
rt = runtime_df[
    runtime_df["dataset"].isin(sel_datasets)
    & runtime_df["domain"].isin(sel_domains)
    & runtime_df["method"].isin(sel_methods)
].copy()

len_df = _load_lengths(DATA_ROOT)

if rt.empty:
    st.info("No runtime rows after filtering.")
elif len_df.empty:
    st.info("Missing per-series length mapping CSV (e.g., data/series_length.csv).")
else:
    rt = rt.merge(len_df, on="series_id", how="left")
    rt = rt.dropna(subset=["length", "runtime"])
    if rt.empty:
        st.info("No rows after merging runtime with length mapping.")
    else:
        # Smooth length-runtime curve by quantile-binning lengths (reduces clutter when there are many series).
        # We render this as a Plotly figure for interactive zoom/pan + legend toggling.

        # UI controls (kept here, not in sidebar)
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            use_log = st.toggle("Log y-axis", value=True, help="Helps compare fast methods when a few methods are very slow.")
        with col_b:
            show_all = st.toggle("Show all methods", value=False, help="If off, we show only the fastest methods to reduce clutter.")
        with col_c:
            top_k = st.slider("Fastest methods to show", min_value=5, max_value=max(5, int(rt['method'].nunique())), value=min(12, int(rt['method'].nunique())), step=1, disabled=show_all)

        # Optionally filter to the fastest methods (based on overall mean runtime)
        methods_sorted = rt.groupby('method')['runtime'].mean().sort_values().index.tolist()
        if not show_all and len(methods_sorted) > top_k:
            keep = set(methods_sorted[:top_k])
            rt_view = rt[rt['method'].isin(keep)].copy()
        else:
            rt_view = rt.copy()

        rows = []
        for m, sub in rt_view.groupby('method', sort=False):
            sub = sub.dropna(subset=['length', 'runtime']).sort_values('length')
            if sub.empty:
                continue

            n = len(sub)
            nbins = int(np.clip(n // 20, 8, 30))  # ~20 series per bin, between 8 and 30 bins

            if n < 40 or sub['length'].nunique() < 8:
                g = sub.groupby('length', as_index=False)['runtime'].mean().sort_values('length')
                rows.extend({
                    'method': m,
                    'length': float(L),
                    'runtime': float(R),
                } for L, R in zip(g['length'].to_numpy(), g['runtime'].to_numpy()))
                continue

            try:
                bins = pd.qcut(sub['length'], q=nbins, duplicates='drop')
                g = (
                    sub.groupby(bins)
                    .agg(length_mean=('length', 'mean'), runtime_mean=('runtime', 'mean'))
                    .sort_values('length_mean')
                )
                rows.extend({
                    'method': m,
                    'length': float(L),
                    'runtime': float(R),
                } for L, R in zip(g['length_mean'].to_numpy(), g['runtime_mean'].to_numpy()))
            except Exception:
                g = sub.groupby('length', as_index=False)['runtime'].mean().sort_values('length')
                rows.extend({
                    'method': m,
                    'length': float(L),
                    'runtime': float(R),
                } for L, R in zip(g['length'].to_numpy(), g['runtime'].to_numpy()))

        smooth_df = pd.DataFrame(rows)
        if smooth_df.empty:
            st.info('Not enough data to build a smoothed runtime curve.')
        else:
            fig_rt = px.line(
                smooth_df,
                x='length',
                y='runtime',
                color='method',
                title='Length vs Runtime (smoothed)',
            )
            fig_rt.update_layout(
                xaxis_title='Time series length',
                yaxis_title='Runtime (Time column)',
                legend_title_text='Method',
                margin=dict(l=10, r=10, t=40, b=10),
                height=520,
            )
            if use_log:
                # Avoid log(0). If zeros exist, Plotly will ignore; users can still toggle to linear.
                fig_rt.update_yaxes(type='log')
            st.plotly_chart(fig_rt, use_container_width=True)


st.divider()

st.subheader(f"Evaluation: {metric}")

# Colored boxplot, sorted by mean (descending)
box_df = flt[["method", "value"]].dropna().copy()
order = (
    box_df.groupby("method")["value"].mean().sort_values(ascending=False).index.tolist()
)

fig_box = px.box(
    box_df,
    x="value",
    y="method",
    color="method",
    orientation="h",
    points=False,
    category_orders={"method": order},
)
fig_box.update_layout(
    xaxis_title=metric,
    yaxis_title="",
    showlegend=False,
    height=max(450, 22 * max(1, len(order)) + 120),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("**Leaderboard (Algorithms × Metrics)**")
leader_src = eval_long[
    eval_long["dataset"].isin(sel_datasets)
    & eval_long["domain"].isin(sel_domains)
    & eval_long["method"].isin(sel_methods)
].copy()

leader = leader_src.pivot_table(index="method", columns="metric", values="value", aggfunc="mean")

# Order columns: put VUS-PR first when available, then alphabetical for stability
cols = list(leader.columns)
if "VUS-PR" in cols:
    cols = ["VUS-PR"] + [c for c in cols if c != "VUS-PR"]
cols = [cols[0]] + sorted(cols[1:]) if cols else cols
leader = leader.reindex(columns=cols)

# Sort rows by the selected metric (default VUS-PR)
sort_col = metric if metric in leader.columns else ("VUS-PR" if "VUS-PR" in leader.columns else leader.columns[0])
leader_sorted = leader.sort_values(by=sort_col, ascending=False)

st.caption(f"Sorted by: {sort_col} (higher is better)")

# Bold the best (max) score for each metric column.
def _bold_best_per_column(col: pd.Series):
    if col.dropna().empty:
        return ["" for _ in col]
    m = col.max(skipna=True)
    return ["font-weight: bold" if pd.notna(v) and v == m else "" for v in col]

styled = leader_sorted.style.format("{:.4f}").apply(_bold_best_per_column, axis=0)
st.dataframe(styled, use_container_width=True)

st.markdown("---")
st.subheader("Critical Diagram (Friedman + Nemenyi)")

if not HAS_CD:
    st.info("CD dependencies missing (scipy/scikit-posthocs). Install them to enable the CD plot.")
else:
    # When many methods are selected, the CD diagram becomes unreadable.
    # Default to showing only the best methods (by the currently selected metric),
    # and size the figure by the number of shown methods.

    cd_show_all = st.checkbox("Show all selected methods in CD diagram", value=False)
    st.caption("Tip: the CD diagram gets cluttered quickly—showing the top-k methods is usually more readable.")

    cd_k = st.slider(
        "Methods shown in CD diagram (top-k)",
        min_value=6,
        max_value=max(6, len(sel_methods)),
        value=min(10, max(6, len(sel_methods))),
        step=1,
        disabled=cd_show_all,
    )

    cd_methods = list(leader_sorted.index)  # already filtered by datasets/domains/methods
    if not cd_show_all:
        cd_methods = cd_methods[:cd_k]

    wide_cd = wide[[m for m in cd_methods if m in wide.columns]].copy()
    # Drop datasets (rows) with missing values across shown methods to avoid NaNs in tests.
    wide_cd = wide_cd.dropna(axis=0, how="any")

    if wide_cd.shape[0] < 2 or wide_cd.shape[1] < 3:
        st.warning("Not enough complete data to compute the CD diagram after filtering (need >=2 series and >=3 methods).")
    else:
        # 1 = best because higher is better
        ranks = wide_cd.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean(axis=0).sort_values(ascending=True)

        # Friedman test
        try:
            arrays = [wide_cd[c].values for c in wide_cd.columns]
            _stat, p = friedmanchisquare(*arrays)
        except Exception:
            p = float("nan")

        st.write(f"Friedman p-value: {p:.4g}" if pd.notna(p) else "Friedman p-value: n/a")

        try:
            pvals = sp.posthoc_nemenyi_friedman(wide_cd)
        except Exception:
            pvals = sp.posthoc_nemenyi_friedman(wide_cd.to_numpy().T)

        n_methods = wide_cd.shape[1]
        fig_w = 15
        fig_h = max(4.0, 0.48 * n_methods + 2.0)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
        ax = plt.gca()
        sp.critical_difference_diagram(avg_ranks, pvals, ax=ax)
        ax.set_title("Critical Difference Diagram")

        # Lighten / thin connector lines to reduce visual clutter.
        for ln in ax.lines:
            try:
                c = str(ln.get_color()).lower()
                if c in ("k", "black", "#000", "#000000"):
                    ln.set_alpha(0.45)
                    ln.set_linewidth(0.8)
                else:
                    ln.set_alpha(0.85)
                    ln.set_linewidth(1.2)
            except Exception:
                pass

        # Make labels smaller when many methods are displayed.
        fs = 10 if n_methods <= 14 else (9 if n_methods <= 20 else 8)
        for t in ax.texts:
            try:
                t.set_fontsize(fs)
            except Exception:
                pass

        try:
            fig.tight_layout()
        except Exception:
            pass

        st.pyplot(fig, use_container_width=True)
