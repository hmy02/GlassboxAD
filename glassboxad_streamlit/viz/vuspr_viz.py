from __future__ import annotations

import json
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from core.vuspr_backend import (
    vus_pr_at_buffer,
    threshold_point_on_slice,
    contiguous_segments,
    expanded_segments_from_labels,
)


def make_vuspr_3d_figure(surface: dict, buffer_size: int, threshold_value: float) -> go.Figure:
    window = np.asarray(surface["window"]).astype(int)
    recall = np.asarray(surface["recall"]).astype(float)
    precision = np.asarray(surface["precision"]).astype(float)

    # 2D grids for plotly surface
    X = np.tile(window[:, None], (1, recall.shape[1]))
    Y = recall
    Z = precision

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            opacity=0.65,
            showscale=False,
            name="VUS-PR Surface",
        )
    )

    # Slice at (nearest) buffer_size
    sl = vus_pr_at_buffer(surface, buffer_size)
    b = int(sl["buffer"])
    r_b = np.asarray(sl["recall"], dtype=float)
    p_b = np.asarray(sl["precision"], dtype=float)

    fig.add_trace(
        go.Scatter3d(
            x=np.full_like(r_b, b, dtype=float),
            y=r_b,
            z=p_b,
            mode="lines",
            name=f"Slice (buffer={b})",
        )
    )

    # Curtain surface for the slice (to emphasize the cut)
    fig.add_trace(
        go.Surface(
            x=np.vstack([np.full_like(r_b, b, dtype=float), np.full_like(r_b, b, dtype=float)]),
            y=np.vstack([r_b, r_b]),
            z=np.vstack([p_b, np.zeros_like(p_b)]),
            opacity=0.18,
            showscale=False,
            name="Slice curtain",
        )
    )

    # Threshold point
    pt = threshold_point_on_slice(surface, b, threshold_value)
    fig.add_trace(
        go.Scatter3d(
            x=[float(b)],
            y=[pt["recall"]],
            z=[pt["precision"]],
            mode="markers",
            marker=dict(size=9),
            name=f"Threshold={pt['threshold']:.3f}",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=35, b=0),
        scene=dict(
            xaxis_title="Buffer Size",
            yaxis_title="Recall",
            zaxis_title="Precision",
        ),
        title=(
            "Volume Under Surface | VUS-PR = "
            f"{float(surface.get('avg_vus_pr', surface.get('avg_vuspr', float('nan')))):.4f}"
        ),
        height=520,
    )
    return fig


def make_pr_slice_figure(surface: dict, buffer_size: int, threshold_value: float) -> go.Figure:
    sl = vus_pr_at_buffer(surface, buffer_size)
    pt = threshold_point_on_slice(surface, buffer_size, threshold_value)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sl["recall"],
            y=sl["precision"],
            mode="lines+markers",
            marker=dict(size=3),
            name="PR curve",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pt["recall"]],
            y=[pt["precision"]],
            mode="markers",
            marker=dict(size=12),
            name="threshold point",
        )
    )

    fig.update_layout(
        title=f"AUC-PR | Window Size = {sl['buffer']} | AUC-PR = {sl['vus_pr']:.4f}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(l=10, r=10, t=45, b=10),
        height=520,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def make_score_timeline_figure(
    score: np.ndarray,
    label: Optional[np.ndarray],
    threshold_value: float,
    buffer_size: int,
) -> go.Figure:
    score = np.asarray(score).astype(float)

    mn = np.nanmin(score)
    mx = np.nanmax(score)
    denom = mx - mn
    if not np.isfinite(denom) or denom < 1e-12:
        score = np.zeros_like(score, dtype=float)
    else:
        score = (score - mn) / (denom + 1e-12)

    x = np.arange(len(score))
    thr = float(threshold_value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=score, mode="lines", name="anomaly score"))
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[thr, thr], mode="lines", name="threshold"))

    y0 = 0
    y1 = 1

    # Label==1 segments (red)
    if label is not None:
        label = np.asarray(label).astype(int)
        gt_segs = contiguous_segments(label > 0)
        for s, e in gt_segs:
            fig.add_shape(
                type="rect",
                x0=s,
                x1=e,
                y0=y0,
                y1=y1,
                fillcolor="red",
                opacity=0.15,
                line_width=0,
            )

        # Buffer expansion trapezoids
        exp = expanded_segments_from_labels(label, buffer_size)
        for s, e, le, re in exp:
            # Trapezoid: (le,y0)->(s,y1)->(e,y1)->(re,y0)
            path = f"M {le},{y0} L {s},{y1} L {e},{y1} L {re},{y0} Z"
            fig.add_shape(
                type="path",
                path=path,
                fillcolor="rgba(255,165,0,0.18)",
                line=dict(width=0),
            )

    # Predicted segments (score >= thr) as light blue rectangles
    pred = score >= thr
    pred_segs = contiguous_segments(pred)
    for s, e in pred_segs:
        fig.add_shape(
            type="rect",
            x0=s,
            x1=e,
            y0=y0,
            y1=y1,
            fillcolor="rgba(0,120,255,0.10)",
            line_width=0,
        )

    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig
