from __future__ import annotations

import numpy as np

from .basic_metrics import basic_metricor


def compute_vus_pr_surface(
    label: np.ndarray,
    score: np.ndarray,
    max_buffer_size: int,
    thre: int = 250,
    version: str = "opt",
) -> dict:
    """Compute the range-based PR surface used for VUS-PR visualization."""

    label = np.asarray(label, dtype=int).ravel()
    score = np.asarray(score, dtype=float).ravel()

    if label.shape[0] != score.shape[0]:
        raise ValueError(f"label and score must have the same length, got {label.shape[0]} vs {score.shape[0]}")
    if max_buffer_size < 0:
        raise ValueError("max_buffer_size must be non-negative")
    if thre < 10:
        raise ValueError("thre is too small; please use at least 10")

    metricor = basic_metricor()
    if version == "opt_mem":
        tpr_3d, _fpr_3d, prec_3d, window_3d, _avg_auc_3d, avg_ap_3d = metricor.RangeAUC_volume_opt_mem(
            labels_original=label,
            score=score,
            windowSize=int(max_buffer_size),
            thre=int(thre),
        )
    else:
        tpr_3d, _fpr_3d, prec_3d, window_3d, _avg_auc_3d, avg_ap_3d = metricor.RangeAUC_volume_opt(
            labels_original=label,
            score=score,
            windowSize=int(max_buffer_size),
            thre=int(thre),
        )

    tpr_3d = np.asarray(tpr_3d, dtype=float)
    prec_3d = np.asarray(prec_3d, dtype=float)
    window = np.asarray(window_3d, dtype=int).ravel()

    # Align recall and precision for PR: recall has one more endpoint in RangeAUC; drop the last (1,1) endpoint.
    recall = tpr_3d[:, :-1]          # (W, thre+1)
    precision = prec_3d              # (W, thre+1)

    # Threshold values used in RangeAUC_volume_opt: score_sorted[linspace(...)].
    score_sorted = -np.sort(-score)
    idxs = np.linspace(0, len(score) - 1, int(thre)).astype(int)
    thresholds = score_sorted[idxs]  # (thre,)

    # AP for each window (matches the implementation in basic_metrics.py)
    ap_per_window = np.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], axis=1)

    return {
        "window": window,
        "recall": recall,
        "precision": precision,
        "thresholds": thresholds,
        "ap_per_window": ap_per_window,
        # Backward/forward compatible key names (viz expects avg_vus_pr)
        "avg_vus_pr": float(avg_ap_3d),
        "avg_vuspr": float(avg_ap_3d),
    }


def pick_threshold_point(thresholds: np.ndarray, recall_row: np.ndarray, precision_row: np.ndarray, thr: float) -> tuple[float, float, int]:
    """Return (recall, precision, idx) for the threshold point.

    `thresholds` has length `thre`, corresponding to recall_row[1:], precision_row[1:].
    """
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    recall_row = np.asarray(recall_row, dtype=float).ravel()
    precision_row = np.asarray(precision_row, dtype=float).ravel()

    if thresholds.ndim != 1:
        raise ValueError("thresholds must be 1D")
    if recall_row.shape[0] != precision_row.shape[0]:
        raise ValueError("recall_row and precision_row must have same length")
    if recall_row.shape[0] != thresholds.shape[0] + 1:
        raise ValueError("recall/precision rows must be length len(thresholds)+1")

    idx = int(np.argmin(np.abs(thresholds - float(thr))))
    r = float(recall_row[idx + 1])
    p = float(precision_row[idx + 1])
    return r, p, idx


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive segments where mask is True."""
    mask = np.asarray(mask).astype(bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _nearest_window_index(window: np.ndarray, buffer_size: int) -> int:
    window = np.asarray(window, dtype=int).ravel()
    if window.size == 0:
        raise ValueError("Empty window grid")
    if int(buffer_size) in set(window.tolist()):
        return int(np.where(window == int(buffer_size))[0][0])
    # fall back to nearest
    return int(np.argmin(np.abs(window - int(buffer_size))))


def vus_pr_at_buffer(surface: dict, buffer_size: int) -> dict:
    """Return PR slice arrays and AUC-PR (area under PR curve) at a given buffer size."""
    window = np.asarray(surface["window"], dtype=int)
    recall = np.asarray(surface["recall"], dtype=float)
    precision = np.asarray(surface["precision"], dtype=float)
    ap = np.asarray(surface["ap_per_window"], dtype=float)

    idx = _nearest_window_index(window, buffer_size)
    b = int(window[idx])
    return {
        "buffer": b,
        "recall": recall[idx].ravel(),
        "precision": precision[idx].ravel(),
        "vus_pr": float(ap[idx]) if ap.size else float("nan"),
        "row_idx": idx,
    }


def threshold_point_on_slice(surface: dict, buffer_size: int, threshold_value: float) -> dict:
    """Pick the closest threshold point on the PR slice."""
    sl = vus_pr_at_buffer(surface, buffer_size)
    thresholds = np.asarray(surface["thresholds"], dtype=float)
    r, p, idx = pick_threshold_point(thresholds, sl["recall"], sl["precision"], float(threshold_value))
    return {
        "buffer": int(sl["buffer"]),
        "threshold": float(thresholds[idx]),
        "recall": float(r),
        "precision": float(p),
        "threshold_idx": int(idx),
    }


def expanded_segments_from_labels(label: np.ndarray, buffer_size: int) -> list[tuple[int, int, int, int]]:
    """Return GT segments and their buffer-expanded boundaries.

    Each item is (start, end, left_expanded, right_expanded) in inclusive indices.
    """
    label = np.asarray(label, dtype=int).ravel()
    n = int(label.shape[0])
    b = int(buffer_size)
    segs = contiguous_segments(label > 0)
    out: list[tuple[int, int, int, int]] = []
    for s, e in segs:
        le = max(0, s - b)
        re = min(n - 1, e + b)
        out.append((int(s), int(e), int(le), int(re)))
    return out
