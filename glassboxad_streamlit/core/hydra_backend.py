from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _try_import_hydra() -> tuple[Any, Optional[Any]]:
    """Import HYDRA + optional TSB-AD helper.

    This project expects the repo to contain sibling folders:
      - HYDRA/
      - TSB-AD/  (optional; only needed for auto window length)
    """
    import sys

    here = Path(__file__).resolve()
    app_dir = here.parents[1]
    repo_root = app_dir.parent

    # Make sure both this app dir and its parent (expected repo root) are importable
    for p in [repo_root, app_dir]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # Add TSB-AD if present
    tsb_dir = repo_root / "TSB-AD"
    if tsb_dir.exists() and str(tsb_dir) not in sys.path:
        sys.path.insert(0, str(tsb_dir))

    try:
        import HYDRA.HYDRA_loader as loader
    except Exception as e:
        raise ImportError(
            "Cannot import HYDRA.HYDRA_loader.\n"
            "Please place this Streamlit app in a repo that contains a top-level 'HYDRA/' package.\n"
            f"Original error: {e}"
        ) from e

    find_length_rank = None
    try:
        from TSB_AD.utils.slidingWindows import find_length_rank as _flr
        find_length_rank = _flr
    except Exception:
        find_length_rank = None

    return loader, find_length_rank


def load_ts_csv(path: str | Path) -> tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """Load a time series CSV.

    Expected formats:
      - columns include 'Data' and optionally 'Label'
      - otherwise uses the first numeric column as data
      - label column can be 'Label' or 'label'

    Returns
    -------
    X : (T,) float array
    y : (T,) int array or None
    df : original dataframe
    """
    path = Path(path)
    df = pd.read_csv(path)

    data_col = None
    for c in ["Data", "data", "value", "Value", "signal", "Signal"]:
        if c in df.columns:
            data_col = c
            break
    if data_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in {path.name}")
        data_col = numeric_cols[0]

    label_col = None
    for c in ["Label", "label", "y", "Y"]:
        if c in df.columns:
            label_col = c
            break

    X = df[data_col].astype(float).to_numpy()
    y = df[label_col].astype(int).to_numpy() if label_col is not None else None

    return X, y, df


def truncate_data(X: np.ndarray, y: Optional[np.ndarray] = None, limit: int = 10000) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if X.shape[0] > limit:
        X = X[:limit]
        if y is not None:
            y = y[:limit]
    return X, y


def auto_window_length(X: np.ndarray, default: int = 30) -> int:
    loader, find_length_rank = _try_import_hydra()
    if find_length_rank is None:
        return default
    try:
        return int(find_length_rank(X.reshape(-1, 1), rank=1))
    except Exception:
        return default


# Backward-compatible alias used by the Streamlit page
def auto_window_size(X: np.ndarray, default: int = 30) -> int:
    return auto_window_length(X, default=default)


def compute_hydra_payload(
    x: np.ndarray,
    labels: Optional[np.ndarray],
    win_size: int = 30,
    mode: str = "approx",
    truncate_limit: int = 10000,
) -> dict:
    """Compute the JSON payload consumed by the HYDRA 3D viewer."""

    loader, _find_length_rank = _try_import_hydra()

    X = np.asarray(x, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel() if labels is not None else None

    X, labels = truncate_data(X, labels, limit=truncate_limit)

    if X.shape[0] < win_size:
        raise ValueError(f"Data length ({len(X)}) must be at least window size ({win_size}).")

    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    # Windowing (1D)
    shape = (X.shape[0] - win_size + 1, win_size)
    strides = (X.strides[0], X.strides[0])
    windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

    # PCA for layout
    pca = PCA(n_components=2)
    windows_pca = pca.fit_transform(windows)
    pca_min = windows_pca.min(axis=0)
    pca_max = windows_pca.max(axis=0)
    windows_pca_norm = (windows_pca - pca_min) / (pca_max - pca_min + 1e-12) * 2 - 1

    # Run HYDRA
    model = loader.HYDRA(win_size=win_size, mode=mode)
    ts_scores, _win_scores_raw = model.compress_and_score_multi(X)

    # Recompute win_scores with dead-zone handling (copied from your FastAPI impl)
    new_win_scores = []
    nn_indices_list = []
    layer_edges = []

    # Level 0: All windows are representatives
    nbrs0 = NearestNeighbors(n_neighbors=min(50, len(windows)), algorithm="auto").fit(windows)
    dists0, indices0 = nbrs0.kneighbors(windows)

    l0_scores = []
    l0_nn_indices = []
    edges0 = []

    for i in range(len(windows)):
        found = False
        # for k in range(indices0.shape[1]):
        #     target_idx = int(indices0[i, k])
        #     if abs(i - target_idx) >= win_size:
        #         l0_scores.append(float(dists0[i, k]))
        #         l0_nn_indices.append(target_idx)
        #         edges0.append([int(i), target_idx])
        #         found = True
        #         break
        if not found:
            # fallback
            if indices0.shape[1] > 1:
                target_idx = int(indices0[i, 1])
                l0_scores.append(float(dists0[i, 1]))
            else:
                target_idx = int(i)
                l0_scores.append(0.0)
            l0_nn_indices.append(target_idx)
            edges0.append([int(i), target_idx])

    new_win_scores.append(l0_scores)
    nn_indices_list.append(l0_nn_indices)
    layer_edges.append(edges0)

    # Subsequent levels: model.rep_levels_idx contains representatives
    for rep_indices in model.rep_levels_idx:
        rep_indices = np.asarray(rep_indices, dtype=int)
        reps = windows[rep_indices]
        nbrs = NearestNeighbors(n_neighbors=min(50, len(reps)), algorithm="auto").fit(reps)

        dists, indices_all = nbrs.kneighbors(windows)

        old_list = np.asarray(new_win_scores[-1], dtype=float)
        new = np.maximum(dists[:, 0].astype(float), old_list)
        new_win_scores.append(new.tolist())
        nn_indices_list.append(indices_all[:, 0].astype(int).tolist())

        # Representative-to-representative edges for layer graph
        edges = []
        dists_reps, indices_reps = nbrs.kneighbors(reps)
        for i in range(len(reps)):
            source_global = int(rep_indices[i])
            found = False
            for local_k in range(indices_reps.shape[1]):
                target_local = int(indices_reps[i, local_k])
                target_global = int(rep_indices[target_local])
                if abs(source_global - target_global) >= win_size:
                    edges.append([source_global, target_global])
                    found = True
                    break
            if not found and indices_reps.shape[1] > 1:
                target_local = int(indices_reps[i, 1] if indices_reps[i, 0] == i else indices_reps[i, 0])
                edges.append([source_global, int(rep_indices[target_local])])
        layer_edges.append(edges)

    win_scores = np.array(new_win_scores)

    # Re-calculate ts_scores from new_win_scores
    new_ts_scores = []
    for ws in new_win_scores:
        ts_score_lvl = model.reverse_windowing_min(np.array(ws), win_size, model.stride)
        new_ts_scores.append(ts_score_lvl.tolist())
    ts_scores = np.array(new_ts_scores)

    # Ensemble anomaly score
    model.win_scores_mat = win_scores
    ts_score_ens, _win_score_ens = model.ensemble_maxpool_windows()

    global_max_nn = float(win_scores.max()) if win_scores.size else 0.0

    levels_data = []
    links = []

    # Level 0 nodes
    l0_nodes = []
    for i, s in enumerate(win_scores[0]):
        l0_nodes.append(
            {
                "id": f"0_{i}",
                "level": 0,
                "global_idx": int(i),
                "score": float(s),
                "x": float(windows_pca_norm[i, 0]),
                "y": float(windows_pca_norm[i, 1]),
                "parent_id": None,
            }
        )
    levels_data.append({"level": 0, "nodes": l0_nodes})

    # Build hierarchy nodes for representative layers
    for k, (rep_indices, parent_map) in enumerate(zip(model.rep_levels_idx, model.parent_maps)):
        level_id = k + 1
        rep_indices = np.asarray(rep_indices, dtype=int)

        level_nodes = []
        for local_idx, global_idx in enumerate(rep_indices):
            level_nodes.append(
                {
                    "id": f"{level_id}_{local_idx}",
                    "level": level_id,
                    "global_idx": int(global_idx),
                    "score": float(win_scores[0][global_idx]),
                    "x": float(windows_pca_norm[global_idx, 0]),
                    "y": float(windows_pca_norm[global_idx, 1]),
                }
            )
        levels_data.append({"level": level_id, "nodes": level_nodes})

        prev_nodes = levels_data[level_id - 1]["nodes"]
        for i, parent_local_idx in enumerate(parent_map):
            if i < len(prev_nodes):
                child_id = prev_nodes[i]["id"]
                parent_id = f"{level_id}_{int(parent_local_idx)}"
                links.append({"source": child_id, "target": parent_id})
                prev_nodes[i]["parent_id"] = parent_id

    payload = {
        "levels": levels_data,
        "links": links,
        "time_series": X.tolist(),
        "win_size": int(win_size),
        "ts_scores": ts_scores.tolist(),
        "ts_score_ens": np.asarray(ts_score_ens, dtype=float).ravel().tolist(),
        "win_scores": win_scores.tolist(),
        "global_max_nn": global_max_nn,
        # Omit subsequences to keep payload small; the frontend can derive them from time_series.
        "subsequences": None,
        "nn_indices": nn_indices_list,
        "layer_edges": layer_edges,
        "labels": labels.tolist() if labels is not None else None,
        "data_name": "preloaded_ts",
    }

    return payload
