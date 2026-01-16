import sys
import os
from pathlib import Path
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import io
from sklearn.decomposition import PCA

# Add parent directory to path to import HYDRA and TSB-AD
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "TSB-AD"))

import HYDRA.HYDRA_loader as loader
from TSB_AD.utils.slidingWindows import find_length_rank

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(current_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(current_dir / "templates"))

# Global storage for the current data
current_data = {
    "X": None,
    "labels": None,
    "name": None,
    "win_size": 30
}

def truncate_data(X, labels=None, limit=10000):
    if len(X) > limit:
        X = X[:limit]
        if labels is not None:
            labels = labels[:limit]
    return X, labels

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...), window_size: int = 30, auto_window: bool = False):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    if "Data" not in df.columns:
        return JSONResponse({"error": "CSV must contain a 'Data' column"}, status_code=400)
    
    X = df["Data"].values.astype(float)
    labels = df["Label"].values if "Label" in df.columns else None
    
    X, labels = truncate_data(X, labels)
    
    if auto_window:
        try:
            # find_length_rank expects 2D (N, 1) or 1D? main.py uses data.values.astype(float)
            # data is df.iloc[:, 0:-1].values.astype(float) -> (N, D)
            window_size = int(find_length_rank(X.reshape(-1, 1), rank=1))
            print(f"Auto-detected window size: {window_size}")
        except Exception as e:
            print(f"Error in find_length_rank: {e}, falling back to {window_size}")
    
    if len(X) < window_size:
        return JSONResponse({"error": f"Data length ({len(X)}) must be at least window size ({window_size})"}, status_code=400)
    
    current_data["X"] = X
    current_data["labels"] = labels
    current_data["name"] = file.filename
    current_data["win_size"] = window_size
    
    return {"status": "success", "length": len(X), "win_size": window_size}

@app.get("/api/generate_random")
async def generate_random(window_size: int = 30, auto_window: bool = False):
    np.random.seed(42)
    # A simple sine wave with an anomaly
    X = np.sin(np.linspace(0, 50, 1000)) 
    # Add anomaly
    X[500:550] += 2.0 * np.sin(np.linspace(0, 10, 50))
    X += 0.1 * np.random.randn(1000)
    
    if auto_window:
        try:
            window_size = int(find_length_rank(X.reshape(-1, 1), rank=1))
            print(f"Auto-detected window size for random: {window_size}")
        except Exception as e:
            print(f"Error for random: {e}")

    current_data["X"] = X
    current_data["labels"] = None
    current_data["name"] = "Random Sine Wave"
    current_data["win_size"] = window_size
    
    return {"status": "success", "length": len(X), "win_size": window_size}

@app.get("/api/data")
async def get_data():
    if current_data["X"] is None:
        # Fallback to generating random if no data loaded
        await generate_random()
    
    X = current_data["X"]
    labels = current_data["labels"]
    win_size = current_data.get("win_size", 30)
    
    # Windowing
    # We need to reconstruct windows to compute PCA
    # HYDRA uses sliding_window_view
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Re-create windows (same logic as HYDRA)
    # X is 1D array
    shape = (X.shape[0] - win_size + 1, win_size)
    strides = (X.strides[0], X.strides[0])
    windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    
    # Compute PCA on all windows
    pca = PCA(n_components=2)
    windows_pca = pca.fit_transform(windows)
    
    # Normalize PCA coordinates to [-1, 1] for easier visualization
    # But keep aspect ratio? Let's just normalize to fit in a box.
    pca_min = windows_pca.min(axis=0)
    pca_max = windows_pca.max(axis=0)
    windows_pca_norm = (windows_pca - pca_min) / (pca_max - pca_min) * 2 - 1
    # Actually, let's keep it simple, frontend can scale. 
    # But normalizing here ensures consistent view.
    
    # Run HYDRA
    model = loader.HYDRA(win_size=win_size, mode='approx')
    ts_scores, win_scores_raw = model.compress_and_score_multi(X)
    
    # We will recompute win_scores to handle dead zones (trivial matches)
    # win_scores will be (L+1, N)
    new_win_scores = []
    nn_indices_list = [] # (L+1, N) global indices
    layer_edges = [] # List of lists of pairs [source_global, target_global]
    
    from sklearn.neighbors import NearestNeighbors
    
    # --- Level 0: All windows are representatives ---
    nbrs0 = NearestNeighbors(n_neighbors=min(50, len(windows)), algorithm='auto').fit(windows)
    dists0, indices0 = nbrs0.kneighbors(windows)
    
    l0_scores = []
    l0_nn_indices = []
    edges0 = []
    
    for i in range(len(windows)):
        found = False
        # Find first neighbor outside dead zone (win_size)
        for k in range(indices0.shape[1]):
            target_idx = indices0[i, k]
            if abs(i - target_idx) >= win_size:
                l0_scores.append(float(dists0[i, k]))
                l0_nn_indices.append(int(target_idx))
                edges0.append([i, int(target_idx)])
                found = True
                break
        if not found:
            # Fallback to second neighbor if nothing outside dead zone
            l0_scores.append(float(dists0[i, 1]) if indices0.shape[1] > 1 else 0.0)
            target_idx = indices0[i, 1] if indices0.shape[1] > 1 else i
            l0_nn_indices.append(int(target_idx))
            edges0.append([i, int(target_idx)])
            
    new_win_scores.append(l0_scores)
    nn_indices_list.append(l0_nn_indices)
    layer_edges.append(edges0)
    
    # --- Subsequent levels: model.rep_levels_idx contains representatives ---
    for k, rep_indices in enumerate(model.rep_levels_idx):
        reps = windows[rep_indices]
        nbrs = NearestNeighbors(n_neighbors=min(50, len(reps)), algorithm='auto').fit(reps)
        
        # 1. Find NN for ALL windows against these representatives
        dists, indices_all = nbrs.kneighbors(windows)
        
        level_scores = []
        level_nn_indices = []
        
        # for j in range(len(windows)):
        #     found = False
        #     for local_k in range(indices_all.shape[1]):
        #         local_idx = indices_all[j, local_k]
        #         global_idx_rep = rep_indices[local_idx]
                
        #         # Exclude if same index or within dead zone
        #         if abs(j - global_idx_rep) >= win_size:
        #             level_scores.append(float(dists[j, local_k]))
        #             level_nn_indices.append(int(global_idx_rep))
        #             found = True
        #             break
        #     if not found:
        #         # Fallback to the closest one (usually itself if it is a rep)
        #         level_scores.append(float(dists[j, 0]))
        #         level_nn_indices.append(int(rep_indices[indices_all[j, 0]]))
        old_list = np.asarray(new_win_scores[-1],dtype=float)
        new = np.maximum(dists[:,0].astype(float),old_list)
        new_win_scores.append(new.tolist())
        nn_indices_list.append(indices_all[:,0].tolist())
        
        # 2. Find NN edges between representatives (for the layer graph)
        # (This is just for visualization of the "layer map")
        edges = []
        dists_reps, indices_reps = nbrs.kneighbors(reps)
        for i in range(len(reps)):
            source_global = rep_indices[i]
            found = False
            for local_k in range(indices_reps.shape[1]):
                target_local = indices_reps[i, local_k]
                target_global = rep_indices[target_local]
                if abs(source_global - target_global) >= win_size:
                    edges.append([int(source_global), int(target_global)])
                    found = True
                    break
            if not found and indices_reps.shape[1] > 1:
                target_local = indices_reps[i, 1] if indices_reps[i, 0] == i else indices_reps[i, 0]
                edges.append([int(source_global), int(rep_indices[target_local])])
        layer_edges.append(edges)

    # Convert to numpy for easier handling
    win_scores = np.array(new_win_scores)
    
    # Re-calculate ts_scores from new_win_scores to ensure consistency
    new_ts_scores = []
    for ws in new_win_scores:
        # reverse_windowing_min is a static method on the HYDRA class
        ts_score_lvl = model.reverse_windowing_min(np.array(ws), win_size, model.stride)
        new_ts_scores.append(ts_score_lvl.tolist())
    ts_scores = np.array(new_ts_scores)

    # Compute Ensemble Anomaly Score (Aggregated)
    # win_scores is (L+1, N)
    model.win_scores_mat = win_scores
    ts_score_ens, win_score_ens = model.ensemble_maxpool_windows()
    
    # Global max NN distance for histogram normalization
    global_max_nn = float(win_scores.max())

    levels_data = []
    links = []
    
    # Level 0: All windows
    l0_scores = win_scores[0]
    l0_nodes = []
    for i in range(len(l0_scores)):
        l0_nodes.append({
            "id": f"0_{i}",
            "level": 0,
            "global_idx": i,
            "score": float(l0_scores[i]),
            "x": float(windows_pca_norm[i, 0]),
            "y": float(windows_pca_norm[i, 1]),
            "parent_id": None # Will be filled by next levels? No, parents are in next level.
        })
    levels_data.append({"level": 0, "nodes": l0_nodes})
    
    # Subsequent levels
    # We need to map children to parents to build the path.
    # parent_maps[k] maps Level k (local idx) to Level k+1 (local idx).
    
    # Let's build a map of child_id -> parent_id
    child_to_parent = {}
    
    for k, (rep_indices, parent_map) in enumerate(zip(model.rep_levels_idx, model.parent_maps)):
        level_id = k + 1
        prev_level_id = k
        
        level_nodes = []
        scores_at_level = win_scores[level_id] # Score of windows at this level
        
        # rep_indices are global indices. 
        # We can look up their PCA coordinates from Level 0.
        
        for local_idx, global_idx in enumerate(rep_indices):
            # For score: win_scores[level_id] is the score of the QUERY windows against this level's reps.
            # That's not the score OF the rep.
            # The score OF the rep should be its intrinsic anomaly score (Level 0 score).
            # OR, maybe we want to see how the score evolves?
            # Let's send the Level 0 score for consistency of "how anomalous is this pattern".
            # But the user asked for "NN distance depicted on each layer".
            # That implies for a selected node (query), what is its distance to NN in this layer.
            # We can compute that on the fly or pre-compute.
            # win_scores[level_id] IS that distance for every window.
            # So for the representatives themselves, we can also look up their distance in this level.
            
            # Wait, win_scores[level_id] has size N (number of original windows).
            # It tells us for every original window, what is the distance to the nearest rep in Level level_id.
            # So we can just send the full win_scores matrix to the frontend?
            # It's (L, N). For N=1000, L=5, that's 5000 floats. Very small.
            
            score = float(win_scores[0][global_idx]) # Intrinsic score
            
            level_nodes.append({
                "id": f"{level_id}_{local_idx}",
                "level": level_id,
                "global_idx": int(global_idx),
                "score": score,
                "x": float(windows_pca_norm[global_idx, 0]),
                "y": float(windows_pca_norm[global_idx, 1])
            })
        levels_data.append({"level": level_id, "nodes": level_nodes})
        
        # Build links and parent mapping
        # parent_map maps Level k local_idx -> Level k+1 local_idx
        
        # Previous level nodes
        prev_nodes = levels_data[prev_level_id]["nodes"]
        
        for i, parent_local_idx in enumerate(parent_map):
            if i < len(prev_nodes):
                child_id = prev_nodes[i]["id"]
                parent_id = f"{level_id}_{parent_local_idx}"
                
                links.append({
                    "source": child_id,
                    "target": parent_id
                })
                
                # Store parent pointer in the child node for easy traversal up
                prev_nodes[i]["parent_id"] = parent_id

    # Subsequences (send all for now, optimization possible later)
    subsequences = {}
    for i in range(len(windows)):
        subsequences[i] = windows[i].tolist()

    return JSONResponse({
        "levels": levels_data,
        "links": links,
        "time_series": X.tolist(),
        "win_size": win_size,
        "ts_scores": ts_scores.tolist(),
        "ts_score_ens": ts_score_ens.tolist(),
        "win_scores": win_scores.tolist(), # (L, N) matrix
        "global_max_nn": global_max_nn,
        "subsequences": subsequences,
        "nn_indices": nn_indices_list, # (L, N) matrix of global indices
        "layer_edges": layer_edges, # List of lists of [source_global, target_global]
        "labels": labels.tolist() if labels is not None else None,
        "data_name": current_data["name"]
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
