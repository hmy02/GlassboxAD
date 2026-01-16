import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def z_score(row):
    std = np.std(row)
    return (row - np.mean(row)) / (std if std > 1e-8 else 1.0)

class Node:
    def __init__(self, idx):
        self.idx = idx        # window index
        self.children = []    # list of child nodes

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"Node({self.idx}, children={len(self.children)})"

class HYDRA:
    """
    HiAD: Hierarchical Anomaly Detection with window-based compression.

    This class implements:
    1. Sliding-window extraction from time series.
    2. Hierarchical window compression (tree-based).
    3. Multi-level anomaly scoring.
    4. Ensemble scoring strategy across multiple levels.
    """

    def __init__(self, win_size, stride=1, K=2, include_all_windows=True, debug=False):
        """
        Parameters
        ----------
        win_size : int
            Window size for sliding window view.
        stride : int
            Step size for sliding window extraction.
        K : int
            Stop condition for compression (target number of representatives).
        include_all_windows : bool
            If True, includes Level-0 where all windows are used as representatives.
        debug : bool
            Print debug logs for each compression level.
        """
        self.win_size = win_size
        self.stride = stride
        self.K = K
        self.include_all_windows = include_all_windows
        self.debug = debug

        # Storage
        self.ts_scores_mat = None
        self.win_scores_mat = None

    # ---------- Core utility functions ----------
    @staticmethod
    def relabel_zero_based(labels):
        """Reassign cluster/parent labels into contiguous 0..m-1 form."""
        _, inv = np.unique(labels, return_inverse=True)
        return inv

    @staticmethod
    def sliding_windows_view(x, win_size, stride=1):
        """Generate overlapping windows from 1D time series using zero-copy view."""
        slides = sliding_window_view(x, window_shape=win_size, axis=0)
        flat = slides.reshape((slides.shape[0], -1))
        return flat[::stride, :].astype(np.float32, copy=False)

    @staticmethod
    def find_nearest_indices(X):
        """Return indices of the nearest neighbor (excluding self) for each row in X."""
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
        return nn.kneighbors(return_distance=False)[:, 1]

    @staticmethod
    def compute_count(nearest_idx, N):
        """Count how many times each index is chosen as nearest neighbor."""
        return np.bincount(nearest_idx, minlength=N)

    @staticmethod
    def anomaly_score(win, reps):
        """
        Compute distance-based anomaly score:
        for each window in `win`, return distance to the closest representative in `reps`.
        """
        nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
        nbrs.fit(reps)
        dist, idx = nbrs.kneighbors(win)
        return dist.ravel(), idx.ravel()

    @staticmethod
    def compress_once_dir(nearest_idx, count):
        """
        Perform one-step directed merging:
        - If node i and j are nearest neighbors,
          assign the one with larger 'count' (or smaller index when tie) as parent.
        - Apply path compression afterwards.
        """
        N = len(nearest_idx)
        parent = np.arange(N, dtype=np.int64)

        for i in range(N):
            j = nearest_idx[i]
            if i == j:
                continue
            if (count[j] > count[i]) or (count[j] == count[i] and j < i):
                parent[i] = j
            else:
                parent[j] = i

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(N):
            parent[i] = find(i)

        return HYDRA.relabel_zero_based(parent)

    @staticmethod
    def reverse_windowing_min(score_win, win_size, stride):
        """
        Map window-level anomaly scores back to time series points.
        Uses averaging when multiple windows overlap the same point.
        """
        if stride == 1:
            num_win = len(score_win)
            weights = np.ones(win_size)
            out = np.convolve(score_win, weights, mode="full")
            denom = np.convolve(np.ones(num_win), weights, mode="full")
            return out[: denom.size] / denom
        else:
            num_win = len(score_win)
            T = stride * (num_win - 1) + win_size
            out = np.zeros(T, dtype=float)
            cnt = np.zeros(T, dtype=float)
            idx = np.arange(win_size)
            for i, s in enumerate(score_win):
                pos = i * stride + idx
                out[pos] += s
                cnt[pos] += 1
            return out / np.maximum(cnt, 1)

    def tree_compress_collect_levels(self, windows):
        """
        Perform hierarchical compression of windows until reaching K representatives.
        Returns representative indices for each level and parent mappings.
        """
        N = len(windows)
        levels = [np.arange(N, dtype=np.int64)]
        parent_maps = []
        rep_levels_idx = []

        cur_idx = levels[-1]
        cur_win = windows[cur_idx]
        level_id = 0

        while len(cur_idx) > self.K:

            nn = self.find_nearest_indices(cur_win)
            cnt = self.compute_count(nn, len(cur_idx))
            inv = self.compress_once_dir(nn, cnt)
            _, first_idx = np.unique(inv, return_index=True)

            parent_maps.append(inv)
            next_idx = cur_idx[first_idx]
            rep_levels_idx.append(next_idx)

            levels.append(next_idx)
            cur_idx = next_idx
            cur_win = windows[cur_idx]
            level_id += 1
        self.parent_maps = parent_maps
        self.rep_levels_idx = rep_levels_idx
        return rep_levels_idx, parent_maps

    def build_tree(self, parent_maps, rep_levels_idx):
        """
        Build a hierarchical tree structure from HYDRA parent_maps and representative indices.
        Handles correct index mapping between compression levels.
        """
        class Node:
            def __init__(self, idx):
                self.idx = int(idx)
                self.children = []
            def add_child(self, child):
                self.children.append(child)
            def __repr__(self):
                return f"Node({self.idx}, children={len(self.children)})"
    
        # All windows at level 0 are global indices [0, 1, 2, ..., N-1]
        num_windows = len(parent_maps[0])
        nodes = {i: Node(i) for i in range(num_windows)}
    
        # Level index lists, where level_indices[i] gives the *global indices*
        level_indices = [np.arange(num_windows)]
        for rep_idx in rep_levels_idx:
            level_indices.append(rep_idx)
    
        # Iterate through levels safely
        for level, parent_map in enumerate(parent_maps):
            cur_idx = level_indices[level]
            for local_child, local_parent in enumerate(parent_map):
                if local_child >= len(cur_idx) or local_parent >= len(cur_idx):
                    continue
                child_global = cur_idx[local_child]
                parent_global = cur_idx[local_parent]
                if child_global != parent_global:
                    nodes[parent_global].add_child(nodes[child_global])
    
        # --- FIX: top-level representatives are the real roots
        root_indices = rep_levels_idx[-1] if len(rep_levels_idx) > 0 else np.arange(len(nodes))
        roots = [nodes[int(i)] for i in root_indices if int(i) in nodes]
        return roots

    # ---------- Main pipeline ----------
    def compress_and_score_multi(self, X):
        """
        Run hierarchical compression and compute multi-level anomaly scores.
        Returns
        -------
        ts_scores_mat : array, shape (L, T)
            Time series anomaly scores at each level.
        win_scores_mat : array, shape (L, num_win)
            Window anomaly scores at each level.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] > 1:
            # For multivariate, z-score each channel first (column-wise)
            X = np.apply_along_axis(z_score, axis=0, arr=X)

        windows = self.sliding_windows_view(X, self.win_size, self.stride)
        num_win = len(windows)

        win_scores_list = []
        ts_scores_list = []
        rep_idx_levels = []

        if self.include_all_windows:
            nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(windows)
            dists, _ = nn.kneighbors(windows, return_distance=True)
            win_scores_L0 = dists[:, 1]  # 2nd nearest (not self)
            ts_scores_L0 = self.reverse_windowing_min(win_scores_L0, self.win_size, self.stride)

            win_scores_list.append(win_scores_L0)
            ts_scores_list.append(ts_scores_L0)
            rep_idx_levels.append(np.arange(num_win, dtype=np.int64))
            if self.debug:
                print(f"Level 0 (all windows): reps={num_win}")

        rep_idx_levels_core, parent_maps = self.tree_compress_collect_levels(windows)
        

        for lvl_idx, rep_idx in enumerate(rep_idx_levels_core, start=int(self.include_all_windows)):
            rep_win = windows[rep_idx]
            win_scores, _ = self.anomaly_score(windows, rep_win)

            win_scores_list.append(win_scores)
            ts_scores = self.reverse_windowing_min(win_scores, self.win_size, self.stride)
            ts_scores_list.append(ts_scores)

            rep_idx_levels.append(rep_idx)
            if self.debug:
                print(f"Level {lvl_idx}: reps={len(rep_idx)}")

        self.win_scores_mat = np.vstack(win_scores_list)
        self.ts_scores_mat = np.vstack(ts_scores_list)
        return self.ts_scores_mat, self.win_scores_mat

    def ensemble_maxpool_windows(self):
        if self.win_scores_mat is None or self.win_scores_mat.size == 0:
            raise ValueError("Run compress_and_score_multi first before ensemble.")

        W = self.win_scores_mat.astype(float, copy=True)
        mu = np.nanmean(W, axis=1, keepdims=True)
        sd = np.nanstd(W, axis=1, keepdims=True)
        ok = np.isfinite(sd) & (sd > 0)
        Z = np.zeros_like(W)
        Z[ok[:, 0]] = (W[ok[:, 0]] - mu[ok[:, 0]]) / sd[ok[:, 0]]

        win_score_final = np.max(Z, axis=0)
        ts_score_final = self.reverse_windowing_min(win_score_final, self.win_size, self.stride)
        return ts_score_final, win_score_final

if __name__ == '__main__':
    X = np.sin(np.linspace(0, 20, 500)) + 0.1 * np.random.randn(500)

    model = HYDRA(win_size=30, stride=1, K=2, include_all_windows=True, debug=True)
    ts_scores, win_scores = model.compress_and_score_multi(X)

    ens_score,_ = model.ensemble_maxpool_windows()