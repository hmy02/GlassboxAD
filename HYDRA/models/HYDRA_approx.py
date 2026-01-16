import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .HNSW_index import ANNIndex

def z_score(row):
    std = np.std(row)
    return (row - np.mean(row)) / (std if std > 1e-8 else 1.0)

# ---------- HiAD core ----------
class HYDRA:
    """
    HiAD: Hierarchical Anomaly Detection with window-based compression (with ANN).
    Key change: when computing reps(L+1), we keep the ANN built on reps(L),
    and reuse it to score level L (fewer index builds; no pruning).
    """
    def __init__(self, win_size, stride=1, K=2, include_all_windows=True, debug=False,
                 # ANN related
                 ann_backend="auto",
                 hnsw_M=16, hnsw_ef_construction=100, hnsw_ef=40,
                 ann_threads=1):
        self.win_size = win_size
        self.stride = stride
        self.K = K
        self.include_all_windows = include_all_windows
        self.debug = debug

        # Storage
        self.ts_scores_mat = None
        self.win_scores_mat = None
        self.level0 = None

        # ANN for scoring at each level: rep_ann_levels[L] is the index built on reps(L)
        self.rep_ann_levels = []

        # ANN config
        self.ann_backend = ann_backend
        self.ann_params = dict(
            metric="euclidean",
            backend=ann_backend,
            hnsw_M=hnsw_M,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef=hnsw_ef,
            n_threads=ann_threads
        )

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
    def compute_count(nearest_idx, N):
        """Count how many times each index is chosen as nearest neighbor."""
        return np.bincount(nearest_idx, minlength=N)

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
            weights = np.ones(win_size, dtype=float)
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
            return out / np.maximum(cnt, 1.0)

    # ---------- ANN helpers ----------
    def _build_ann(self, X):
        return ANNIndex(**self.ann_params).fit(X)

    def _nearest_indices_and_dist(self, X, n_neighbors=2, ann_obj=None):
        """
        Return 2NN distances and indices, and the ann used (for reuse upstream).
        - If ann_obj is provided, reuse it; otherwise build a new index.
        """
        ann = ann_obj if ann_obj is not None else self._build_ann(X)
        dist, ind = ann.kneighbors(None, n_neighbors=n_neighbors, return_distance=True)  # self-query
        return dist, ind, ann

    # ---------- Hierarchical compression (store per-level ANN for scoring) ----------
    def tree_compress_collect_levels(self, windows):
        """
        Perform hierarchical compression of windows until reaching K representatives.
        Returns representative indices for each level and parent mappings.
        """
        N = len(windows)
        levels = [np.arange(N, dtype=np.int64)]
        parent_maps = []
        rep_levels_idx = []
        rep_ann_levels = []

        cur_idx = levels[-1]          # indices of all windows (L=0)
        cur_win = windows[cur_idx]    # database at level L
        level_id = 0

        while len(cur_idx) > self.K:
            # 1) Build index on reps(L)=cur_win to run 2NN
            dist_full, nn_full, ann_cur = self._nearest_indices_and_dist(cur_win, n_neighbors=2)
            dist = dist_full[:, 0]                 # 1NN distance (after removing self-match)
            nn = nn_full[:, 0].astype(np.int64)    # 1NN indices (in cur_win's coordinate)

            # 2) Count popularity and compress to obtain reps(L+1)
            cnt = self.compute_count(nn, len(cur_idx))
            inv = self.compress_once_dir(nn, cnt)
            _, first_idx = np.unique(inv, return_index=True)   # representative row ids in cur_win
            next_idx = cur_idx[first_idx]                      # convert back to global indices

            parent_maps.append(inv)
            rep_levels_idx.append(next_idx)

            # 3) Save the scoring index for level L (database = reps(L) = cur_win)
            rep_ann_levels.append(ann_cur)

            # 4) Move to the next level
            levels.append(next_idx)
            cur_idx = next_idx
            cur_win = windows[cur_idx]
            level_id += 1

            # Record level-0 1NN distances for include_all_windows
            if level_id == 1:
                self.level0 = dist.astype(np.float32, copy=False)

        return rep_levels_idx, parent_maps, rep_ann_levels

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

        # Windowing
        windows = self.sliding_windows_view(X, self.win_size, self.stride)

        win_scores_list = []
        ts_scores_list = []
        rep_idx_levels = []

        # Compress while retrieving the ANN used for scoring at each level
        rep_idx_levels_core, parent_maps, rep_ann_levels = self.tree_compress_collect_levels(windows)

        # Store hierarchy for visualization
        self.rep_levels_idx = rep_idx_levels_core
        self.parent_maps = parent_maps

        # level-0: distances from all windows to their nearest neighbor (from the first 2NN dist[:,0])
        if self.include_all_windows:
            if self.level0 is None:
                # Rare edge fallback (should not trigger normally)
                dist_full, _, _ = self._nearest_indices_and_dist(windows, n_neighbors=2)
                self.level0 = dist_full[:, 0].astype(np.float32, copy=False)
            win_scores_list.append(self.level0)
            ts_scores_list.append(self.reverse_windowing_min(self.level0, self.win_size, self.stride))
            if self.debug:
                print(f"[Level {0}] reps={len(self.level0)}")

        # For each level L: reuse the ANN built on reps(L) to score (1NN to reps(L))
        for lvl_k, (rep_idx, ann_for_level) in enumerate(
            zip(rep_idx_levels_core, rep_ann_levels),
            start=int(self.include_all_windows)
        ):
            # 1NN to reps(L) (database = reps(L); query = all windows)
            dist, _ = ann_for_level.kneighbors(windows, n_neighbors=1, return_distance=True)
            win_scores = dist.ravel()

            win_scores_list.append(win_scores.astype(np.float32, copy=False))
            ts_scores_list.append(self.reverse_windowing_min(win_scores, self.win_size, self.stride))

            rep_idx_levels.append(rep_idx)

            if self.debug:
                print(f"[Level {lvl_k}] reps={len(rep_idx)}")

        # Stack outputs
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