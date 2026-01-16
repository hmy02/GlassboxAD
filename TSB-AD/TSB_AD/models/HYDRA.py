import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
import time

# Optional deps: prefer hnswlib, then pynndescent; fall back to sklearn if neither is available
try:
    import hnswlib
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

try:
    from pynndescent import NNDescent
    _HAS_PYNND = True
except Exception:
    _HAS_PYNND = False

try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKLEARN_NN = True
except Exception:
    _HAS_SKLEARN_NN = False


# ---------- Normalization utilities ----------
def z_score(row):
    std = np.std(row)
    return (row - np.mean(row)) / (std if std > 1e-8 else 1.0)


# ---------- Unified ANN wrapper ----------
class ANNIndex:
    """
    A unified ANN wrapper:
      backend priority: hnswlib -> pynndescent -> sklearn (exact, fallback)
    Only Euclidean distance is supported (to match the original code).
    """
    def __init__(self, metric="euclidean", backend="auto",
                 # hnswlib params
                 hnsw_M=16, hnsw_ef_construction=200, hnsw_ef=200, hnsw_space="l2",
                 # pynndescent params
                 pynnd_fraction=1.0, pynnd_n_neighbors=30, pynnd_random_state=42,
                 n_threads=-1):
        self.metric = metric
        self.backend = backend
        self.n_threads = n_threads

        # Choose backend
        if backend == "auto":
            if _HAS_HNSW:
                self.backend = "hnsw"
            elif _HAS_PYNND:
                self.backend = "pynnd"
            else:
                self.backend = "sklearn"

        # Save params
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef = hnsw_ef
        self.hnsw_space = "l2" if metric == "euclidean" else "ip"

        self.pynnd_fraction = pynnd_fraction
        self.pynnd_n_neighbors = pynnd_n_neighbors
        self.pynnd_random_state = pynnd_random_state

        # Runtime state
        self._fitted = False
        self._X = None
        self._index = None  # hnswlib.Index or NNDescent or sklearn.NearestNeighbors

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32, order="C")
        self._X = X
        n, d = X.shape

        if self.backend == "hnsw":
            p = hnswlib.Index(space=self.hnsw_space, dim=d)
            p.init_index(max_elements=n, ef_construction=self.hnsw_ef_construction, M=self.hnsw_M)
            p.add_items(X, np.arange(n, dtype=np.int32), num_threads=1)
            p.set_ef(self.hnsw_ef)
            self._index = p

        elif self.backend == "pynnd":
            # Build PyNNDescent (suited for self-query; use larger k for stable 2NN)
            k0 = max(self.pynnd_n_neighbors, 30)
            self._index = NNDescent(
                X,
                n_neighbors=k0,
                metric=self.metric,
                random_state=self.pynnd_random_state,
                n_jobs=self.n_threads if self.n_threads != 0 else None
            )

        else:
            # sklearn fallback (exact KNN)
            if not _HAS_SKLEARN_NN:
                raise ImportError("sklearn.neighbors.NearestNeighbors is required for the sklearn backend.")
            algo = "auto"
            self._index = NearestNeighbors(
                n_neighbors=2, algorithm=algo, metric=self.metric,
                n_jobs=self.n_threads if self.n_threads != 0 else None
            )
            self._index.fit(X)

        self._fitted = True
        return self

    def kneighbors(self, Xq=None, n_neighbors=1, return_distance=True):
        """
        Same API as sklearn:
          - If Xq is None, perform self-query against the training set, and auto-remove self-matches.
          - Return (dist, ind) or ind.
        """
        if not self._fitted:
            raise RuntimeError("ANNIndex not fitted yet.")

        if Xq is None:
            Xq = self._X
        Xq = np.asarray(Xq, dtype=np.float32, order="C")
        k = int(n_neighbors)

        # For self-query, ask for one extra neighbor to drop the self-match
        extra = 1 if Xq is self._X else 0
        k_query = k + extra

        if self.backend == "hnsw":
            ind, dist = self._index.knn_query(Xq, k=k_query, num_threads=self.n_threads)
            # hnswlib's l2 returns squared Euclidean distance
            if self.hnsw_space == "l2":
                dist = np.sqrt(dist, where=dist >= 0, out=dist)

        elif self.backend == "pynnd":
            ind, dist = self._index.query(Xq, k=k_query)

        else:
            dist, ind = self._index.kneighbors(Xq, n_neighbors=k_query, return_distance=True)

        # If self-query, drop the self neighbor and truncate to k
        if Xq is self._X:
            rows = np.arange(ind.shape[0])
            sel = (ind == rows[:, None])
            has_self = sel.any(axis=1)

            pruned_ind, pruned_dist = [], []
            for i in range(ind.shape[0]):
                if has_self[i]:
                    j = np.argmax(sel[i])  # column of the self-id
                    mask = np.ones(ind.shape[1], dtype=bool)
                    mask[j] = False
                else:
                    mask = np.ones(ind.shape[1], dtype=bool)
                    mask[0] = False
                pruned_ind.append(ind[i][mask][:k])
                pruned_dist.append(dist[i][mask][:k])

            ind = np.vstack(pruned_ind)
            dist = np.vstack(pruned_dist)
        else:
            # External queries: just truncate
            ind = ind[:, :k]
            dist = dist[:, :k]

        return (dist, ind) if return_distance else ind


# ---------- HiAD core ----------
class HYDRA:
    """
    HiAD: Hierarchical Anomaly Detection with window-based compression (with ANN).
    Key change: when computing reps(L+1), we keep the ANN built on reps(L),
    and reuse it to score level L (fewer index builds; no pruning).
    """
    def __init__(self, win_size, stride=1, K=1, include_all_windows=True, debug=False,
                 # ANN related
                 ann_backend="auto",
                 hnsw_M=16, hnsw_ef_construction=100, hnsw_ef=40,
                 pynnd_fraction=1.0, pynnd_n_neighbors=2, pynnd_random_state=42,
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
            pynnd_fraction=pynnd_fraction,
            pynnd_n_neighbors=pynnd_n_neighbors,
            pynnd_random_state=pynnd_random_state,
            n_threads=ann_threads
        )

    # ---------- Basic utilities ----------
    @staticmethod
    def relabel_zero_based(labels):
        _, inv = np.unique(labels, return_inverse=True)
        return inv

    @staticmethod
    def sliding_windows_view(x, win_size, stride=1):
        """
        x: (T, D) or (T,); return shape (num_windows, win_size*D)
        """
        slides = sliding_window_view(x, window_shape=win_size, axis=0)
        flat = slides.reshape((slides.shape[0], -1))
        return flat[::stride, :].astype(np.float32, copy=False)

    @staticmethod
    def compute_count(nearest_idx, N):
        return np.bincount(nearest_idx, minlength=N)

    @staticmethod
    def compress_once_dir(nearest_idx, count):
        """
        One directional merge: the more "popular" point becomes the representative;
        if tied, take the smaller index.
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
        Project window scores back to the time axis (mean overlap; can be swapped for min/median etc.)
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
        Main loop:
          - Current level database = cur_win = reps(L)
          - Build ann_cur (once), use 2NN on it to obtain reps(L+1)
          - Save ann_cur as the scoring index for level L
        Returns:
          - rep_levels_idx: global original indices of reps for each level
          - parent_maps   : compression maps
          - rep_ann_levels: ANN per level (database = reps of that level)
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
        Returns:
          - ts_scores_mat: (L', T)   per-level time-series scores (including/excluding level-0 per include_all_windows)
          - win_scores_mat:(L', num_win) per-level window scores
        Also records:
          - self.exe: time for compression stage
          - self.inf: time for scoring stage
        """
        t1 = time.time()

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
        t2 = time.time()

        # level-0: distances from all windows to their nearest neighbor (from the first 2NN dist[:,0])
        if self.include_all_windows:
            if self.level0 is None:
                # Rare edge fallback (should not trigger normally)
                dist_full, _, _ = self._nearest_indices_and_dist(windows, n_neighbors=2)
                self.level0 = dist_full[:, 0].astype(np.float32, copy=False)
            win_scores_list.append(self.level0)
            ts_scores_list.append(self.reverse_windowing_min(self.level0, self.win_size, self.stride))

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
                print(f"[Level {lvl_k}] reps={len(rep_idx)}  (reused ANN for scoring)")

        # Stack outputs
        self.win_scores_mat = np.vstack(win_scores_list)
        self.ts_scores_mat = np.vstack(ts_scores_list)

        t3 = time.time()
        self.exe = t2 - t1
        self.inf = t3 - t2
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
