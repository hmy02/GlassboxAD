import numpy as np

# Optional deps: prefer hnswlib, then pynndescent; fall back to sklearn if neither is available
try:
    import hnswlib
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKLEARN_NN = True
except Exception:
    _HAS_SKLEARN_NN = False

class ANNIndex:
    def __init__(self, metric="euclidean", backend="auto",
                 hnsw_M=16, hnsw_ef_construction=200, hnsw_ef=200,
                 pynnd_fraction=1.0, pynnd_n_neighbors=30, pynnd_random_state=42,
                 n_threads=-1):
        self.metric = metric
        self.backend = backend
        self.n_threads = n_threads
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef = hnsw_ef
        self.hnsw_space = "l2" if metric == "euclidean" else "ip"
        self.pynnd_fraction = pynnd_fraction
        self.pynnd_n_neighbors = pynnd_n_neighbors
        self.pynnd_random_state = pynnd_random_state

        # resolve backend early + validate
        valid = {"hnsw", "sklearn"}
        if self.backend == "auto":
            if _HAS_HNSW: self.backend = "hnsw"
            elif _HAS_SKLEARN_NN: self.backend = "sklearn"
            else:
                raise RuntimeError("No ANN backend available: install hnswlib or scikit-learn.")
        elif self.backend not in valid:
            raise ValueError(f"Unknown ANN backend: {self.backend}. Choose from {valid}.")
        self._fitted = False
        self._X = None
        self._index = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32, order="C")
        self._X = X
        n, d = X.shape

        if self.backend == "hnsw":
            p = hnswlib.Index(space=self.hnsw_space, dim=d)
            p.init_index(max_elements=n, ef_construction=self.hnsw_ef_construction, M=self.hnsw_M)
            p.add_items(X)
            p.set_ef(self.hnsw_ef)
            self._index = p

        else:  # sklearn
            nn = NearestNeighbors(
                n_neighbors=2, algorithm="auto", metric=self.metric,
                n_jobs=self.n_threads if self.n_threads != 0 else None
            )
            nn.fit(X)
            self._index = nn

        self._fitted = True
        return self

    def kneighbors(self, Xq=None, n_neighbors=1, return_distance=True):
        if not self._fitted:
            raise RuntimeError("ANNIndex not fitted yet.")

        if Xq is None:
            Xq = self._X
            is_self = True
        else:
            is_self = False

        Xq = np.asarray(Xq, dtype=np.float32, order="C")
        k = int(n_neighbors)
        extra = 1 if is_self else 0
        k_query = k + extra

        dist = ind = None
        if self.backend == "hnsw":
            ind, dist = self._index.knn_query(Xq, k=k_query)
            if self.hnsw_space == "l2":
                dist = np.sqrt(dist, where=dist >= 0, out=dist)
        elif self.backend == "sklearn":
            dist, ind = self._index.kneighbors(Xq, n_neighbors=k_query, return_distance=True)
        else:
            raise ValueError(f"Unknown backend at query time: {self.backend}")

        if ind is None or dist is None:
            raise RuntimeError(f"kneighbors() failed for backend={self.backend}")

        if is_self:
            rows = np.arange(ind.shape[0])
            sel = (ind == rows[:, None])
            has_self = sel.any(axis=1)

            pruned_i, pruned_d = [], []
            for i in range(ind.shape[0]):
                if has_self[i]:
                    j = np.argmax(sel[i])
                    mask = np.ones(ind.shape[1], dtype=bool); mask[j] = False
                else:
                    mask = np.ones(ind.shape[1], dtype=bool); mask[0] = False
                pruned_i.append(ind[i][mask][:k])
                pruned_d.append(dist[i][mask][:k])
            ind = np.vstack(pruned_i)
            dist = np.vstack(pruned_d)
        else:
            ind = ind[:, :k]
            dist = dist[:, :k]

        return (dist, ind) if return_distance else ind