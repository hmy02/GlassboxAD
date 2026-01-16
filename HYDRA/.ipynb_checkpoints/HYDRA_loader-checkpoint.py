from .models.HYDRA_approx import HYDRA as HYDRAApprox
from .models.HYDRA_exact import HYDRA as HYDRAExact
import numpy as np

def HYDRA(win_size, stride=1, K=2, include_all_windows=True, debug=False,
                 # ANN related
                 ann_backend="auto",
                 hnsw_M=16, hnsw_ef_construction=100, hnsw_ef=40,
                 ann_threads=1, mode='approx'):
    if mode == 'exact':
        return HYDRAExact(win_size, stride=stride, K=K, include_all_windows=include_all_windows, debug=debug)
    elif mode == 'approx':
        return HYDRAApprox(win_size, stride=stride, K=K, include_all_windows=include_all_windows, debug=debug,
                           ann_threads=ann_threads, hnsw_M=hnsw_M, hnsw_ef_construction=hnsw_ef_construction,
                           hnsw_ef=hnsw_ef, ann_backend=ann_backend)

