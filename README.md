# <b>HYDRA</b>: A Multi-Level <b>H</b>ierarch<b>Y</b>-<b>D</b>riven Approach for <b>R</b>obust <b>A</b>nomaly Detection in Time Series</h3>

![logo](assets/logo.png)

<!-- ABOUT THE PROJECT -->
## About HYDRA

Time-series anomaly detection is critical across various domains. Despite advances in neural networks and foundation models, recent studies show that traditional data mining methods remain highly competitive due to their effectiveness and scalability. However, these approaches suffer from distinct limitations: discord-based methods fail in the presence of repeated anomalies, whereas clustering-based techniques, though mitigating this issue, struggle to capture fine-grained deviations.
Moreover, both approaches rely on distance computation, whose effectiveness fundamentally depends on data normalization, with z-score serving as the de facto standard. However, we observe that while normalization reduces scale bias and can enhance anomaly detectability, it may also suppress amplitude-driven anomalies, making the choice of an appropriate normalization scheme both critical and non-trivial.
To address these challenges, we propose HYDRA, a multi-level hierarchical and unsupervised approach that integrates the strengths of distance-based methods while reducing reliance on explicit normalization. HYDRA (i) employs a lightweight approximate nearest-neighbor detector with graph-based selection to identify representative subsequences; (ii) constructs multi-resolution representations of the time series and aggregates anomaly evidence from fine to coarse scales; and (iii) introduces a hierarchical ensemble mechanism that fuses level-wise scores to improve robustness against contamination and scale imbalance. This design allows HYDRA to detect diverse anomaly types, from short, isolated discords to long, persistent deviations, allowing it to detect patterns overlooked by single-scale methods.
Extensive evaluation on 40 univariate and multivariate time-series anomaly detection datasets from the TSB-AD benchmark demonstrates that HYDRA achieves state-of-the-art performance, ranking first among 40 competing algorithms, while maintaining scalability to ultra-long sequences.


<!-- GETTING STARTED -->
## Getting Started

### Benchmark

Benchmark for evaluation: [TSB-AD](https://github.com/TheDatumOrg/TSB-AD)

### Installation

To install HYDRA from source, you will need the following tools:
- `git`
- `conda` (anaconda or miniconda)

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone #place holder
```

**Step 2:** Create and activate a `conda` environment named `TSB-AD`.

```bash
conda create -n HYDRA python=3.11    # Currently we support python>=3.8, up to 3.12
conda activate HYDRA
```

**Step 3:** Install the dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

If you have problem installing `torch` using pip, try the following:
```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

If you have problem installing `hnswlib` using pip, try the following:
```bash
conda install -c conda-forge hnswlib
```

**Step 4:** Install the package:
```bash
pip install -e .
```

## Basic Usage

See Example in `HYDRA/main.py`

```bash
python -m HYDRA.main
```

Or the following example on how to run HYDRA in 10 lines of code:
```bash
import HYDRA_loader as loader

def run_HYDRA(data, winsize=30):
    model = loader.HYDRA(winsize, mode='approx')
    model.compress_and_score_multi(data)
    ens_score,_ = model.ensemble_maxpool_windows()
    return ens_score.ravel()
  
X = np.sin(np.linspace(0, 20, 500)) + 0.1 * np.random.randn(500)
score = run_HYDRA(X)
```
