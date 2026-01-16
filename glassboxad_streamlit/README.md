# GlassboxAD Streamlit App

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected repo layout for HYDRA
Put this Streamlit folder **inside** your project repo, so the repo root contains:

- `HYDRA/`
- `TSB-AD/` (optional; used for auto window-size helper)
- `glassboxad_streamlit/` (this app)

The HYDRA page uses the same logic as the original FastAPI tool (hierarchy + 3D viewer).

## Data

- Put CSV files in `data/preloaded_ts/` with columns (recommended):
  - `Data`
  - `Label` (required for VUS-PR)

If your CSV uses different column names, the app will fall back to the first numeric column as `Data`, and tries common label names (Label/label/y/is_anomaly/...)

## Benchmark

Point `Results path` to a directory or CSV containing evaluation + runtime results. The loader is schema-flexible.

Recommended long format:

- evaluation: `dataset, method, metric, value`
- runtime: `dataset, method, runtime`

