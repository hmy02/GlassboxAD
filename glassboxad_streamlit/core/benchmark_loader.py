from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


METRIC_COLUMNS_DEFAULT_ORDER = [
    "VUS-PR",
    "AUC-PR",
    "PA-F1",
    "Standard-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
    "VUS-ROC",
    "AUC-ROC",
    "Event-based-F1",  # keep duplicates harmless
]


@dataclass(frozen=True)
class TSBId:
    series_id: str
    seq: str | None
    dataset: str | None
    ds_id: str | None
    domain: str | None


def parse_tsb_ad_series_id(series_id: str) -> TSBId:
    """Parse IDs like: 001_NAB_id_1_Facility_tr_1007_1st_2014

    We extract:
      - seq: first token
      - dataset: second token
      - ds_id: the token after literal 'id'
      - domain: token after ds_id

    If the pattern doesn't match, we still return the original series_id.
    """
    try:
        parts = str(series_id).split("_")
        seq = parts[0] if len(parts) > 0 else None
        dataset = parts[1] if len(parts) > 1 else None
        ds_id = None
        domain = None
        if "id" in parts:
            i = parts.index("id")
            if i + 1 < len(parts):
                ds_id = parts[i + 1]
            if i + 2 < len(parts):
                domain = parts[i + 2]
        return TSBId(series_id=str(series_id), seq=seq, dataset=dataset, ds_id=ds_id, domain=domain)
    except Exception:
        return TSBId(series_id=str(series_id), seq=None, dataset=None, ds_id=None, domain=None)


def _metric_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "Time"]
    # Keep a stable, user-friendly order when possible
    ordered = []
    for c in METRIC_COLUMNS_DEFAULT_ORDER:
        if c in cols and c not in ordered:
            ordered.append(c)
    for c in cols:
        if c not in ordered:
            ordered.append(c)
    return ordered


def load_tsb_ad_benchmark(data_root: str | Path, mode: str = "uni") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load TSB-AD-style benchmark results.

    Expected structure:
      data_root/
        uni/
          <method>.csv
          ...
        multi/
          <method>.csv
          ...

    Each CSV:
      - index: series_id (e.g., 001_NAB_id_1_Facility_tr_...)
      - columns: Time + metrics (AUC-PR, VUS-PR, ...)

    Returns:
      eval_long: columns [series_id, dataset, domain, method, metric, value]
      runtime:   columns [series_id, dataset, domain, method, runtime]
    """
    data_root = Path(data_root)
    folder = data_root / mode
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    eval_rows = []
    rt_rows = []

    for p in csvs:
        method = p.stem
        df = pd.read_csv(p, index_col=0)
        df.index = df.index.astype(str)
        # Normalize runtime column name
        if "Time" not in df.columns:
            # allow lowercase or other variants
            for cand in ["time", "runtime", "Runtime", "TIME"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "Time"})
                    break

        # Parse ids
        parsed = [parse_tsb_ad_series_id(s) for s in df.index]
        ds = [x.dataset for x in parsed]
        dom = [x.domain for x in parsed]

        # Runtime rows
        if "Time" in df.columns:
            for sid, dsn, domain, t in zip(df.index, ds, dom, df["Time"].tolist()):
                rt_rows.append([sid, dsn, domain, method, t])

        # Metrics
        metrics = _metric_columns(df)
        for metric in metrics:
            if metric not in df.columns:
                continue
            vals = df[metric]
            for sid, dsn, domain, v in zip(df.index, ds, dom, vals.tolist()):
                eval_rows.append([sid, dsn, domain, method, metric, v])

    eval_long = pd.DataFrame(
        eval_rows, columns=["series_id", "dataset", "domain", "method", "metric", "value"]
    )
    runtime = pd.DataFrame(rt_rows, columns=["series_id", "dataset", "domain", "method", "runtime"])

    # Coerce numeric
    eval_long["value"] = pd.to_numeric(eval_long["value"], errors="coerce")
    runtime["runtime"] = pd.to_numeric(runtime["runtime"], errors="coerce")

    return eval_long, runtime


def available_methods(data_root: str | Path, mode: str) -> list[str]:
    folder = Path(data_root) / mode
    if not folder.exists():
        return []
    return sorted([p.stem for p in folder.glob("*.csv")])
