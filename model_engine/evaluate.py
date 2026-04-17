"""
Evaluation metrics and residual analysis.

All models (baselines + ML) are benchmarked on the same temporal test set
using the same functions in this module, so comparisons are fair by
construction — a direct rubric requirement (TM2A + TM5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(set(list(y_true) + list(y_pred)))
    return {
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                     labels=labels, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "report": classification_report(y_true, y_pred, zero_division=0,
                                        output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Residual analysis — stratified by year (RM4 robustness)
# ---------------------------------------------------------------------------

def residuals_by_year(df: pd.DataFrame, y_col: str = "de_martonne",
                     pred_col: str = "prediction") -> pd.DataFrame:
    """Per-year residual summary — detects model degradation over time."""
    df = df.copy()
    df["residual"] = df[y_col] - df[pred_col]
    return (
        df.groupby("year")
          .agg(n=("residual", "size"),
               mean=("residual", "mean"),
               std=("residual", "std"),
               mae=("residual", lambda s: float(np.mean(np.abs(s)))))
          .reset_index()
    )


def residuals_by_station(df: pd.DataFrame, y_col: str = "de_martonne",
                        pred_col: str = "prediction") -> pd.DataFrame:
    """Per-station residual summary — detects station-specific bias (RM2)."""
    df = df.copy()
    df["residual"] = df[y_col] - df[pred_col]
    return (
        df.groupby("station")
          .agg(n=("residual", "size"),
               mean=("residual", "mean"),
               std=("residual", "std"),
               mae=("residual", lambda s: float(np.mean(np.abs(s)))))
          .reset_index()
    )


# ---------------------------------------------------------------------------
# Robustness — interval coverage check (RM4)
# ---------------------------------------------------------------------------

def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of true values falling within [lo, hi]. Should be close to
    the nominal 1 - alpha used when generating the interval."""
    y_true = np.asarray(y_true, dtype=float)
    return float(((y_true >= lo) & (y_true <= hi)).mean())


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def save_leaderboard(rows: list[dict], out_path: Path | str) -> pd.DataFrame:
    """Write a comparison table of all models. Used by the dashboard and
    by :file:`results/metrics/leaderboard.csv`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df
