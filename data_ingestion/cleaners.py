"""
Data cleaning and imputation.

The known data challenges (documented in the problem formulation) are:

1. **Missing values** in Ras Baalbeck's earliest months.
2. **Format heterogeneity** across stations (handled in :mod:`loaders`).
3. **Temporal fragmentation** in Tal Amara's sensor deployments.

All imputation decisions are flagged in the returned DataFrame via an
``imputed_*`` boolean column so downstream consumers (and the dashboard)
can honestly surface "this reading was imputed" to end users — a direct
responsible-ML requirement (RM2 bias disclosure, RM3 data integrity).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("bekaasense")


def add_imputation_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add ``imputed_<col>`` boolean flags before imputation runs."""
    for c in cols:
        df[f"imputed_{c}"] = df[c].isna()
    return df


def impute_station_month(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Impute missing values using the station-specific monthly climatology.

    For each missing row, we substitute the median value observed for the
    same (station, month) pair across other years. This preserves seasonal
    structure far better than a global mean and avoids the common mistake
    of using a rolling window that would leak future information.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``station`` and ``month`` columns.
    cols : list of str
        Columns to impute.
    """
    df = df.copy()
    df = add_imputation_flags(df, cols)

    for col in cols:
        # Station-month climatology (median is robust to outliers)
        climatology = df.groupby(["station", "month"])[col].transform("median")
        df[col] = df[col].fillna(climatology)

        # Fallback: station-wide median
        station_median = df.groupby("station")[col].transform("median")
        df[col] = df[col].fillna(station_median)

        # Last-resort: overall median (shouldn't trigger in practice)
        df[col] = df[col].fillna(df[col].median())

    return df


def flag_outliers_iqr(df: pd.DataFrame, col: str,
                      k: float = 3.0) -> pd.DataFrame:
    """Flag (but do NOT remove) outliers using station-scoped IQR.

    Climate extremes are often real — a 280 mm month in the Bekaa is
    meteorologically plausible. We flag them for inspection but leave the
    values in place; removing them would bias the model toward an
    artificially calm climatology.
    """
    df = df.copy()
    flag_col = f"outlier_{col}"
    df[flag_col] = False
    for station, group in df.groupby("station"):
        q1, q3 = group[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (df["station"] == station) & ((df[col] < lo) | (df[col] > hi))
        df.loc[mask, flag_col] = True
    n_flagged = df[flag_col].sum()
    if n_flagged:
        log.info("Flagged %d outlier(s) in %s (k=%.1f)", n_flagged, col, k)
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full clean: impute, flag outliers, enforce sane bounds."""
    numeric = ["temp_avg", "temp_max", "temp_min", "precip_sum"]
    df = impute_station_month(df, numeric)

    # Sanity bounds (hard clip for obvious sensor faults)
    df["precip_sum"] = df["precip_sum"].clip(lower=0, upper=500)
    df["temp_avg"] = df["temp_avg"].clip(lower=-15, upper=40)
    df["temp_max"] = df["temp_max"].clip(lower=-10, upper=50)
    df["temp_min"] = df["temp_min"].clip(lower=-25, upper=35)

    for col in ["precip_sum", "temp_avg"]:
        df = flag_outliers_iqr(df, col)

    return df
