"""
Feature engineering with explicit leakage control.

The design rule — enforced here and verified in :mod:`tests.test_features` —
is that every feature visible at inference time for date ``t`` depends only
on observations with date ``< t`` (lags) or the deterministic calendar
(seasonal encodings). No rolling window includes the current month's value
without explicit lag.

The feature set is derived from the EDA findings documented in
``notebooks/01_eda.ipynb`` (correlations with the De Martonne target):

=====================  ======  =======================================
Feature                |r|     Role
=====================  ======  =======================================
precip_sum             0.98    Dominant predictor
dm_roll3               0.72    3-month moisture memory
precip_roll3           0.72    Rolling precipitation
temp_avg               0.56    Secondary climatic driver
spi3                   0.63    Drought-context index
dm_lag1                0.46    Short-term persistence
month_sin/month_cos    —       Seasonal encoding
=====================  ======  =======================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_ingestion.indices import (
    DE_MARTONNE_ZONES,
    classify_aridity,
    de_martonne_monthly,
    spi,
)


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

SEASONAL_FEATURES = ["month_sin", "month_cos"]
LAG_FEATURES = ["dm_lag1", "dm_lag2", "dm_lag3",
                "precip_lag1", "temp_lag1"]
ROLLING_FEATURES = ["precip_roll3", "precip_roll6", "precip_roll12",
                    "dm_roll3", "dm_roll6", "temp_roll3"]
CLIMATE_FEATURES = ["spi3", "spi6", "spi12"]
CURRENT_FEATURES = ["precip_sum", "temp_avg", "temp_max", "temp_min"]

#: Features safe to use for *forecasting future* De Martonne. These use
#: only the past — required for honest out-of-sample evaluation.
FORECAST_FEATURE_SET = (
    SEASONAL_FEATURES + LAG_FEATURES + ROLLING_FEATURES
    + ["spi3_lag1", "spi6_lag1"]
)

#: Features safe to use for *nowcasting* — classifying the current month
#: once its precipitation and temperature are observed.
NOWCAST_FEATURE_SET = (
    SEASONAL_FEATURES + CURRENT_FEATURES + LAG_FEATURES
    + ROLLING_FEATURES + CLIMATE_FEATURES
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical month encoding — avoids the Dec/Jan discontinuity."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def _lags_and_rolls(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station lag and rolling features. Lags shift by 1+ months so no
    row contains its own target on the right-hand side."""
    out = []
    for station, g in df.groupby("station", sort=False):
        g = g.sort_values(["year", "month"]).copy()

        g["dm_lag1"] = g["de_martonne"].shift(1)
        g["dm_lag2"] = g["de_martonne"].shift(2)
        g["dm_lag3"] = g["de_martonne"].shift(3)
        g["precip_lag1"] = g["precip_sum"].shift(1)
        g["temp_lag1"] = g["temp_avg"].shift(1)

        # Rolling averages — the rolling includes the current row, so we
        # shift(1) BEFORE rolling to keep everything past-only.
        for w in (3, 6, 12):
            g[f"precip_roll{w}"] = g["precip_sum"].shift(1).rolling(w, min_periods=1).mean()
        for w in (3, 6):
            g[f"dm_roll{w}"] = g["de_martonne"].shift(1).rolling(w, min_periods=1).mean()
        g["temp_roll3"] = g["temp_avg"].shift(1).rolling(3, min_periods=1).mean()

        # SPI features (already use rolling precipitation; compute lagged copy)
        g["spi3"] = spi(g["precip_sum"], window=3)
        g["spi6"] = spi(g["precip_sum"], window=6)
        g["spi12"] = spi(g["precip_sum"], window=12)
        g["spi3_lag1"] = g["spi3"].shift(1)
        g["spi6_lag1"] = g["spi6"].shift(1)

        out.append(g)

    return pd.concat(out, ignore_index=True)


def build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature matrix from a cleaned canonical DataFrame.

    The pipeline is:
    1. Compute De Martonne index (target).
    2. Add cyclical month encoding.
    3. Add per-station lags + rollings (shift-1 to prevent leakage).
    4. Compute SPI-3/6/12 and their lagged copies.
    5. Add aridity zone label + station one-hot encoding.
    """
    df = cleaned.copy()
    df["de_martonne"] = de_martonne_monthly(df["precip_sum"], df["temp_avg"])
    df["aridity_zone"] = classify_aridity(df["de_martonne"])

    df = _seasonal(df)
    df = _lags_and_rolls(df)

    # One-hot encode station for cross-station models
    station_dummies = pd.get_dummies(df["station"], prefix="stn",
                                    drop_first=False).astype(int)
    df = pd.concat([df, station_dummies], axis=1)

    return df


def temporal_split(df: pd.DataFrame,
                   train_end: str = "2021-12-31",
                   val_end: str = "2022-12-31") -> tuple[pd.DataFrame, ...]:
    """Strict temporal train / validation / test split.

    Training: ``<= train_end``. Validation: ``(train_end, val_end]``.
    Test: ``> val_end``. The cutoffs default to the values specified in the
    problem formulation (train 2015-2021, val 2022, test 2023+).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Guardrail — fails loudly if any test row predates any train row.

    Called from tests. This is cheap insurance against somebody accidentally
    shuffling splits and getting impressive-looking numbers.
    """
    if train.empty or test.empty:
        return
    assert train["date"].max() < test["date"].min(), (
        "LEAKAGE DETECTED: train contains a date >= earliest test date. "
        f"train_max={train['date'].max()}, test_min={test['date'].min()}"
    )
