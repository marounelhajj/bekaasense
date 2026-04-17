"""
Non-AI baselines.

The announcement requires at least one explicit non-AI baseline (TM2A);
BekaaSense ships **three**, to support a genuinely fair comparison:

1. :class:`LinearTrendBaseline` — per-station OLS of annual De Martonne
   index on year, projected forward.
2. :class:`SarimaBaseline` — SARIMA(1,0,1)(1,1,1,12) on the monthly index
   per station, using :mod:`statsmodels`.
3. :class:`RuleBaseline` — a climatological rule of the form
   "IF rolling precipitation < station historical mean AND T° > station
   historical mean THEN 'drier-than-normal'" — a strict classification
   baseline for the aridity-zone task.

All baselines implement the minimal ``fit(X, y)`` / ``predict(X)``
interface so they plug into the same evaluation harness as the ML models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Linear trend on annual aggregate
# ---------------------------------------------------------------------------

@dataclass
class LinearTrendBaseline:
    """Ordinary least-squares linear extrapolation on annual index.

    For each station, the annual mean De Martonne index is regressed on
    ``year``; the fitted line is used as the monthly forecast for every
    month of a projected year. Intentionally crude — the point is to show
    whether the ML models actually beat naive extrapolation.
    """

    coef_: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "LinearTrendBaseline":
        for station, g in df.groupby("station"):
            yearly = g.groupby("year")["de_martonne"].mean().reset_index()
            if len(yearly) < 2:
                self.coef_[station] = (0.0, float(yearly["de_martonne"].mean()))
                continue
            x = yearly["year"].values.astype(float)
            y = yearly["de_martonne"].values.astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            self.coef_[station] = (float(slope), float(intercept))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = np.empty(len(df))
        for i, row in enumerate(df.itertuples(index=False)):
            slope, intercept = self.coef_.get(row.station, (0.0, 0.0))
            pred[i] = slope * row.year + intercept
        return pred


# ---------------------------------------------------------------------------
# 2. SARIMA — classical time-series baseline
# ---------------------------------------------------------------------------

@dataclass
class SarimaBaseline:
    """SARIMA(1,0,1)(1,1,1,12) per station on the monthly De Martonne series.

    The seasonal period is 12 months. Order is fixed — auto-selection
    would constitute hyperparameter tuning, which we reserve for the ML
    models to keep the baseline honestly simple.
    """

    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)
    fitted_: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "SarimaBaseline":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        for station, g in df.groupby("station"):
            series = g.sort_values(["year", "month"])["de_martonne"].values
            if len(series) < 24:  # SARIMA needs >= 2 full seasons
                self.fitted_[station] = ("mean", float(series.mean()))
                continue
            try:
                res = SARIMAX(
                    series, order=self.order, seasonal_order=self.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
                self.fitted_[station] = ("sarima", res)
            except Exception:  # pragma: no cover - numerical fallback
                self.fitted_[station] = ("mean", float(series.mean()))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Expects df grouped by (station, year, month) in forecast order."""
        out = np.empty(len(df))
        idx = 0
        for station, g in df.groupby("station", sort=False):
            n = len(g)
            kind = self.fitted_.get(station, ("mean", 0.0))
            if kind[0] == "sarima":
                fcast = kind[1].forecast(steps=n)
                out[idx:idx + n] = np.asarray(fcast)
            else:
                out[idx:idx + n] = kind[1]
            idx += n
        return out


# ---------------------------------------------------------------------------
# 3. Rule-based classifier
# ---------------------------------------------------------------------------

@dataclass
class RuleBaseline:
    """A hand-crafted climatological rule.

    Logic, per station:

    ``IF precip_roll3 < station_mean(precip_roll3) AND temp_avg >
    station_mean(temp_avg) THEN 'drier-than-normal' ELSE 'normal-or-wetter'``

    This is a **classification** baseline; aridity zone is inferred from
    station climatology rather than via the De Martonne formula, giving
    the ML models a non-trivial comparator.
    """

    station_means_: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "RuleBaseline":
        for station, g in df.groupby("station"):
            self.station_means_[station] = {
                "precip_roll3": float(g["precip_roll3"].mean()),
                "temp_avg": float(g["temp_avg"].mean()),
            }
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns labels ``'drier-than-normal'`` / ``'normal-or-wetter'``."""
        labels: list[str] = []
        for row in df.itertuples(index=False):
            m = self.station_means_.get(row.station, {})
            if not m:
                labels.append("normal-or-wetter")
                continue
            drier = (row.precip_roll3 < m["precip_roll3"]
                     and row.temp_avg > m["temp_avg"])
            labels.append("drier-than-normal" if drier else "normal-or-wetter")
        return np.array(labels)
