"""
Aridity and drought index computation.

Exposes two pure-function computations used across the project:

* :func:`de_martonne_monthly` — Monthly De Martonne Aridity Index.
* :func:`spi` — Standardized Precipitation Index (rolling 3/6/12-month).

References
----------
De Martonne, E. (1926). Une nouvelle fonction climatologique: l'indice
d'aridité. La Météorologie, 2, 449–458.

McKee, T.B., Doesken, N.J., Kleist, J. (1993). The Relationship of Drought
Frequency and Duration to Time Scales. Eighth Conf. on Applied Climatology.

Notes
-----
The monthly De Martonne formulation multiplies by 12 to annualize the
denominator — the direct P/(T+10) form is an annual formula; for monthly
resolution we use I_m = 12 * P_m / (T_m + 10). This is the convention used
in Mediterranean climate literature (Baltas, 2007; Croitoru et al., 2013).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

# ---------------------------------------------------------------------------
# De Martonne
# ---------------------------------------------------------------------------

#: Classification thresholds (inclusive lower bound). Used both in regression
#: post-processing and classification labels.
DE_MARTONNE_ZONES = [
    ("Hyper-arid", -np.inf, 5.0),
    ("Arid", 5.0, 10.0),
    ("Semi-arid", 10.0, 20.0),
    ("Sub-humid", 20.0, 30.0),
    ("Humid", 30.0, np.inf),
]


def de_martonne_monthly(precip_mm: pd.Series, temp_c: pd.Series) -> pd.Series:
    """Monthly De Martonne Aridity Index.

    Parameters
    ----------
    precip_mm : pd.Series
        Monthly precipitation in millimetres.
    temp_c : pd.Series
        Monthly mean temperature in degrees Celsius.

    Returns
    -------
    pd.Series
        Monthly aridity index. Higher = wetter. Dimensionless.

    Notes
    -----
    Temperatures below -10°C would cause a division blow-up; in the Bekaa
    Valley the minimum recorded monthly mean is roughly -2°C, so the
    formulation is safe, but we clip defensively at T >= -9.5°C.
    """
    t_safe = np.maximum(temp_c.astype(float), -9.5)
    return (12.0 * precip_mm.astype(float)) / (t_safe + 10.0)


def classify_aridity(index_values: pd.Series) -> pd.Series:
    """Classify De Martonne values into ordinal aridity zones."""
    labels = pd.Series(index=index_values.index, dtype="object")
    for name, lo, hi in DE_MARTONNE_ZONES:
        mask = (index_values >= lo) & (index_values < hi)
        labels.loc[mask] = name
    return labels


# ---------------------------------------------------------------------------
# Standardized Precipitation Index (SPI)
# ---------------------------------------------------------------------------

def _fit_gamma(values: np.ndarray) -> tuple[float, float, float]:
    """Fit a 2-parameter gamma distribution to non-zero values.

    Returns (shape, loc=0, scale). Uses MLE; falls back to method of moments
    if MLE fails (small sample, degenerate series).
    """
    nonzero = values[values > 0]
    if len(nonzero) < 5:
        return (1.0, 0.0, max(nonzero.mean() if len(nonzero) else 1.0, 1e-6))
    try:
        shape, loc, scale = gamma.fit(nonzero, floc=0)
        return shape, loc, scale
    except Exception:
        mean = nonzero.mean()
        var = nonzero.var()
        scale = var / mean if mean > 0 else 1.0
        shape = mean / scale if scale > 0 else 1.0
        return shape, 0.0, scale


def spi(precip_monthly: pd.Series, window: int = 3) -> pd.Series:
    """Standardized Precipitation Index over a rolling window.

    The window-aggregated precipitation is fit to a 2-parameter gamma
    distribution. The resulting CDF is transformed via the inverse standard
    normal CDF, yielding an anomaly z-score comparable across stations.

    Parameters
    ----------
    precip_monthly : pd.Series
        Monthly precipitation (mm). Index should be a monthly DatetimeIndex
        for time-aware grouping; a plain integer index works too.
    window : int
        Rolling window size in months. Commonly 3 (short-term drought),
        6 (seasonal), or 12 (long-term).

    Returns
    -------
    pd.Series
        SPI values. Leading ``window-1`` entries are NaN by construction.
    """
    rolled = precip_monthly.rolling(window=window, min_periods=window).sum()
    valid = rolled.dropna().values
    if len(valid) < window + 1:
        return pd.Series(np.nan, index=precip_monthly.index)

    shape, loc, scale = _fit_gamma(valid)

    # Probability of zero — adjusted CDF (McKee et al., 1993 Eq. 4)
    p_zero = float((valid == 0).mean())

    def _transform(x: float) -> float:
        if np.isnan(x):
            return np.nan
        if x == 0:
            cdf = p_zero
        else:
            cdf = p_zero + (1 - p_zero) * gamma.cdf(x, shape, loc=loc, scale=scale)
        # Clip to avoid inf at the distribution tails
        cdf = min(max(cdf, 1e-6), 1 - 1e-6)
        return float(norm.ppf(cdf))

    return rolled.apply(_transform)
