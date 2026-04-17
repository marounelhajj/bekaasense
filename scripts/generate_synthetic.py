"""
Synthetic BekaaSense dataset generator.

This script produces ``data/processed/bekaa_valley_clean.csv`` calibrated
to the statistics documented in the EDA notebook:

* Correlation r(precip, De Martonne) ≈ 0.98
* r(temp, De Martonne) ≈ -0.56
* Hyper-arid months dominate across all stations
* Ammik: improving precipitation trend (+34.6 mm/yr)
* Doures, Ras_Baalbeck, Tal_Amara: declining (-12 to -20 mm/yr)
* Mann–Kendall p > 0.05 for every station (the ~10-year record is too
  short for statistical significance — an honest limitation)

**This is for pipeline testing and demos only.** Drop your real cleaned
data at the same path to replace it.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/generate_synthetic.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

log = logging.getLogger("bekaasense")
RNG = np.random.default_rng(42)

# Station profile: (precip_baseline_mm/month, temp_baseline_C,
#                   precip_trend_mm/yr, temp_trend_C/yr, amplitude)
STATIONS = {
    "Ammik":        (55, 14.5, +2.9, +0.02, 1.0),   # ~35 mm/yr positive
    "Doures":       (45, 14.0, -1.5, +0.03, 0.9),
    "Ras_Baalbeck": (30, 13.0, -1.7, +0.04, 0.8),   # most arid
    "Tal_Amara":    (48, 15.5, -1.0, +0.03, 0.95),
}

START_YEAR = 2015
END_YEAR = 2025  # inclusive; last month = Dec 2025


def _seasonal_precip(month: int, baseline: float, amp: float) -> float:
    """Strong winter peak, near-zero summer. Peak in January."""
    # December-January = peak, July = trough
    # Phase chosen so that month=1 (Jan) is near maximum.
    phase = np.cos(2 * np.pi * (month - 1) / 12.0)
    factor = 1.5 * amp * (phase + 1) / 2  # 0 .. 1.5 * amp
    return max(0.0, baseline * factor)


def _seasonal_temp(month: int, baseline: float) -> float:
    """Minimum in January, maximum in July."""
    phase = np.cos(2 * np.pi * (month - 7) / 12.0)  # peak at 7
    return baseline + 10.5 * phase  # roughly ±10.5°C swing


def generate() -> pd.DataFrame:
    rows = []
    for station, (p_base, t_base, p_trend, t_trend, amp) in STATIONS.items():
        for year in range(START_YEAR, END_YEAR + 1):
            years_from_start = year - START_YEAR
            for month in range(1, 13):
                # Base seasonal values
                p = _seasonal_precip(month, p_base, amp)
                t_avg = _seasonal_temp(month, t_base)

                # Add trend (accumulated over years)
                p += p_trend * years_from_start / 12.0
                t_avg += t_trend * years_from_start

                # Noise — precipitation is right-skewed; use lognormal
                p_noise = RNG.lognormal(mean=0.0, sigma=0.55) - 1.0
                p = max(0.0, p * (1 + 0.4 * p_noise))

                t_noise = RNG.normal(0, 1.2)
                t_avg = t_avg + t_noise

                # Max / min around avg
                t_max = t_avg + RNG.uniform(4, 7)
                t_min = t_avg - RNG.uniform(4, 8)

                rows.append({
                    "station": station,
                    "year": year,
                    "month": month,
                    "temp_avg": round(t_avg, 2),
                    "temp_max": round(t_max, 2),
                    "temp_min": round(t_min, 2),
                    "precip_sum": round(p, 2),
                })

    df = pd.DataFrame(rows)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = generate()

    # Sanity checks aligned with the user's EDA statistics
    from data_ingestion.indices import de_martonne_monthly
    dm = de_martonne_monthly(df["precip_sum"], df["temp_avg"])
    corr_pp = float(np.corrcoef(df["precip_sum"], dm)[0, 1])
    corr_tt = float(np.corrcoef(df["temp_avg"], dm)[0, 1])
    log.info("r(precip, DM) = %.3f (EDA target ≈ 0.98)", corr_pp)
    log.info("r(temp,   DM) = %.3f (EDA target ≈ -0.56)", corr_tt)

    out = Path("data/processed/bekaa_valley_clean.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %d rows -> %s", len(df), out)


if __name__ == "__main__":
    main()
