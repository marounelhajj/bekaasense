"""
Station data loaders.

The BekaaSense raw dataset comprises four stations with *heterogeneous*
source formats — some are binary `.xls`, some are XML-SpreadsheetML
masquerading as `.xls`, and Tal Amara consists of multiple sensor
deployments that must be stitched.

This module provides loaders that return a *canonical schema* regardless of
the source format::

    station, year, month, temp_avg, temp_max, temp_min, precip_sum

Missing values propagate as NaN — imputation happens in :mod:`cleaners`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("bekaasense")

CANONICAL_COLS = [
    "station", "year", "month",
    "temp_avg", "temp_max", "temp_min", "precip_sum",
]

STATIONS = ["Ammik", "Doures", "Ras_Baalbeck", "Tal_Amara"]


def load_clean_csv(path: Path | str) -> pd.DataFrame:
    """Load the already-cleaned canonical CSV.

    This is the entry point the rest of the pipeline expects. Produce this
    CSV either by running :func:`build_canonical_dataset` on your raw
    station files, or by using the bundled synthetic generator for demos.
    """
    df = pd.read_csv(path)
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Canonical CSV at {path} is missing columns: {missing}. "
            "Re-run the data ingestion pipeline."
        )
    df["station"] = df["station"].astype(str)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["date"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=1)
    )
    return df.sort_values(["station", "date"]).reset_index(drop=True)


def build_canonical_dataset(raw_dir: Path | str, out_path: Path | str) -> pd.DataFrame:
    """Assemble the canonical monthly dataset from raw station files.

    The function looks for files named ``<Station>.xlsx`` (or ``.xls``) in
    ``raw_dir``. Each file must expose columns that include (in any order)::

        year, month, temp_avg, temp_max, temp_min, precip_sum

    Column-name heuristics tolerate common variations (``t_avg``,
    ``precipitation``, etc.). Unknown or exotic layouts should be handled
    by extending the alias map below.
    """
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)

    aliases = {
        "temp_avg": ["temp_avg", "t_avg", "tavg", "mean_temp", "temperature_avg"],
        "temp_max": ["temp_max", "t_max", "tmax", "max_temp"],
        "temp_min": ["temp_min", "t_min", "tmin", "min_temp"],
        "precip_sum": ["precip_sum", "precipitation", "precip", "rain", "rainfall"],
        "year": ["year", "yr"],
        "month": ["month", "mo", "mon"],
    }

    frames: list[pd.DataFrame] = []
    for station in STATIONS:
        candidates = list(raw_dir.glob(f"{station}*.xls*")) + \
                     list(raw_dir.glob(f"{station}*.csv"))
        if not candidates:
            log.warning("No raw file for station %s in %s", station, raw_dir)
            continue
        src = candidates[0]
        log.info("Loading %s from %s", station, src.name)

        df = (pd.read_excel(src) if src.suffix != ".csv" else pd.read_csv(src))
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Map aliases -> canonical
        rename = {}
        for canon, alts in aliases.items():
            for alt in alts:
                if alt in df.columns and canon not in rename.values():
                    rename[alt] = canon
                    break
        df = df.rename(columns=rename)

        df["station"] = station
        missing = [c for c in CANONICAL_COLS if c not in df.columns]
        if missing:
            log.error("Station %s missing canonical columns: %s", station, missing)
            continue
        frames.append(df[CANONICAL_COLS])

    if not frames:
        raise FileNotFoundError(
            f"No station files could be loaded from {raw_dir}. "
            "Use scripts/generate_synthetic.py for a demo dataset."
        )

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["station", "year", "month"]).reset_index(drop=True)
    full.to_csv(out_path, index=False)
    log.info("Wrote canonical dataset: %d rows -> %s", len(full), out_path)
    return full
