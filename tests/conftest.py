"""Shared pytest fixtures."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bekaasense.settings")


@pytest.fixture(scope="session")
def synthetic_df():
    """Produce a small synthetic DataFrame inline (no disk dependency)."""
    rows = []
    for station in ["Ammik", "Doures"]:
        for year in [2020, 2021, 2022, 2023]:
            for month in range(1, 13):
                rows.append({
                    "station": station,
                    "year": year,
                    "month": month,
                    "temp_avg": 10 + 10 * ((month - 4) % 12) / 12,
                    "temp_max": 20 + 8 * ((month - 4) % 12) / 12,
                    "temp_min": 3 + 8 * ((month - 4) % 12) / 12,
                    "precip_sum": 60 if month in (1, 2, 12) else 10,
                })
    return pd.DataFrame(rows)
