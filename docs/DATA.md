# Data

## Sources

Four weather stations in Lebanon's Bekaa Valley:

| Station       | Temporal coverage     | Records | Notes                         |
|---------------|-----------------------|--------:|-------------------------------|
| Ammik         | Jan 2015 – present    |    ~109 | Highest elevation             |
| Doures        | Jan 2015 – present    |    ~125 | Central Bekaa                 |
| Ras Baalbeck  | Nov 2015 – present    |    ~122 | Most arid                     |
| Tal Amara     | 2004 – 2025           |    400+ | 3 fragmented sensor deployments |

Variables: monthly mean / max / min temperature (°C) and monthly
precipitation (mm).

## Canonical schema

```
station       str    One of {Ammik, Doures, Ras_Baalbeck, Tal_Amara}
year          int
month         int    1..12
temp_avg      float  °C
temp_max      float  °C
temp_min      float  °C
precip_sum    float  mm
```

Derived columns (added by `data_ingestion.features.build_features`):

```
de_martonne   float  12 * P / (T + 10)
aridity_zone  str    {Hyper-arid, Arid, Semi-arid, Sub-humid, Humid}
month_sin     float  sin(2π * month / 12)
month_cos     float  cos(2π * month / 12)
dm_lag1..3    float  De Martonne at t-1, t-2, t-3
precip_lag1   float  Precipitation at t-1
temp_lag1     float  Temperature at t-1
precip_roll{3,6,12}  Rolling precipitation means (past-only)
dm_roll{3,6}         Rolling De Martonne means (past-only)
temp_roll3           3-month temperature roll (past-only)
spi{3,6,12}          Standardized Precipitation Index
spi3_lag1, spi6_lag1 Lagged SPI
stn_Ammik..Tal_Amara Station one-hot (0/1)
imputed_<col>        True if <col> was imputed at this row
outlier_<col>        True if <col> flagged by station-scoped IQR
```

## Known challenges

**Missing values.** Ras Baalbeck is missing several months in late 2015.
Handled by station-month median imputation (see `cleaners.py`); flagged
in `imputed_<col>`.

**Format heterogeneity.** Some station files were delivered as XML-
SpreadsheetML even though their filename said `.xls`. The alias-tolerant
loader handles the common layouts.

**Tal Amara fragmentation.** Three separate sensor deployments
(TA1 / TA2 / TA3) had to be stitched. Per-sensor calibration offsets are
not available, so we treat the station-month climatology as the fallback
— this introduces an unmeasured but bounded bias that is disclosed in
`docs/LIMITATIONS.md`.

## Demo dataset

`scripts/generate_synthetic.py` produces a calibrated demo dataset that
reproduces the EDA correlations observed in the real data:

- r(precip_sum, de_martonne) ≈ 0.98
- r(temp_avg, de_martonne)   ≈ -0.56
- Hyper-arid dominates across all stations
- Ammik shows a positive precipitation trend; the other three show
  negative trends
- Mann–Kendall p > 0.05 everywhere (the ~10-year record is too short)

This lets reviewers run the full pipeline end-to-end in one command even
without access to the raw station files.

## Licensing

Raw station data is held under the data-provider's licence; the
repository's `.gitignore` excludes `data/raw/` by default. The processed
`bekaa_valley_clean.csv` contains only the canonical schema — no raw
sensor IDs, deployment metadata, or observer notes.
