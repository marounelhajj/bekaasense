# Limitations

An honest accounting. The project is deliberately designed to be useful
*despite* these limitations, not to hide them.

## 1. Short historical record

The ~10-year training record is far shorter than the 20–30 years
conventionally required for confident climatic trend detection.
Consequences:

- **Mann–Kendall trend tests are not yet statistically significant**
  on any station (p > 0.05 for all four). The magnitudes of the trends
  are consistent with regional climate expectations, but we cannot
  confirm them from this dataset alone.
- **Long-horizon forecasts (> 3 years) are reported with visibly widening
  intervals** — the interval widening is enforced by construction in
  `model_engine.inference.forecast_station` so no user sees a precise
  long-horizon point estimate.

## 2. Limited spatial coverage

Four stations cover ~5,000 km² of the Bekaa Valley unevenly. **No
predictions are made for unmonitored locations.** The API only accepts
the four station names.

## 3. De Martonne is a simplified index

The formula `I = 12P / (T + 10)` captures first-order rainfall-vs-
temperature aridity but ignores:

- **Wind speed** and **solar radiation**, both of which drive
  evapotranspiration.
- **Soil moisture memory** beyond what the SPI partially captures.
- **Actual evapotranspiration** (required for more sophisticated indices
  like SPEI or Thornthwaite water balance).

SPEI is a planned future enhancement once longer temperature series
become available.

## 4. Tal Amara sensor fragmentation

Tal Amara's temperature series spans three separate sensor deployments
(TA1, TA2, TA3). Per-sensor calibration offsets are not available. The
pipeline treats the station-month climatology as a fallback, which
introduces an unmeasured but bounded bias. Users who need Tal-Amara-
specific absolute temperatures should treat the values as station-level
trend indicators rather than calibrated measurements.

## 5. Class imbalance

Hyper-arid months dominate (~80 %). We address this with class weighting
and optional SMOTE, and report both weighted and macro F1 so the reader
can see how much of the score comes from the majority class. **Humid and
Sub-humid predictions are fundamentally less reliable** given how rarely
they occur in the training data.

## 6. External forcings not modelled

The feature set does not include large-scale circulation drivers (ENSO,
NAO, North Atlantic Oscillation, solar activity). Those features would
likely help but are not available at monthly resolution for this region
without external API dependencies that compromise reproducibility.

## 7. Distribution shift

The training data (2015–2022) may not represent the climate of 2030+.
Climate change itself is a distribution shift, and no amount of
modelling can fully compensate. Mitigations:

- Bootstrapped intervals widen with horizon (`inference.forecast_station`
  enforces +5 % spread per step).
- Residuals are stratified by year so degradation is visible
  (`results/metrics/residuals_by_year.csv`).
- The dashboard's limitations section discloses this risk to end users.

## 8. Scope of claims

BekaaSense is a **decision support** tool. Its intended use is to
inform, not replace, expert judgement by agronomists, hydrologists,
and policy analysts. It does not model crop yields, soil dynamics,
irrigation economics, or socio-economic outcomes — all of which would
be needed for operational agricultural planning.
