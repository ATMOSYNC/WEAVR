# HEPPI reference dataset (third-party, not redistributed)

Unlike the three sources in [data-sources.md](data-sources.md), this is not
an open data feed with a client that fetches it on demand — it's a static
research dataset a person has to download by hand, so this doc covers what
it is, why it's worth keeping around, and how to load it, rather than access
mechanics.

## What it is

HEPPI (Michael Angus, WCSSP India project, 2020-2021) pairs NCMRWF NEPS-G
ensemble forecasts with IMD observed rainfall, plus two bias-correction
variants of the forecast the original authors already produced:

| File | Contents | Shape |
|---|---|---|
| `IMD_observed.nc` | Observed rainfall | 334 samples × 135 lon × 129 lat |
| `NCMRWF_orig_forecast.nc` | Raw 23-member ensemble forecast | 334 × 23 members × 135 × 129 |
| `NCMRWF_UQM_forecast.nc` | Same, after univariate quantile mapping | same shape |
| `NCMRWF_EMOS_forecast.nc` | Same, after ensemble MOS | same shape |

Also included, as reference (not integrated into weavr, not exercised by
any test — read them if you're implementing Phase 1 bias correction):
`example_EMOS_fit.R` (their EMOS fit), `Generalized_QM.m` (their quantile
mapping), `MQM_WCSSP.zip` (5 Python files implementing multivariate bias
correction), `verif_hitmiss.m` / `rank_histograms.m` / `ver_rd.m`
(verification scripts: hit/miss fraction, rank histograms, reliability
diagrams), and `test_locations.xlsx` (9 named point locations with lat/lon
and their IMD grid-cell indices — Mumbai, Rajasthan, Kerala, Shimla, Delhi,
Hyderabad, Patna, Bhubaneswar, Meghalaya).

Source: <https://edata.bham.ac.uk/698/>. Not committed to this repo or its
Git history: it's a third-party research output of unclear redistribution
license, and the three forecast `.nc` files are ~1GB each. Download it
yourself and point the loaders below at the directory you put it in.

## Why it's worth loading at all

Confirmed by actually opening the files (not assumed from the README): the
lat/lon grid HEPPI was built on is, point for point, weavr's own locked
common grid (`weavr.grid.COMMON_LAT`/`COMMON_LON` — lat 6.5-38.5, lon
66.5-100.0, 0.25°, 129×135). That means it needs no regridding to be usable
here — a real, ensemble NCMRWF forecast (there's no other open access route
to NCMRWF ensemble data in Phase 0) already paired with IMD ground truth and
with two known bias-correction methods already applied, on exactly the grid
the rest of the project targets. Best use: a methodology reference for
Phase 1 bias correction (validate weavr's own quantile-mapping/EMOS
implementation by checking it can reproduce, or reasonably match, the UQM/
EMOS files here) and for later verification work (the .m scripts show what
scores the original project used).

## Calendar dates: resolved (Step 03)

The 334-sample axis was originally `Dimensions without coordinates: forecast`
in the raw files — no timestamps in the netCDF headers, as the original release
paired the data with a calendar-date lookup table (`IMD_dates.mat`) that was not
part of the public download.

**Resolution (`scripts/recover_heppi_dates.py` -> `docs/heppi-date-map.csv`):**
The calendar dates have been recovered empirically against IMD's 15-year (2006–2020)
daily 0.25° gridded rainfall climatology (`data/imd_seeps_climatology_jjas.zarr`)
and full-year 2018/2019 records:
- **JJAS 2018 (indices 0–119)**: Maps to 2018-06-02 .. 2018-09-30, with 2018-06-25 missing (120 days).
- **JJAS 2019 (indices 181–302)**: Maps to 2019-06-01 .. 2019-09-30 (122 days).
- **Confidence**: **242 of 242 (100.0%)** JJAS days match their hypothesized date as the
  unique rank-1 lowest MAE among all 1,830 candidate days.
- **Shift sensitivity**: Shifting by -1 day or +1 day drops matches to **0 / 241** and
  **0 / 240** respectively, proving zero calendar ambiguity.
- **Forecast alignment**: Pearson correlation of the NCMRWF ensemble mean with IMD observations
  peaks at **r = 0.5946** at index offset 0 (vs. 0.4966 at -1, 0.4462 at +1, 0.3455 at -2, 0.3156 at +2),
  confirming forecast index $i$ validates against observation index $i$.
- **Post-monsoon indices (120–180 and 303–333)**: 303–333 match Oct 1–31, 2019 (31/31), while
  120–180 match Oct–Nov 2018 (55/61 due to dry-season zero-rain ties across India).
  Non-JJAS indices are honestly left unconfirmed (`confirmed=False`, `date=""`).
- The full map is committed at `docs/heppi-date-map.csv`. When `date_map` is passed to
  `load_imd_observed` or `load_ncmrwf_forecast`, a real `time` coordinate is attached
  and unconfirmed samples are dropped by default (yielding the 242 confirmed JJAS days).

## Loading it

[`weavr.data.heppi_reference`](../src/weavr/data/heppi_reference.py):

```python
from weavr.data.heppi_reference import load_imd_observed, load_ncmrwf_forecast

obs = load_imd_observed("/path/to/HEPPI/IMD_observed.nc")
# (sample, lat, lon), lat/lon are real coordinates matching weavr.grid.COMMON_LAT/COMMON_LON

fc = load_ncmrwf_forecast("/path/to/HEPPI", variant="uqm")
# (sample, member, lat, lon); variant is "orig" | "uqm" | "emos"
```

Both raise `HeppiGridMismatchError` if the file's `IMD_lat`/`IMD_lon` don't
match weavr's common grid exactly — checked on every load, not assumed from
this doc, in case a different HEPPI release ever ships a different grid.

## Tests

`tests/data/test_heppi_reference.py` builds tiny synthetic netCDF fixtures
with the same variable names/shapes as the real files (not the real 1GB+
data, which isn't available in CI) and checks dimension standardization,
variant-to-filename mapping, and the grid-mismatch guard.
