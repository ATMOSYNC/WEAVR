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

## Known gap: no calendar dates

The 334-sample axis is `Dimensions without coordinates: forecast` in the raw
files — no timestamps. The original release paired the data with a
calendar-date lookup table (`IMD_dates.mat`) that is **not** part of this
download; the README says only "two monsoon seasons." Day-index comments
inside `verif_hitmiss.m` (`6956:6978, 6980:7290`, IMD's own day-numbering
convention, with a noted missing day for 24 June) confirm the seasons are
real and contiguous-with-one-gap, but not which two years. Loaders below
expose this axis as a plain `sample` index and stamp every returned
`DataArray` with a `note` attribute repeating this caveat — do not assume
`sample=0` is a specific date without recovering `IMD_dates.mat` first.

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
