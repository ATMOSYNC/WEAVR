# Data sources — access notes

Status of the three open data sources needed for Phase 0. All three are
confirmed working against live data as of 2026-09-27. None require
authentication or an API key.

## 1. WeatherBench 2

- **Client**: [`weavr.data.weatherbench2`](../src/weavr/data/weatherbench2.py)
- **Access**: public Google Cloud Storage bucket `gs://weatherbench2/`, read
  anonymously via `xarray.open_zarr(..., storage_options={"token": "anon"})`.
  No GCP project or credentials needed.
- **Catalog**: `gs://weatherbench2/datasets/<model>/<year>/<range>.zarr`.
  Confirmed available models include `graphcast`, `pangu`, `hres`, `ifs_ens`,
  `gencast`, `fuxi`, `neuralgcm_deterministic`, `neuralgcm_ens`, `aurora`,
  and ERA5 reanalysis/climatology products. List a model's years/variants
  with `gcsfs.GCSFileSystem(token="anon").ls("gs://weatherbench2/datasets/<model>")`.
- **Tested**: opened the GraphCast 2020 eval-period store
  (`date_range_2019-11-16_2021-02-01_12_hours-240x121_...zarr`), pulled one
  timestep of `2m_temperature` sliced to the India bounding box (lat 6-38,
  lon 68-98 — note WeatherBench 2 longitudes run 0-360, not -180..180).
  Result: correct shape, plausible temperature values (~300K).
- **Gotcha**: the full store is ~1TB (lazy/dask-backed); always slice before
  `.load()`. Coordinate convention differs from ECMWF's (0-360 vs -180-180
  longitude) — don't reuse the same bounding-box slice object across both.

## 2. ECMWF open data (IFS + AIFS)

- **Client**: [`weavr.data.ecmwf_open_data`](../src/weavr/data/ecmwf_open_data.py)
- **Access**: `ecmwf-opendata` Python package. Data is GRIB2, opened via
  `cfgrib` (bundles its own eccodes binary via pip on this platform — no
  system package install needed; confirmed working in a fresh venv).
- **Gotcha (important)**: ECMWF's own direct portal (`source="ecmwf"`)
  enforces a 500-simultaneous-connection cap and returned repeated
  `HTTP 429 Too Many Requests` when tested, even for a single small request.
  ECMWF's own client startup message says the feed is replicated on AWS,
  Azure and GCS specifically for this reason. **Use `source="aws"`** (the
  client's default in this codebase) — confirmed to return immediately with
  no rate-limiting.
- **Tested**: fetched the latest available IFS forecast, `type=fc`, `step=0`,
  `param=tp` (total precipitation), via the AWS mirror. Result: valid
  GRIB2 → xarray dataset, global 0.25° grid (721×1440), sliced correctly to
  the India bounding box (129×121 cells).
- **License**: CC BY 4.0 — attribute ECMWF in any output that uses this data.

## 3. IMD gridded rainfall (0.25°)

- **Client**: [`weavr.data.imd_gridded`](../src/weavr/data/imd_gridded.py)
- **Access**: `imdlib` package, downloads IMD's public binary `.grd` files
  directly from imdpune.gov.in and caches them locally. No credentials.
- **Tested**: downloaded full-year 2023 daily rainfall (`var_type="rain"`),
  confirmed: 365 days, grid spans lat 6.5-38.5 / lon 66.5-100 (0.25°
  spacing), and — importantly — `get_xarray()` already converts IMD's
  internal missing-data fill value to `NaN`. (An earlier draft of this
  client assumed the fill value leaked through as `99.9` and needed manual
  masking; that was wrong — imdlib already handles it. Don't re-add masking
  logic without re-checking this.)
- **Gotcha**: a full year is ~51MB and takes real time to download/parse —
  not instant like the other two sources. Fine for a one-off pull, but the
  bulk historical puller (Phase 0 step 5) should cache aggressively and not
  re-download per run.

## Smoke tests

`tests/data/test_weatherbench2.py`, `tests/data/test_ecmwf_open_data.py`,
`tests/data/test_imd_gridded.py` each exercise the corresponding client
against live data and skip gracefully (rather than fail the build) if the
network call raises — no source needs a secret, so a skip here means a real
outage or an upstream API change, worth investigating but not a merge
blocker.
