"""Phase 5's regime covariates (issue #7) -- day-level labels a weighting
model can condition on, distinct from Phase 3's `regions.py` (a spatial
partition) and Phase 4's `rain_bins.py` (a forecast-value partition).

Per `docs/phase5-regime-covariate-scope.md`'s step 1 findings, three real,
checked covariates are in scope:

- **Monsoon active/break** (`classify_monsoon_active_break` below): real,
  free, computed entirely from data already in this repo -- Rajeevan,
  Gadgil & Bhate (2010), *Journal of Earth System Science* 119, 229-247.
  Their real operational definition: a day is "active" when the
  normalized rainfall anomaly over the **monsoon core zone** (18-28N,
  65-88E) exceeds +1, "break" when it is below -1, each sustained for at
  least 3 consecutive days -- otherwise neutral. Their own validated
  window is July-August only; this module applies the same rule across
  the full JJAS season (a documented extension, not a claim their
  original validation covers June/September too).
- **MJO phase** (`scripts/fetch_omi_mjo_index.py` builds the store this
  module's `load_mjo_phase` reads): Kiladis et al. (2014)'s OMI index,
  chosen over the standard Wheeler & Hendon RMM index because this
  project's own fetch attempt against the RMM index's host (BoM) returned
  HTTP 403, while OMI's NOAA-hosted file was directly, successfully
  fetched.
- **Monsoon-depression presence** (`scripts/build_monsoon_depression_index.py`
  builds the store this module's `load_monsoon_depression_presence`
  reads): a real, ERA5-derived low-pressure-system track catalogue
  (`kieranmrhunt/monsoon-low-atlas` v5.5.1, Zenodo DOI
  10.5281/zenodo.22142640 -- the actually-*published* version; a newer
  v5.6 exists in that project's own GitHub repo but its own release
  manifest marks it `"zenodo_status": "not_published"`, checked directly
  rather than assumed downloadable from its version number alone).

Western-disturbance presence and Neal et al.'s 30-weather-pattern
classification are **not** implemented here -- both real, both ruled out
in step 1 for reasons specific to this project's data (see
`docs/phase5-regime-covariate-scope.md`), not attempted and abandoned.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Rajeevan, Gadgil & Bhate (2010)'s monsoon core zone.
MONSOON_CORE_ZONE_LAT = (18.0, 28.0)
MONSOON_CORE_ZONE_LON = (65.0, 88.0)

# Their own normalized-anomaly threshold and minimum-duration rule.
ACTIVE_BREAK_ANOMALY_THRESHOLD = 1.0
MIN_CONSECUTIVE_DAYS = 3

MONSOON_PHASE_LABELS = ("active", "break")
NEUTRAL_LABEL = ""


class EmptyCoreZoneError(ValueError):
    """Raised when the monsoon core zone selects zero real gridpoints from
    `rain`'s own latitude/longitude coordinates -- a grid mismatch (e.g. a
    source not on weavr's common India grid), not something to silently
    average into an all-NaN time series.
    """


def core_zone_daily_mean(rain: xr.DataArray) -> xr.DataArray:
    """Spatial mean of `rain` (a `(time, latitude, longitude)` array) over
    the monsoon core zone -- the single time series both the real 2020
    season and its multi-year climatology are normalized against.
    """
    subset = rain.sel(
        latitude=slice(*MONSOON_CORE_ZONE_LAT), longitude=slice(*MONSOON_CORE_ZONE_LON)
    )
    if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
        raise EmptyCoreZoneError(
            f"The monsoon core zone (lat {MONSOON_CORE_ZONE_LAT}, lon "
            f"{MONSOON_CORE_ZONE_LON}) selected zero gridpoints from this array's own "
            "latitude/longitude coordinates -- check it's on weavr's common India grid."
        )
    return subset.mean(dim=["latitude", "longitude"], skipna=True)


def _apply_min_run_length(labels: np.ndarray, min_days: int) -> np.ndarray:
    """Keeps a run of identical non-neutral labels only if it is at least
    `min_days` long, per Rajeevan et al.'s own "sustained" requirement --
    a lone active/break day (or a run shorter than the threshold) reverts
    to neutral rather than being reported as a real spell.
    """
    result = labels.copy()
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        if labels[i] != NEUTRAL_LABEL and (j - i) < min_days:
            result[i:j] = NEUTRAL_LABEL
        i = j
    return result


def classify_monsoon_active_break(
    rain: xr.DataArray,
    climatology_rain: xr.DataArray,
) -> xr.DataArray:
    """Labels each day in `rain` (a real `(time, latitude, longitude)`
    daily rainfall array, e.g. `imd_observed.rain`) as `"active"`,
    `"break"`, or `""` (neutral), per Rajeevan, Gadgil & Bhate (2010).

    `climatology_rain` is a multi-year daily rainfall archive (e.g.
    `scripts.build_seeps_climatology.load_climatology`'s own output) used
    only to compute the core zone's climatological mean/std that `rain`'s
    own daily values are normalized against.

    Documented simplification: the climatological mean/std is pooled over
    every JJAS day across every climatology year (not a per-calendar-day
    climatology) -- the 15-year archive this project has gives only ~15
    samples for any single calendar day, too few for a stable per-day
    normal, whereas pooling gives ~2,000+ real daily values. This follows
    the same spirit as the cited paper's own "departure from the
    historical record" (a normalized anomaly against real prior data), not
    an invented threshold.
    """
    daily_series = core_zone_daily_mean(rain)
    climatology_series = core_zone_daily_mean(climatology_rain)

    climatology_mean = float(climatology_series.mean(skipna=True))
    climatology_std = float(climatology_series.std(skipna=True))

    anomaly = (daily_series - climatology_mean) / climatology_std

    raw_labels = xr.where(
        anomaly > ACTIVE_BREAK_ANOMALY_THRESHOLD,
        "active",
        xr.where(anomaly < -ACTIVE_BREAK_ANOMALY_THRESHOLD, "break", NEUTRAL_LABEL),
    )
    filtered = _apply_min_run_length(raw_labels.values.astype(object), MIN_CONSECUTIVE_DAYS)

    return xr.DataArray(
        filtered, coords={"time": rain["time"]}, dims=["time"], name="monsoon_phase"
    )


def load_mjo_phase(store_path: str) -> xr.DataArray:
    """Loads `scripts/fetch_omi_mjo_index.py`'s own per-day MJO phase
    (1-8, standard `atan2(PC2, PC1)` convention) and amplitude, indexed by
    `time`.
    """
    ds = xr.open_zarr(store_path, consolidated=True)
    return ds["phase"]


def load_monsoon_depression_presence(store_path: str) -> xr.DataArray:
    """Loads `scripts/build_monsoon_depression_index.py`'s own per-day
    boolean "a monsoon depression or stronger low-pressure system was
    active over the real India domain" indicator, indexed by `time`.
    """
    ds = xr.open_zarr(store_path, consolidated=True)
    return ds["depression_present"]
