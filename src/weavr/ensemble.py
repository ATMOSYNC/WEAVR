"""Turn a deterministic AI forecast source into a lagged pseudo-ensemble (Phase 2).

GraphCast and Pangu each produce one deterministic forecast per init time --
no spread of their own to feed CRPS/Brier or later BMA/EMOS fitting. A lagged
ensemble substitutes for that spread: several nearby init times (`n_lags`
starts on either side, `lag_spacing_hours` apart) are each re-targeted, by
adjusting their own lead, to forecast the *same valid time* as the nominal
forecast -- the standard cheap way to extract a pseudo-ensemble from a
deterministic model (see docs/phase-plan.md's Phase 2 section).

This module does not pull data from GCS itself -- `build_lagged_ensemble`
takes an already-open `source_ds` shaped like WeatherBench 2's own archives,
`(init_time_dim, lead_dim, latitude, longitude)` with a plain 12-hourly
`init_time_dim` and an integer-hours `lead_dim` -- exactly the shape both
`scripts/build_baseline_store.py` and `scripts/build_lagged_ensemble_store.py`
already open. Nothing here reimplements the GCS pull or the store's schema;
`scripts/build_lagged_ensemble_store.py` is what makes that data actually
present at the init times/leads this module then needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Confirmed live against WeatherBench 2's GraphCast/Pangu archives (see
# docs/baseline-store.md's Phase 2 addition): prediction_timedelta runs
# 6h..240h in 6h steps -- no 0h or negative lead exists in either source. A
# lagged member whose required lead falls outside this range genuinely
# cannot be fetched, not a bug to route around.
MIN_SOURCE_LEAD_HOURS = 6
MAX_SOURCE_LEAD_HOURS = 240


class NoLaggedMembersError(ValueError):
    """Raised when not one single lag member -- including the nominal,
    0h-offset member -- has a valid lead for the requested init_time/lead.

    A 9-member array that is entirely NaN would still look, by shape alone,
    like a complete ensemble to downstream code; this fails loudly instead
    of returning that.
    """


def _lag_offsets(n_lags: int, lag_spacing_hours: int) -> list[int]:
    """+/-n_lags starts, lag_spacing_hours apart, including the nominal (0h) member."""
    return [i * lag_spacing_hours for i in range(-n_lags, n_lags + 1)]


def build_lagged_ensemble(
    source_ds: xr.Dataset,
    variable: str,
    init_time: np.datetime64 | pd.Timestamp | str,
    lead_hours: int,
    n_lags: int = 4,
    lag_spacing_hours: int = 12,
    init_time_dim: str = "time",
    lead_dim: str = "prediction_timedelta",
) -> xr.DataArray:
    """Assemble a lagged ensemble for one nominal (init_time, lead_hours) forecast.

    Returns an `xr.DataArray` with a `member` dimension of size
    `2 * n_lags + 1` -- the shape `weavr.verify.crps`/`brier_score` expect
    via their `member_dim` argument. `member` is indexed by each member's
    offset in hours from the nominal init time (e.g. -48..48 for the
    default 4 lags at 12h spacing); two extra non-dimension coordinates,
    `source_init_time` and `source_lead_hours`, record which real
    (init_time, lead) pair actually produced each member, for inspection.

    **Members with no valid lead are NaN, not dropped, not raised on
    individually.** Two genuinely different reasons a member can be
    unavailable, both checked directly, not assumed to be the same case:
    (1) the *index* doesn't exist -- a member initialized after the
    nominal init time needs a *shorter* lead to reach the same valid time
    (required_lead = lead_hours - offset), and this can fall below the
    source's minimum lead (6h in both GraphCast's and Pangu's archives);
    (2) the index exists but the *data* is NaN there anyway -- e.g.
    WeatherBench 2's `total_precipitation_24hr` is NaN for any lead below
    24h regardless of source, because a 24-hour accumulation isn't defined
    until a full 24h has elapsed, a real constraint the generic 6h
    `MIN_SOURCE_LEAD_HOURS` doesn't know about. Both cases are checked
    explicitly (see docs/baseline-store.md's measured, per-variable
    member-count pattern by lead). NaN-padding to a fixed
    9-member shape, rather than returning a shorter array per sample, was
    checked against `xskillscore.crps_ensemble` directly (used by
    `weavr.verify.crps`) rather than assumed safe: a 9-member array with 3
    NaN members and an unpadded 6-member array of the same real values
    produce the *identical* CRPS score -- `crps_ensemble` already treats
    NaN members as missing per-sample. Padding is used here because it also
    keeps every sample's ensemble the same shape, which is what lets
    multiple samples be concatenated into one array for step 3's scoring
    run.

    Raises `NoLaggedMembersError` only when *every* offset's required lead
    falls outside the source's valid range -- a genuinely unusable request,
    not a normal short-window case.

    Precipitation units are **not** converted here. If `variable` is
    `total_precipitation_24hr`, the returned array is still in meters (the
    WeatherBench 2/ECMWF convention) -- the same unit fact
    `scripts/run_tier0_baseline.py`'s `PRECIP_M_TO_MM` constant documents.
    Conversion is left to the caller, matching how `run_tier0_baseline.py`
    keeps that conversion at the scoring-script level rather than inside a
    shared data-shaping function.
    """
    init_timestamp = pd.Timestamp(init_time)
    offsets = _lag_offsets(n_lags, lag_spacing_hours)

    template = source_ds[variable].isel({init_time_dim: 0, lead_dim: 0}, drop=True)

    member_arrays: list[xr.DataArray] = []
    source_init_times: list[np.datetime64] = []
    source_leads: list[float] = []
    n_valid = 0

    for offset in offsets:
        required_lead = lead_hours - offset
        member_init_time = init_timestamp + pd.Timedelta(hours=offset)
        member_da = None

        if MIN_SOURCE_LEAD_HOURS <= required_lead <= MAX_SOURCE_LEAD_HOURS:
            try:
                member_da = source_ds[variable].sel(
                    {
                        init_time_dim: member_init_time.to_datetime64(),
                        lead_dim: required_lead,
                    }
                )
                member_da = member_da.drop_vars([init_time_dim, lead_dim], errors="ignore")
                if bool(member_da.isnull().all()):
                    # The (init_time, lead) index entry exists -- .sel() did
                    # not raise -- but the source's own data is entirely NaN
                    # there, a genuinely different case from a missing
                    # index. Confirmed live, not assumed: WeatherBench 2's
                    # total_precipitation_24hr is NaN for any lead below
                    # 24h (a 24-hour accumulation isn't defined until a full
                    # 24h has elapsed), even though MIN_SOURCE_LEAD_HOURS=6
                    # (correct for temperature, which has no such floor)
                    # lets the .sel() call through. Treating a fetch that
                    # "succeeded" into an all-NaN array as a valid member
                    # would overcount n_valid_members and could let
                    # NoLaggedMembersError miss a genuinely all-NaN request.
                    member_da = None
            except KeyError:
                member_da = None

        if member_da is not None:
            n_valid += 1
            member_arrays.append(member_da)
            source_init_times.append(member_init_time.to_datetime64())
            source_leads.append(float(required_lead))
        else:
            member_arrays.append(xr.full_like(template, np.nan))
            source_init_times.append(np.datetime64("NaT", "ns"))
            source_leads.append(np.nan)

    if n_valid == 0:
        raise NoLaggedMembersError(
            f"No lag member (of {len(offsets)} offsets, spacing={lag_spacing_hours}h) "
            f"produced usable data for init_time={init_timestamp}, lead_hours={lead_hours}, "
            f"variable={variable!r} -- every required lead either fell outside the source's "
            f"[{MIN_SOURCE_LEAD_HOURS}, {MAX_SOURCE_LEAD_HOURS}]h range or returned all-NaN "
            "data, including the nominal (0h) member."
        )

    ensemble = xr.concat(member_arrays, dim=pd.Index(offsets, name="member"))
    ensemble = ensemble.assign_coords(
        source_init_time=("member", source_init_times),
        source_lead_hours=("member", source_leads),
    )
    ensemble.attrs["valid_time"] = str(init_timestamp + pd.Timedelta(hours=lead_hours))
    ensemble.attrs["n_valid_members"] = n_valid
    return ensemble
