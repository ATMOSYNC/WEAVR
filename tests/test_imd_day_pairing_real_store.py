"""Regression test for issue #106: forecasts are paired with the right IMD day.

IMD labels a day by the date its 03 UTC -> 03 UTC window ends on. Pairing
through `weavr.grid.imd_day_for` must therefore put the forecast-vs-observation
day-to-day correlation at its maximum at offset 0. Before the fix every runner
paired a forecast with the day *before* the one it covers, and the maximum sat
at offset +1 (GraphCast, 24 h, 2020: 0.40 at the old pairing, 0.51 one day
later).

This is the check that would have caught it. It reads the real store, so it
skips where `data/` is not present (CI); a pure-synthetic test cannot catch a
wrong reading of a data provider's labelling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.grid import imd_day_for

STORE = Path(__file__).resolve().parent.parent / "data" / "baseline_2020_jjas_daily.zarr"
PRECIP = "total_precipitation_24hr"
OFFSETS = (-1, 0, 1)

pytestmark = pytest.mark.skipif(not STORE.exists(), reason=f"real store not present: {STORE}")


def _lag_correlations(source: str, lead_hours: int) -> dict[int, float]:
    """Land-cell correlation, per-cell mean removed, at each day offset from `imd_day_for`."""
    obs = xr.open_zarr(STORE, group="imd_observed")["rain"].load()
    land = np.isfinite(obs.values).all(axis=0)
    observed = obs.values[:, land]
    obs_days = pd.DatetimeIndex(obs.time.values)

    forecast = xr.open_zarr(STORE, group=source)[PRECIP]
    values = forecast.sel(prediction_timedelta=lead_hours).load().values[:, land] * 1000.0
    paired_day = imd_day_for(pd.DatetimeIndex(forecast.time.values), lead_hours)

    out: dict[int, float] = {}
    for offset in OFFSETS:
        idx = obs_days.get_indexer(paired_day + pd.Timedelta(days=offset))
        ok = idx >= 0
        a = values[ok] - values[ok].mean(axis=0)
        b = observed[idx[ok]] - observed[idx[ok]].mean(axis=0)
        out[offset] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return out


@pytest.mark.parametrize(
    ("source", "lead_hours"),
    [("graphcast", 24), ("graphcast", 72), ("ifs_ens_mean", 24), ("hres", 24)],
)
def test_forecast_observation_correlation_peaks_at_offset_zero(source, lead_hours):
    corr = _lag_correlations(source, lead_hours)

    assert corr[0] == max(corr.values()), (
        f"{source} {lead_hours}h: correlation by day offset {corr}; the pairing in "
        "weavr.grid.imd_day_for is off by a day if the maximum is not at 0"
    )
    # Not merely a tie-break: the correct day must clearly beat the day before it.
    assert corr[0] > corr[-1] + 0.1
