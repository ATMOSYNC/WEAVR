from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tail_repair_qm_emos import (  # noqa: E402
    parse_args,
    probability_sedi,
    quantile_mapped_members,
)

from weavr.quantile_mapping import climatology_excluding_years  # noqa: E402


def _ensemble(values: np.ndarray) -> xr.DataArray:
    n_sample, n_member, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("sample", "member", "latitude", "longitude"),
        coords={
            "sample": pd.date_range("2018-06-01", periods=n_sample),
            "member": np.arange(n_member),
            "latitude": np.arange(n_lat, dtype=float),
            "longitude": np.arange(n_lon, dtype=float),
        },
    )


def test_maps_are_fitted_on_training_days_only() -> None:
    """Changing the test days' forecasts must not change the corrected training days."""
    rng = np.random.default_rng(0)
    values = rng.gamma(0.8, 12.0, size=(60, 4, 3, 3))
    train = np.zeros(60, dtype=bool)
    train[:40] = True
    regions = xr.DataArray(
        np.full((3, 3), "CI"), dims=("latitude", "longitude"),
        coords={"latitude": np.arange(3.0), "longitude": np.arange(3.0)},
    )
    reference = xr.DataArray(
        rng.gamma(0.7, 15.0, size=(40, 3, 3)),
        dims=("sample", "latitude", "longitude"),
        coords={"latitude": np.arange(3.0), "longitude": np.arange(3.0)},
    )

    original = quantile_mapped_members(_ensemble(values), train, reference, regions, 24)
    perturbed = values.copy()
    perturbed[~train] *= 10.0
    changed = quantile_mapped_members(_ensemble(perturbed), train, reference, regions, 24)

    np.testing.assert_allclose(
        original.isel(sample=train).values, changed.isel(sample=train).values
    )


def test_climatology_drops_every_held_out_year() -> None:
    times = pd.to_datetime(["2017-07-01", "2018-07-01", "2018-08-01", "2019-07-01"])
    clim = xr.DataArray(np.arange(4.0), dims="time", coords={"time": times})
    kept = climatology_excluding_years(clim, pd.to_datetime(["2018-06-15"]).values)
    assert list(pd.DatetimeIndex(kept["time"].values).year) == [2017, 2019]


def test_climatology_refuses_to_hold_out_everything() -> None:
    times = pd.to_datetime(["2018-07-01"])
    clim = xr.DataArray([1.0], dims="time", coords={"time": times})
    with pytest.raises(ValueError, match="no reference remains"):
        climatology_excluding_years(clim, times.values)


def test_sedi_scores_probabilities_not_values() -> None:
    """A sharp, correct probability forecast scores high SEDI.

    `weavr.verify.sedi` would threshold the probabilities themselves at
    115.6, find no forecast event anywhere, and return a degenerate score.
    """
    obs = np.zeros((1, 10, 10))
    obs[0, :2, :] = 200.0
    prob = np.where(obs > 0, 0.95, 0.02)
    assert probability_sedi(prob, obs, 115.6) > 0.8
    assert probability_sedi(1.0 - prob, obs, 115.6) < 0.0


def test_test_fraction_default_matches_the_other_runners() -> None:
    assert parse_args([]).test_fraction == 0.2
