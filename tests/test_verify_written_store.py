"""The canary that a structurally-valid, semantically-empty store is rejected.

`data/ifs_ens_2020_jjas_daily.zarr` was once 3.4 MB of zeros with the right
shape, the right dates and the right dtype. Every structural check passed and
a two-season Tier 2 run spent 2h15m of CPU scoring against it. These tests
pin the value-level guard that would have caught it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_ifs_ensemble_store import (  # noqa: E402
    MIN_NONZERO_FRACTION,
    PRECIP_VARIABLE,
    verify_written_store,
)


def _write_store(path: Path, fill: str) -> None:
    n_time, n_lead, n_lat, n_lon = 6, 2, 4, 5
    if fill == "zeros":
        data = np.zeros((n_time, n_lead, n_lat, n_lon), dtype=np.float32)
    else:
        rng = np.random.default_rng(0)
        data = rng.gamma(shape=1.5, scale=0.3, size=(n_time, n_lead, n_lat, n_lon))
        data[data < 0.2] = 0.0  # a realistic dry fraction, not all-positive
    ds = xr.Dataset(
        {
            PRECIP_VARIABLE: (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                data,
            )
        },
        coords={
            "time": np.arange(n_time),
            "prediction_timedelta": [24, 48],
            "latitude": np.linspace(8.0, 36.0, n_lat),
            "longitude": np.linspace(68.0, 96.0, n_lon),
        },
    )
    ds.to_zarr(path, mode="w")


def test_all_zero_store_is_rejected_and_deleted(tmp_path: Path) -> None:
    store = tmp_path / "stub.zarr"
    _write_store(store, "zeros")

    # The structural checks all pass on this store -- that is the point.
    assert xr.open_zarr(store)[PRECIP_VARIABLE].sizes["time"] == 6
    assert not np.isnan(xr.open_zarr(store)[PRECIP_VARIABLE].values).any()

    with pytest.raises(ValueError, match="non-zero"):
        verify_written_store(store)

    # Deleted, not merely flagged: a store that trips this must not be left
    # where the next reader trusts it.
    assert not store.exists()


def test_real_looking_store_is_accepted(tmp_path: Path) -> None:
    store = tmp_path / "good.zarr"
    _write_store(store, "data")
    frac = verify_written_store(store)
    assert frac >= MIN_NONZERO_FRACTION
    assert store.exists()


def test_acceptance_does_not_require_every_value_nonzero(tmp_path: Path) -> None:
    """A dry field is unusual but not broken; only an all-zero store is.

    This is the test that keeps the guard honest. An earlier threshold of 50%
    would have rejected this store -- and, in production, would have rejected
    any genuinely dry grid while doing nothing extra against the real failure,
    which was 0%.
    """
    store = tmp_path / "mostly_dry.zarr"
    n_time, n_lead, n_lat, n_lon = 6, 2, 4, 5
    data = np.zeros((n_time, n_lead, n_lat, n_lon), dtype=np.float32)
    data[:, :, :3, :3] = 5.0  # 9 of 20 cells wet (45%); the rest genuinely dry
    xr.Dataset(
        {PRECIP_VARIABLE: (("time", "prediction_timedelta", "latitude", "longitude"), data)},
        coords={
            "time": np.arange(n_time),
            "prediction_timedelta": [24, 48],
            "latitude": np.linspace(8.0, 36.0, n_lat),
            "longitude": np.linspace(68.0, 96.0, n_lon),
        },
    ).to_zarr(store, mode="w")

    frac = verify_written_store(store)
    assert 0.40 < frac < 0.50
    assert store.exists()
