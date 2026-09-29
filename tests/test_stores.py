import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.stores import open_multi_season


def _write_season(path, year, latitude=(6.5, 6.75), time_dim="time"):
    ds = xr.Dataset(
        {"rain": ((time_dim, "latitude", "longitude"), np.ones((2, 2, 2)))},
        coords={
            time_dim: pd.date_range(f"{year}-06-01", periods=2),
            "latitude": list(latitude),
            "longitude": [66.5, 66.75],
        },
    )
    ds.to_zarr(path, group="graphcast", mode="w")


@pytest.mark.parametrize("time_dim", ["time", "nominal_time"])
def test_open_multi_season_combines_matching_grids(tmp_path, time_dim):
    first, second = tmp_path / "2018.zarr", tmp_path / "2020.zarr"
    _write_season(first, 2018, time_dim=time_dim)
    _write_season(second, 2020, time_dim=time_dim)

    combined = open_multi_season([first, second], "graphcast")
    assert combined.sizes[time_dim] == 4
    assert pd.DatetimeIndex(combined[time_dim].values).year.tolist() == [2018, 2018, 2020, 2020]
    combined.close()


def test_open_multi_season_rejects_grid_mismatch(tmp_path):
    first, second = tmp_path / "2018.zarr", tmp_path / "2020.zarr"
    _write_season(first, 2018)
    _write_season(second, 2020, latitude=(6.5, 7.0))

    with pytest.raises(ValueError, match="latitude grid differs"):
        open_multi_season([first, second], "graphcast")


def test_open_multi_season_rejects_overlapping_seasons(tmp_path):
    first, second = tmp_path / "first.zarr", tmp_path / "second.zarr"
    _write_season(first, 2018)
    _write_season(second, 2018)

    with pytest.raises(ValueError, match="overlap"):
        open_multi_season([first, second], "graphcast")


def test_resolve_store_paths_precedence(tmp_path):
    from weavr.stores import resolve_store_paths

    # 1. Explicit legacy path
    assert resolve_store_paths(legacy_single_path="legacy.zarr") == ["legacy.zarr"]

    # 2. Explicit specified paths
    assert resolve_store_paths(specified_paths=["a.zarr", "b.zarr"]) == ["a.zarr", "b.zarr"]

    # 3. Existing defaults
    f1 = tmp_path / "f1.zarr"
    f1.mkdir()
    res = resolve_store_paths(default_multi_paths=[f1, tmp_path / "nonexistent.zarr"])
    assert res == [f1]

    # 4. Fallback legacy
    leg = tmp_path / "leg.zarr"
    leg.mkdir()
    res2 = resolve_store_paths(
        default_multi_paths=[tmp_path / "missing1.zarr"],
        legacy_fallback_path=leg,
    )
    assert res2 == [leg]

