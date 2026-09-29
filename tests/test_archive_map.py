import numpy as np
import pytest
import xarray as xr

from weavr.archives import archive_for, normalize_coordinates


def test_archive_map_selects_distinct_graphcast_windows_and_shared_nwp_archive():
    old = archive_for("graphcast", 2018)
    new = archive_for("graphcast", 2020)
    assert "/2018/" in old.path
    assert "/2020/" in new.path
    assert old.path != new.path
    assert archive_for("hres", 2018).path == archive_for("hres", 2020).path
    assert archive_for("ifs_ens", 2018).path.endswith("2018-2022-1440x721.zarr")


def test_2018_graphcast_coordinates_are_renamed_before_spatial_selection():
    ds = xr.Dataset(
        {"rain": (("lat", "lon"), np.ones((3, 3)))},
        coords={"lat": [6.0, 6.25, 6.5], "lon": [66.0, 66.25, 66.5]},
    )
    normalized = normalize_coordinates(ds, archive_for("graphcast", 2018))
    assert set(normalized.dims) == {"latitude", "longitude"}
    assert normalized.sel(latitude=slice(6.0, 6.5), longitude=slice(66.0, 66.5)).sizes == {
        "latitude": 3,
        "longitude": 3,
    }


def test_missing_native_coordinate_fails_clearly():
    ds = xr.Dataset(coords={"lat": [6.0]})
    with pytest.raises(ValueError, match="lon"):
        normalize_coordinates(ds, archive_for("graphcast", 2018))


def test_unknown_archive_fails_clearly():
    with pytest.raises(ValueError, match="No WeatherBench 2 archive"):
        archive_for("graphcast", 2019)
