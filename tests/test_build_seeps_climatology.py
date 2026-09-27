import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_seeps_climatology import (  # noqa: E402
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    _clear_encoding,
    load_climatology,
)


class TestClearEncoding:
    def test_strips_encoding_from_all_variables_and_coords(self):
        da = xr.DataArray(np.zeros(3), coords={"time": [1, 2, 3]}, dims=["time"])
        da.encoding = {"chunks": (1,), "compressors": "fake"}
        da["time"].encoding = {"chunks": (1,)}
        ds = xr.Dataset({"rain": da})

        out = _clear_encoding(ds)

        assert out["rain"].encoding == {}
        assert out["time"].encoding == {}


class TestDefaultYearWindow:
    def test_covers_fifteen_years_ending_2020(self):
        assert DEFAULT_END_YEAR == 2020
        assert DEFAULT_END_YEAR - DEFAULT_START_YEAR + 1 == 15


def _write_year_group(store_path: Path, year: int, mode: Literal["w", "a"]) -> None:
    times = pd.date_range(f"{year}-06-01", f"{year}-06-03")
    da = xr.DataArray(
        np.full((len(times), 2, 2), float(year)),
        coords={"time": times, "latitude": [6.5, 6.75], "longitude": [66.5, 66.75]},
        dims=["time", "latitude", "longitude"],
    )
    xr.Dataset({"rain": da}).to_zarr(store_path, group=f"y{year}", mode=mode)


class TestLoadClimatology:
    def test_concatenates_and_sorts_out_of_order_year_groups(self, tmp_path):
        store_path = tmp_path / "climatology.zarr"
        # Write 2010 first, then 2008 -- simulating a retried-earlier-year
        # landing after a later one, the real ordering hazard this guards
        # against (see module docstring: 2009/2018 retried after 2019/2020).
        _write_year_group(store_path, 2010, mode="w")
        _write_year_group(store_path, 2008, mode="a")

        combined = load_climatology(store_path)

        times = combined["time"].values
        assert list(times) == sorted(times)
        assert pd.Timestamp(times[0]).year == 2008
        assert pd.Timestamp(times[-1]).year == 2010
