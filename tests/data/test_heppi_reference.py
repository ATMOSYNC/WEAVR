from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.data.heppi_reference import (
    SAMPLE_DATE_CAVEAT,
    HeppiGridMismatchError,
    load_date_map,
    load_imd_observed,
    load_ncmrwf_forecast,
)
from weavr.grid import COMMON_LAT, COMMON_LON

N_SAMPLES = 3
N_MEMBERS = 2


def _write_imd_observed(path: Path, lat=COMMON_LAT, lon=COMMON_LON) -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "IMD_lat": ("lat", lat),
            "IMD_lon": ("lon", lon),
            "IMD_rainfall_observed": (
                ("forecast", "lon", "lat"),
                rng.random((N_SAMPLES, len(lon), len(lat))),
            ),
        }
    )
    ds.to_netcdf(path)


def _write_ncmrwf_forecast(path: Path, var_name: str, lat=COMMON_LAT, lon=COMMON_LON) -> None:
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "IMD_lat": ("lat", lat),
            "IMD_lon": ("lon", lon),
            "ensemble_members": ("ens", np.arange(N_MEMBERS, dtype=float)),
            var_name: (
                ("forecast", "ens", "lon", "lat"),
                rng.random((N_SAMPLES, N_MEMBERS, len(lon), len(lat))),
            ),
        }
    )
    ds.to_netcdf(path)


def _create_sample_date_map() -> pd.DataFrame:
    """Create a sample 3-row date map matching N_SAMPLES = 3."""
    return pd.DataFrame({
        "heppi_index": [0, 1, 2],
        "date": ["2018-06-02", "2018-06-03", ""],
        "best_match_date": ["2018-06-02", "2018-06-03", "2018-10-01"],
        "best_mae": [0.4, 0.7, 0.1],
        "second_best_mae": [3.6, 2.8, 1.2],
        "rank_of_mapped_date": [1, 1, 2],
        "confirmed": [True, True, False],
    })


class TestLoadDateMap:
    def test_loads_valid_date_map(self, tmp_path):
        csv_path = tmp_path / "date_map.csv"
        df = _create_sample_date_map()
        df.to_csv(csv_path, index=False)

        loaded = load_date_map(csv_path)
        assert len(loaded) == 3
        assert list(loaded["confirmed"]) == [True, True, False]
        assert loaded["date"].iloc[0] == "2018-06-02"

    def test_missing_required_columns_raises(self, tmp_path):
        csv_path = tmp_path / "bad_date_map.csv"
        pd.DataFrame({"index": [0, 1]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_date_map(csv_path)


class TestLoadImdObserved:
    def test_loads_and_standardizes_dims(self, tmp_path):
        f = tmp_path / "IMD_observed.nc"
        _write_imd_observed(f)

        da = load_imd_observed(f)

        assert da.dims == ("sample", "lat", "lon")
        assert da.sizes["sample"] == N_SAMPLES
        assert np.allclose(da["lat"].values, COMMON_LAT)
        assert np.allclose(da["lon"].values, COMMON_LON)
        assert "note" in da.attrs
        assert SAMPLE_DATE_CAVEAT in da.attrs["note"]

    def test_grid_mismatch_raises(self, tmp_path):
        f = tmp_path / "IMD_observed.nc"
        _write_imd_observed(f, lat=COMMON_LAT + 1.0)

        with pytest.raises(HeppiGridMismatchError):
            load_imd_observed(f)

    def test_attaches_dates_and_drops_unconfirmed_by_default(self, tmp_path):
        f = tmp_path / "IMD_observed.nc"
        _write_imd_observed(f)
        date_map = _create_sample_date_map()

        da = load_imd_observed(f, date_map=date_map)

        assert "time" in da.coords
        assert da.sizes["sample"] == 2  # dropped 1 unconfirmed
        assert pd.to_datetime(da["time"].values[0]) == pd.Timestamp("2018-06-02")
        assert pd.to_datetime(da["time"].values[1]) == pd.Timestamp("2018-06-03")
        assert SAMPLE_DATE_CAVEAT not in da.attrs["note"]

    def test_preserves_unconfirmed_when_drop_unconfirmed_false(self, tmp_path):
        f = tmp_path / "IMD_observed.nc"
        _write_imd_observed(f)
        date_map = _create_sample_date_map()

        da = load_imd_observed(f, date_map=date_map, drop_unconfirmed=False)

        assert da.sizes["sample"] == 3
        assert "time" in da.coords

    def test_wrong_length_date_map_raises(self, tmp_path):
        f = tmp_path / "IMD_observed.nc"
        _write_imd_observed(f)
        short_date_map = _create_sample_date_map().iloc[:2]

        with pytest.raises(ValueError, match="Date map length .* does not match"):
            load_imd_observed(f, date_map=short_date_map)


class TestLoadNcmrwfForecast:
    def test_loads_and_standardizes_dims(self, tmp_path):
        f = tmp_path / "NCMRWF_orig_forecast.nc"
        _write_ncmrwf_forecast(f, "NCMRWF_frcst")

        da = load_ncmrwf_forecast(tmp_path, variant="orig")

        assert da.dims == ("sample", "member", "lat", "lon")
        assert da.sizes["member"] == N_MEMBERS
        assert "orig" in da.attrs["note"]

    def test_unknown_variant_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown variant"):
            load_ncmrwf_forecast(tmp_path, variant="bogus")

    def test_uqm_and_emos_variants_use_correct_filenames(self, tmp_path):
        _write_ncmrwf_forecast(tmp_path / "NCMRWF_UQM_forecast.nc", "NCMRWF_UQM_frcst")
        _write_ncmrwf_forecast(tmp_path / "NCMRWF_EMOS_forecast.nc", "NCMRWF_EMOS_frcst")

        uqm = load_ncmrwf_forecast(tmp_path, variant="uqm")
        emos = load_ncmrwf_forecast(tmp_path, variant="emos")

        assert uqm.name == "ncmrwf_uqm_forecast"
        assert emos.name == "ncmrwf_emos_forecast"

    def test_attaches_dates_and_updates_note(self, tmp_path):
        f = tmp_path / "NCMRWF_orig_forecast.nc"
        _write_ncmrwf_forecast(f, "NCMRWF_frcst")
        date_map = _create_sample_date_map()

        da = load_ncmrwf_forecast(tmp_path, variant="orig", date_map=date_map)

        assert "time" in da.coords
        assert da.sizes["sample"] == 2
        assert "orig" in da.attrs["note"]
        assert SAMPLE_DATE_CAVEAT not in da.attrs["note"]
