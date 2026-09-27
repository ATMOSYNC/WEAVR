import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_ifs_ensemble_store import (  # noqa: E402
    _load_manifest,
    _save_manifest,
    fetch_one_timestamp,
    real_baseline_timestamps,
    staging_path,
)


class TestStagingPath:
    def test_is_deterministic_and_filesystem_safe(self, tmp_path):
        ts = np.datetime64("2020-06-01T00:00:00")

        path = staging_path(tmp_path, ts)

        assert path.parent == tmp_path
        assert ":" not in path.name
        assert path.suffix == ".nc"

    def test_distinct_timestamps_get_distinct_paths(self, tmp_path):
        a = staging_path(tmp_path, np.datetime64("2020-06-01T00:00:00"))
        b = staging_path(tmp_path, np.datetime64("2020-06-08T00:00:00"))

        assert a != b

    def test_same_timestamp_is_stable_across_calls(self, tmp_path):
        ts = np.datetime64("2020-06-01T00:00:00")

        assert staging_path(tmp_path, ts) == staging_path(tmp_path, ts)


class TestManifestRoundtrip:
    def test_missing_manifest_returns_empty_dict(self, tmp_path):
        assert _load_manifest(tmp_path / "does_not_exist.json") == {}

    def test_saved_manifest_reloads_identically(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest = {"2020-06-01T00:00:00": {"status": "ok", "elapsed_seconds": 12.3}}

        _save_manifest(path, manifest)

        assert _load_manifest(path) == manifest


class TestRealBaselineTimestamps:
    def test_reads_exact_time_coordinate_from_the_real_group(self, tmp_path):
        store_path = tmp_path / "fake_baseline.zarr"
        times = np.array(["2020-06-01", "2020-06-08"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {"total_precipitation_24hr": (("time",), [1.0, 2.0])},
            coords={"time": times},
        )
        ds.to_zarr(store_path, group="ifs_ens_mean", mode="w")

        result = real_baseline_timestamps(str(store_path))

        np.testing.assert_array_equal(result, times)


class TestFetchOneTimestamp:
    def test_selects_requested_timestamp_and_leads_only(self):
        times = np.array(["2020-06-01", "2020-06-08"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "total_precipitation_24hr": (
                    ("time", "prediction_timedelta"),
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                )
            },
            coords={"time": times, "prediction_timedelta": [24, 48, 72]},
        )

        result = fetch_one_timestamp(ds, times[0], [24, 72])

        assert result.sizes["prediction_timedelta"] == 2
        np.testing.assert_array_equal(
            result["total_precipitation_24hr"].values, [1.0, 3.0]
        )
