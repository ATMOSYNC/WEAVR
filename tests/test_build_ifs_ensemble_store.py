import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_ifs_ensemble_store import (  # noqa: E402
    _load_manifest,
    _save_manifest,
    fetch_and_stage_one_timestamp,
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

    def test_falls_back_to_alternate_group_if_requested_missing(self, tmp_path):
        store_path = tmp_path / "fake_baseline2.zarr"
        times = np.array(["2020-06-01", "2020-06-02"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {"2m_temperature": (("time",), [290.0, 291.0])},
            coords={"time": times},
        )
        ds.to_zarr(store_path, group="graphcast", mode="w")

        result = real_baseline_timestamps(str(store_path), group="non_existent_group")
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


class TestFetchAndStageOneTimestamp:
    def test_fetches_and_writes_staging_atomically(self, tmp_path):
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        manifest = {}
        lock = threading.Lock()
        per_timestamp_seconds = {}

        times = np.array(["2020-06-01"], dtype="datetime64[ns]")
        sample_ds = xr.Dataset(
            {"total_precipitation_24hr": (("prediction_timedelta",), [5.0])},
            coords={"prediction_timedelta": [24]},
        )

        fake_ds = MagicMock()
        with patch("build_ifs_ensemble_store.fetch_one_timestamp", return_value=sample_ds):
            ok = fetch_and_stage_one_timestamp(
                fake_ds,
                times[0],
                [24],
                staging_dir,
                manifest,
                manifest_path,
                lock,
                per_timestamp_seconds,
                force=False,
                max_retries=2,
            )

        assert ok is True
        ts_key = str(np.datetime_as_string(times[0], unit="s"))
        assert manifest[ts_key]["status"] == "ok"
        staged_file = staging_path(staging_dir, times[0])
        assert staged_file.exists()

        # Check file content
        loaded = xr.open_dataset(staged_file)
        assert "total_precipitation_24hr" in loaded
        assert float(loaded["total_precipitation_24hr"].values[0]) == 5.0

    def test_retries_on_transient_failure_and_succeeds(self, tmp_path):
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        manifest = {}
        lock = threading.Lock()
        per_timestamp_seconds = {}

        times = np.array(["2020-06-01"], dtype="datetime64[ns]")
        sample_ds = xr.Dataset(
            {"total_precipitation_24hr": (("prediction_timedelta",), [5.0])},
            coords={"prediction_timedelta": [24]},
        )

        fake_ds = MagicMock()
        attempts = 0

        def flaky_fetch(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Transient GCS network timeout")
            return sample_ds

        with patch("build_ifs_ensemble_store.fetch_one_timestamp", side_effect=flaky_fetch):
            ok = fetch_and_stage_one_timestamp(
                fake_ds,
                times[0],
                [24],
                staging_dir,
                manifest,
                manifest_path,
                lock,
                per_timestamp_seconds,
                force=False,
                max_retries=3,
            )

        assert ok is True
        assert attempts == 2
        ts_key = str(np.datetime_as_string(times[0], unit="s"))
        assert manifest[ts_key]["status"] == "ok"
        assert manifest[ts_key]["attempts"] == 2

    def test_fails_after_max_retries(self, tmp_path):
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        manifest = {}
        lock = threading.Lock()
        per_timestamp_seconds = {}

        times = np.array(["2020-06-01"], dtype="datetime64[ns]")
        fake_ds = MagicMock()

        with patch(
            "build_ifs_ensemble_store.fetch_one_timestamp",
            side_effect=RuntimeError("Persistent error"),
        ):
            ok = fetch_and_stage_one_timestamp(
                fake_ds,
                times[0],
                [24],
                staging_dir,
                manifest,
                manifest_path,
                lock,
                per_timestamp_seconds,
                force=False,
                max_retries=2,
            )

        assert ok is False
        ts_key = str(np.datetime_as_string(times[0], unit="s"))
        assert manifest[ts_key]["status"] == "failed"
        assert manifest[ts_key]["attempts"] == 2
        assert not staging_path(staging_dir, times[0]).exists()
