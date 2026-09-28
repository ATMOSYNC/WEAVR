import numpy as np
import pytest
import xarray as xr

from weavr.drift import (
    DRIFT_THRESHOLD_STD_MULTIPLIER,
    TRAILING_WINDOW_SAMPLES,
    compute_baseline_stats,
    detect_drift,
    rescore_trailing_window,
    trailing_window_value,
)
from weavr.verify import rmse


class TestComputeBaselineStats:
    def test_mean_and_std_of_a_real_looking_score_series(self):
        scores = [13.9, 13.5, 14.4, 15.0, 14.7]

        stats = compute_baseline_stats(scores, metric_name="rmse_mm")

        assert stats.mean == pytest.approx(np.mean(scores))
        assert stats.std == pytest.approx(np.std(scores, ddof=1))
        assert stats.n_samples == 5
        assert stats.metric_name == "rmse_mm"

    def test_fewer_than_two_scores_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_baseline_stats([5.0], metric_name="rmse_mm")


class TestDetectDrift:
    def _baseline(self):
        return compute_baseline_stats([10.0, 12.0, 11.0, 13.0, 9.0], metric_name="rmse_mm")

    def test_value_within_normal_variance_does_not_trigger(self):
        baseline = self._baseline()  # mean=11, std=1.581
        rolling_value = baseline.mean + 1.0 * baseline.std  # inside the 2*std bar

        result = detect_drift(rolling_value, baseline)

        assert not result.is_drift
        expected_threshold = baseline.mean + DRIFT_THRESHOLD_STD_MULTIPLIER * baseline.std
        assert result.threshold == pytest.approx(expected_threshold)

    def test_known_score_shift_beyond_two_std_triggers(self):
        baseline = self._baseline()  # mean=11, std=1.581
        rolling_value = baseline.mean + 5.0 * baseline.std  # a real, large shift

        result = detect_drift(rolling_value, baseline)

        assert result.is_drift
        assert "threshold" in result.reason
        assert f"{rolling_value:.3f}" in result.reason

    def test_value_exactly_at_threshold_does_not_trigger(self):
        # Strictly greater-than, not >=, matches every other combiner's own
        # convention (a boundary value is not itself an outlier).
        baseline = self._baseline()
        threshold = baseline.mean + DRIFT_THRESHOLD_STD_MULTIPLIER * baseline.std

        result = detect_drift(threshold, baseline)

        assert not result.is_drift

    def test_custom_k_is_respected(self):
        baseline = self._baseline()
        rolling_value = baseline.mean + 1.5 * baseline.std

        loose = detect_drift(rolling_value, baseline, k=1.0)
        strict = detect_drift(rolling_value, baseline, k=2.0)

        assert loose.is_drift
        assert not strict.is_drift


class TestTrailingWindowValue:
    def test_uses_only_the_most_recent_window_entries(self):
        scores = [100.0, 100.0, 100.0, 1.0, 3.0]  # last 2 average to 2.0

        value = trailing_window_value(scores, window=2)

        assert value == pytest.approx(2.0)

    def test_fewer_entries_than_window_uses_all_of_them(self):
        scores = [4.0, 6.0]

        value = trailing_window_value(scores, window=TRAILING_WINDOW_SAMPLES)

        assert value == pytest.approx(5.0)

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="no scores"):
            trailing_window_value([])


class TestRescoreTrailingWindow:
    def _forecast_and_obs(self, n_samples):
        rng = np.random.default_rng(0)
        coords = {
            "sample": np.arange(n_samples),
            "latitude": [10.0, 10.25],
            "longitude": [70.0, 70.25],
        }
        forecast = xr.DataArray(
            rng.normal(loc=5.0, scale=1.0, size=(n_samples, 2, 2)),
            coords=coords,
            dims=["sample", "latitude", "longitude"],
        )
        obs = xr.DataArray(
            rng.normal(loc=5.0, scale=1.0, size=(n_samples, 2, 2)),
            coords=coords,
            dims=["sample", "latitude", "longitude"],
        )
        return forecast, obs

    def test_only_scores_the_trailing_slice_not_the_full_series(self):
        forecast, obs = self._forecast_and_obs(n_samples=10)
        # Corrupt the earliest samples with a huge error that a full-series
        # score would pick up but a trailing window should not.
        forecast = forecast.copy()
        forecast[:5] = forecast[:5] + 1000.0

        windowed = rescore_trailing_window(forecast, obs, metric_fn=rmse, window=3)
        full = float(rmse(forecast, obs))

        assert windowed < full

    def test_fewer_samples_than_window_scores_all_of_them(self):
        forecast, obs = self._forecast_and_obs(n_samples=2)

        windowed = rescore_trailing_window(forecast, obs, metric_fn=rmse, window=14)
        full = float(rmse(forecast, obs))

        assert windowed == pytest.approx(full)

    def test_no_samples_raises(self):
        forecast, obs = self._forecast_and_obs(n_samples=0)

        with pytest.raises(ValueError, match="no samples"):
            rescore_trailing_window(forecast, obs, metric_fn=rmse)
