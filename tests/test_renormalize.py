import numpy as np
import pytest
import xarray as xr

from weavr.renormalize import (
    MIN_SOURCES_FOR_BLEND,
    missing_for_sample,
    present_sources_for_sample,
    renormalize_weights,
)


class TestRenormalizeWeights:
    def test_no_source_missing_returns_original_weights_unchanged(self):
        weights = {"a": 0.2, "b": 0.3, "c": 0.5}

        result = renormalize_weights(weights, present_sources=["a", "b", "c"])

        assert result.weights == pytest.approx(weights)
        assert result.present_sources == ("a", "b", "c")
        assert result.dropped_sources == ()
        assert not result.is_fallback

    def test_one_source_dropped_renormalizes_remaining_proportionally(self):
        # a, b originally 0.2/0.3 out of a 0.5 total among {a, b} -> rescaled
        # to sum to 1: 0.4, 0.6.
        weights = {"a": 0.2, "b": 0.3, "c": 0.5}

        result = renormalize_weights(weights, present_sources=["a", "b"])

        assert result.weights == pytest.approx({"a": 0.4, "b": 0.6})
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert result.present_sources == ("a", "b")
        assert result.dropped_sources == ("c",)
        assert not result.is_fallback
        assert result.reason is None

    def test_known_three_source_weight_set_with_one_dropped(self):
        weights = {"graphcast": 0.5, "hres": 0.3, "ifs_ens": 0.2}

        result = renormalize_weights(weights, present_sources=["graphcast", "ifs_ens"])

        # 0.5 / (0.5 + 0.2) = 0.7142..., 0.2 / 0.7 = 0.2857...
        assert result.weights == pytest.approx({"graphcast": 5 / 7, "ifs_ens": 2 / 7})
        assert result.dropped_sources == ("hres",)

    def test_only_one_source_present_passes_through_with_weight_one_and_flags_fallback(self):
        weights = {"a": 0.2, "b": 0.3, "c": 0.5}

        result = renormalize_weights(weights, present_sources=["b"])

        assert result.weights == {"b": 1.0}
        assert result.present_sources == ("b",)
        assert set(result.dropped_sources) == {"a", "c"}
        assert result.is_fallback
        assert "only 1 of 3" in result.reason

    def test_too_few_sources_threshold_is_min_sources_for_blend(self):
        assert MIN_SOURCES_FOR_BLEND == 2

    def test_present_sources_original_weights_sum_to_zero_falls_back_to_equal_split(self):
        weights = {"a": 0.0, "b": 0.0, "c": 1.0}

        result = renormalize_weights(weights, present_sources=["a", "b"])

        assert result.weights == pytest.approx({"a": 0.5, "b": 0.5})
        assert result.is_fallback
        assert "summed to zero" in result.reason

    def test_no_source_present_raises(self):
        weights = {"a": 0.5, "b": 0.5}

        with pytest.raises(ValueError, match="no source"):
            renormalize_weights(weights, present_sources=["c"])

    def test_present_sources_not_in_weights_are_ignored(self):
        weights = {"a": 1.0}

        result = renormalize_weights(weights, present_sources=["a", "z"])

        assert result.weights == {"a": 1.0}
        assert result.present_sources == ("a",)


class TestMissingForSample:
    def test_all_nan_sample_is_missing(self):
        forecast = xr.DataArray(
            np.array(
                [
                    np.full((2, 2), np.nan),
                    np.ones((2, 2)),
                ]
            ),
            coords={
                "sample": [0, 1],
                "latitude": [10.0, 10.25],
                "longitude": [70.0, 70.25],
            },
            dims=["sample", "latitude", "longitude"],
        )

        result = missing_for_sample(forecast)

        np.testing.assert_array_equal(result.values, [True, False])

    def test_partially_nan_sample_is_not_missing(self):
        forecast = xr.DataArray(
            np.array([[np.nan, 1.0], [2.0, 3.0]]),
            coords={"latitude": [10.0, 10.25], "longitude": [70.0, 70.25]},
            dims=["latitude", "longitude"],
        )

        assert bool(missing_for_sample(forecast)) is False


class TestPresentSourcesForSample:
    def _forecast(self, day0_value, day1_value):
        return xr.DataArray(
            np.array([np.full((2, 2), day0_value), np.full((2, 2), day1_value)]),
            coords={
                "sample": [0, 1],
                "latitude": [10.0, 10.25],
                "longitude": [70.0, 70.25],
            },
            dims=["sample", "latitude", "longitude"],
        )

    def test_absent_key_and_nan_filled_day_are_both_excluded(self):
        forecasts = {
            "graphcast": self._forecast(1.0, np.nan),  # missing on day 1 only
            "hres": self._forecast(2.0, 2.0),  # always present
            "ifs_ens": None,  # never pulled at all
        }

        assert present_sources_for_sample(forecasts, sample=0) == ("graphcast", "hres")
        assert present_sources_for_sample(forecasts, sample=1) == ("hres",)
