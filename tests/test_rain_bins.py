import numpy as np
import pytest
import xarray as xr

from weavr.rain_bins import (
    MISSING_LABEL,
    RAIN_BIN_LABELS,
    classify_rain_bin,
)
from weavr.verify import IMD_RAIN_THRESHOLDS_MM


class TestClassifyRainBinBoundaries:
    @pytest.mark.parametrize(
        "value,expected_label",
        [
            (0.0, "dry"),
            (7.499, "dry"),
            (7.5, "light"),  # left-closed: exactly the threshold is the higher bin
            (64.499, "light"),
            (64.5, "heavy"),
            (115.599, "heavy"),
            (115.6, "very_heavy"),
            (204.499, "very_heavy"),
            (204.5, "extremely_heavy"),
            (1000.0, "extremely_heavy"),
        ],
    )
    def test_value_lands_in_expected_bin(self, value, expected_label):
        da = xr.DataArray([value], dims=["sample"])

        result = classify_rain_bin(da)

        assert result.values[0] == expected_label

    def test_left_closed_matches_contingency_scores_event_convention(self):
        # verify.contingency_scores treats an event as `value >= threshold` --
        # classify_rain_bin's bin boundaries must agree with that convention,
        # not silently use a different (e.g. strict >) comparison.
        threshold = IMD_RAIN_THRESHOLDS_MM[0]
        da = xr.DataArray([threshold], dims=["sample"])

        result = classify_rain_bin(da)

        assert result.values[0] != RAIN_BIN_LABELS[0]


class TestBinOrderingIsMonotonic:
    def test_labels_are_in_increasing_intensity_order(self):
        values = [0.0, 10.0, 70.0, 120.0, 210.0]
        da = xr.DataArray(values, dims=["sample"])

        result = classify_rain_bin(da)

        label_positions = [RAIN_BIN_LABELS.index(label) for label in result.values]
        assert label_positions == sorted(label_positions)


class TestMissingDataHandling:
    def test_nan_is_not_silently_placed_in_the_last_bin(self):
        # numpy.digitize on its own places NaN in the final bin index --
        # classify_rain_bin must override that, not inherit it.
        da = xr.DataArray([np.nan], dims=["sample"])

        result = classify_rain_bin(da)

        assert result.values[0] == MISSING_LABEL
        assert result.values[0] != RAIN_BIN_LABELS[-1]

    def test_nan_mixed_with_real_values_only_masks_the_nan_cells(self):
        da = xr.DataArray([1.0, np.nan, 300.0], dims=["sample"])

        result = classify_rain_bin(da)

        assert list(result.values) == ["dry", MISSING_LABEL, "extremely_heavy"]


class TestShapePreservation:
    def test_output_shares_dims_and_coords_with_input(self):
        lat = np.array([10.0, 20.0])
        lon = np.array([70.0, 80.0])
        samples = np.arange(3)
        da = xr.DataArray(
            np.zeros((3, 2, 2)),
            coords={"sample": samples, "latitude": lat, "longitude": lon},
            dims=["sample", "latitude", "longitude"],
        )

        result = classify_rain_bin(da)

        assert result.dims == da.dims
        assert result.shape == da.shape
        np.testing.assert_array_equal(result["sample"].values, da["sample"].values)
        np.testing.assert_array_equal(result["latitude"].values, da["latitude"].values)
        np.testing.assert_array_equal(result["longitude"].values, da["longitude"].values)

    def test_no_dimension_is_added_or_dropped(self):
        da = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]), dims=["x", "y"])

        result = classify_rain_bin(da)

        assert result.ndim == da.ndim


class TestThresholdValidation:
    def test_wrong_number_of_thresholds_raises(self):
        da = xr.DataArray([1.0], dims=["sample"])

        with pytest.raises(ValueError, match="thresholds must have"):
            classify_rain_bin(da, thresholds=(1.0, 2.0))
