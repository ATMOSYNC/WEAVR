import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier1_regional_baseline import (  # noqa: E402
    blend_with_region_weights,
    build_region_weight_grid,
    equal_weight_blend,
)

from weavr.weighting import RegionWeightResult  # noqa: E402


def _region_labels() -> xr.DataArray:
    lat = np.array([10.0, 20.0])
    lon = np.array([70.0, 80.0])
    labels = np.array([["R1", "R1"], ["R2", "R2"]])
    return xr.DataArray(
        labels, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"]
    )


class TestBuildRegionWeightGrid:
    def test_each_gridpoint_gets_its_own_region_weight(self):
        region_labels = _region_labels()
        weight_results = {
            "R1": RegionWeightResult(region="R1", weights={"a": 0.7, "b": 0.3}),
            "R2": RegionWeightResult(region="R2", weights={"a": 0.1, "b": 0.9}),
        }

        grids = build_region_weight_grid(weight_results, region_labels, ["a", "b"])

        np.testing.assert_array_equal(grids["a"].values, [[0.7, 0.7], [0.1, 0.1]])
        np.testing.assert_array_equal(grids["b"].values, [[0.3, 0.3], [0.9, 0.9]])

    def test_weight_grids_share_region_labels_coords_and_dims(self):
        region_labels = _region_labels()
        weight_results = {
            "R1": RegionWeightResult(region="R1", weights={"a": 1.0}),
            "R2": RegionWeightResult(region="R2", weights={"a": 1.0}),
        }

        grids = build_region_weight_grid(weight_results, region_labels, ["a"])

        assert grids["a"].dims == region_labels.dims
        np.testing.assert_array_equal(
            grids["a"]["latitude"].values, region_labels["latitude"].values
        )


class TestBlendWithRegionWeights:
    def test_weighted_sum_matches_manual_computation(self):
        samples = np.arange(2)
        region_labels = _region_labels()
        forecast_a = xr.DataArray(
            np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]),
            coords={"sample": samples},
            dims=["sample", "latitude", "longitude"],
        )
        forecast_b = xr.DataArray(
            np.array([[[10.0, 10.0], [10.0, 10.0]], [[20.0, 20.0], [20.0, 20.0]]]),
            coords={"sample": samples},
            dims=["sample", "latitude", "longitude"],
        )
        weight_results = {
            "R1": RegionWeightResult(region="R1", weights={"a": 0.8, "b": 0.2}),
            "R2": RegionWeightResult(region="R2", weights={"a": 0.5, "b": 0.5}),
        }
        weight_grids = build_region_weight_grid(
            weight_results, region_labels, ["a", "b"]
        )

        blend = blend_with_region_weights({"a": forecast_a, "b": forecast_b}, weight_grids)

        # R1 (row 0): 0.8*1 + 0.2*10 = 2.8 (sample 0), 0.8*2 + 0.2*20 = 5.6 (sample 1)
        # R2 (row 1): 0.5*1 + 0.5*10 = 5.5 (sample 0), 0.5*2 + 0.5*20 = 11.0 (sample 1)
        np.testing.assert_allclose(blend.isel(sample=0).values, [[2.8, 2.8], [5.5, 5.5]])
        np.testing.assert_allclose(blend.isel(sample=1).values, [[5.6, 5.6], [11.0, 11.0]])


class TestEqualWeightBlend:
    def test_averages_across_sources_ignoring_nan(self):
        samples = np.arange(1)
        dims = ["sample", "latitude", "longitude"]
        a = xr.DataArray(np.array([[[1.0, np.nan]]]), coords={"sample": samples}, dims=dims)
        b = xr.DataArray(np.array([[[3.0, 5.0]]]), coords={"sample": samples}, dims=dims)

        blend = equal_weight_blend({"a": a, "b": b})

        # Column 0: mean(1, 3) = 2. Column 1: only b has data -> 5 (skipna).
        np.testing.assert_allclose(blend.values, [[[2.0, 5.0]]])
