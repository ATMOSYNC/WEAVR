import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.colors import PROBABILITY_COLORSCALE, fallback_overlay  # noqa: E402
from dashboard.data_loading import (  # noqa: E402
    DEFAULT_PROBABILITY_GRID_NPZ,
    available_probability_grid_leads,
    load_probability_grid,
)


class TestFallbackOverlay:
    def test_marks_fallback_cells_and_leaves_others_nan(self):
        is_fallback = np.array([[True, False], [False, True]])

        overlay = fallback_overlay(is_fallback)

        assert overlay[0, 0] == 1.0
        assert overlay[1, 1] == 1.0
        assert np.isnan(overlay[0, 1])
        assert np.isnan(overlay[1, 0])

    def test_all_false_gives_an_entirely_nan_overlay(self):
        overlay = fallback_overlay(np.zeros((3, 3), dtype=bool))
        assert np.all(np.isnan(overlay))


class TestProbabilityColorscale:
    def test_spans_zero_to_one_with_imds_real_colour_identities(self):
        positions = [stop[0] for stop in PROBABILITY_COLORSCALE]
        colors = [stop[1] for stop in PROBABILITY_COLORSCALE]
        assert positions[0] == 0.0
        assert positions[-1] == 1.0
        assert positions == sorted(positions)
        # Same 5 real colour identities dashboard.colors.RAIN_BIN_COLORS
        # uses for the blended map -- not a second, invented palette.
        assert colors == ["#ffffff", "#2ecc40", "#ffdc00", "#ff8c00", "#ff0000"]


class TestLoadProbabilityGrid:
    def _write_fixture(self, tmp_path):
        path = tmp_path / "example_probability_grid.npz"
        lat = np.array([10.0, 10.25])
        lon = np.array([70.0, 70.25])
        np.savez(
            path,
            latitude=lat,
            longitude=lon,
            lead_hours=np.array([24, 48]),
            probability_lead_24=np.array([[0.01, 0.0], [0.3, 0.0]]),
            is_fallback_lead_24=np.array([[False, False], [True, False]]),
            sample_time_lead_24=np.array("2020-06-01T00:00:00"),
            probability_lead_48=np.array([[0.0, 0.0], [0.0, 0.0]]),
            is_fallback_lead_48=np.array([[False, False], [False, False]]),
            sample_time_lead_48=np.array("2020-06-02T00:00:00"),
        )
        return path

    def test_loads_probability_and_fallback_for_the_requested_lead(self, tmp_path):
        path = self._write_fixture(tmp_path)

        grid = load_probability_grid(24, path)

        assert grid["lead_hours"] == 24
        assert grid["sample_time"] == "2020-06-01T00:00:00"
        np.testing.assert_array_equal(
            grid["probability"], np.array([[0.01, 0.0], [0.3, 0.0]])
        )
        np.testing.assert_array_equal(
            grid["is_fallback"], np.array([[False, False], [True, False]])
        )

    def test_unknown_lead_hours_raises(self, tmp_path):
        path = self._write_fixture(tmp_path)

        with pytest.raises(ValueError, match="72"):
            load_probability_grid(72, path)

    def test_available_leads_reads_from_the_export(self, tmp_path):
        path = self._write_fixture(tmp_path)
        assert available_probability_grid_leads(path) == [24, 48]


class TestRealCommittedExport:
    def test_real_export_exists_probabilities_in_range_and_fallback_cells_have_zero_probability(
        self,
    ):
        assert DEFAULT_PROBABILITY_GRID_NPZ.exists(), (
            "dashboard/data/example_probability_grid.npz must be committed -- "
            "run scripts/export_dashboard_example_grids.py"
        )
        leads = available_probability_grid_leads()
        assert leads == [24, 48, 72, 96, 120]
        for lead in leads:
            grid = load_probability_grid(lead)
            probability = grid["probability"]
            is_fallback = grid["is_fallback"]

            assert probability.shape == (129, 135)
            assert not np.isnan(probability).any()
            assert np.all(probability >= 0.0) and np.all(probability <= 1.0)

            # A fallback cell's own bin regression was never fit for real
            # (docs/phase4-data-and-combiner-scope.md's finding) --
            # predict_csgd_params's fallback output collapses to a point
            # mass at zero, so its exceedance probability must be ~0, not
            # a confident-looking number.
            if is_fallback.any():
                assert np.allclose(probability[is_fallback], 0.0, atol=1e-6)
