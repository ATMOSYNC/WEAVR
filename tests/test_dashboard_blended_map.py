import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.colors import (  # noqa: E402
    RAIN_BIN_COLORS,
    build_discrete_colorscale,
    rain_bin_index,
)
from dashboard.data_loading import (  # noqa: E402
    DEFAULT_BLEND_GRID_NPZ,
    available_blend_grid_leads,
    load_blend_grid,
)
from weavr.rain_bins import RAIN_BIN_LABELS  # noqa: E402


class TestRainBinIndex:
    def test_bins_values_at_imds_real_thresholds(self):
        values = np.array([0.0, 5.0, 7.5, 50.0, 64.5, 100.0, 115.6, 150.0, 204.5, 300.0])
        index = rain_bin_index(values)

        labels = np.asarray(RAIN_BIN_LABELS)[index]
        assert list(labels) == [
            "dry",
            "dry",
            "light",
            "light",
            "heavy",
            "heavy",
            "very_heavy",
            "very_heavy",
            "extremely_heavy",
            "extremely_heavy",
        ]

    def test_negative_numerical_noise_falls_in_dry_bin(self):
        # GraphCast's own real negative-precipitation numerical-noise
        # artifacts (documented elsewhere in this project, e.g.
        # docs/tier2-hierarchical-baseline-results.md) show up in the real
        # committed example export too -- confirm they don't crash binning
        # or land in a bin other than "dry".
        index = rain_bin_index(np.array([-0.2, -0.01]))
        assert list(np.asarray(RAIN_BIN_LABELS)[index]) == ["dry", "dry"]


class TestBuildDiscreteColorscale:
    def test_produces_hard_steps_not_a_gradient(self):
        scale = build_discrete_colorscale(["#ffffff", "#2ecc40"])
        assert scale == [
            [0.0, "#ffffff"],
            [0.5, "#ffffff"],
            [0.5, "#2ecc40"],
            [1.0, "#2ecc40"],
        ]

    def test_covers_every_real_rain_bin_color(self):
        colors = [RAIN_BIN_COLORS[label] for label in RAIN_BIN_LABELS]
        scale = build_discrete_colorscale(colors)
        assert len(scale) == 2 * len(RAIN_BIN_LABELS)


class TestLoadBlendGrid:
    def _write_fixture(self, tmp_path):
        path = tmp_path / "example_blend_grid.npz"
        lat = np.array([10.0, 10.25])
        lon = np.array([70.0, 70.25])
        np.savez(
            path,
            latitude=lat,
            longitude=lon,
            lead_hours=np.array([24, 48]),
            blend_lead_24=np.array([[1.0, 2.0], [3.0, 4.0]]),
            sample_time_lead_24=np.array("2020-06-01T00:00:00"),
            blend_lead_48=np.array([[5.0, 6.0], [7.0, 8.0]]),
            sample_time_lead_48=np.array("2020-06-02T00:00:00"),
        )
        return path

    def test_loads_the_requested_leads_grid(self, tmp_path):
        path = self._write_fixture(tmp_path)

        grid = load_blend_grid(24, path)

        assert grid["lead_hours"] == 24
        assert grid["sample_time"] == "2020-06-01T00:00:00"
        np.testing.assert_array_equal(grid["values"], np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(grid["latitude"], np.array([10.0, 10.25]))

    def test_unknown_lead_hours_raises(self, tmp_path):
        path = self._write_fixture(tmp_path)

        with pytest.raises(ValueError, match="72"):
            load_blend_grid(72, path)

    def test_available_leads_reads_from_the_export(self, tmp_path):
        path = self._write_fixture(tmp_path)

        assert available_blend_grid_leads(path) == [24, 48]


class TestRealCommittedExport:
    def test_real_export_exists_and_has_every_lead_with_no_nans(self):
        assert DEFAULT_BLEND_GRID_NPZ.exists(), (
            "dashboard/data/example_blend_grid.npz must be committed -- "
            "run scripts/export_dashboard_example_grids.py"
        )
        leads = available_blend_grid_leads()
        assert leads == [24, 48, 72, 96, 120]
        for lead in leads:
            grid = load_blend_grid(lead)
            assert grid["values"].shape == (129, 135)
            assert not np.isnan(grid["values"]).any()
