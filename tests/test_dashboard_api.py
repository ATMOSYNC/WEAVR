import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# CI's base install (`pip install -e ".[dev]"`) never installs the
# `dashboard-api` group -- same real situation the existing
# tests/test_dashboard_*.py files were already in with the old `dashboard`
# group (streamlit/plotly, removed once Streamlit was retired -- see
# README.md's "Streamlit's status"): nothing under tests/ imports these
# optional groups at module level, so CI never needed them installed.
# Skip this whole file cleanly rather than failing collection when
# fastapi isn't present, instead of adding a new required CI dependency
# for an optional layer.
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dashboard.api import app  # noqa: E402
from dashboard.colors import RAIN_BIN_COLORS  # noqa: E402
from weavr.rain_bins import RAIN_BIN_LABELS  # noqa: E402
from weavr.verify import IMD_RAIN_THRESHOLDS_MM  # noqa: E402

client = TestClient(app)

REAL_LEADS = [24, 48, 72, 96, 120]


class TestMetaLeads:
    def test_returns_the_real_committed_leads_for_both_grids(self):
        response = client.get("/api/meta/leads")
        assert response.status_code == 200
        assert response.json() == {"leads": REAL_LEADS}


class TestColors:
    def test_returns_the_real_imd_colour_identities_unchanged_from_dashboard_colors(self):
        response = client.get("/api/colors")
        assert response.status_code == 200
        body = response.json()
        assert body["rain_bin_colors"] == RAIN_BIN_COLORS
        assert body["rain_bin_labels"] == list(RAIN_BIN_LABELS)
        assert body["imd_rain_thresholds_mm"] == list(IMD_RAIN_THRESHOLDS_MM)
        positions = [stop[0] for stop in body["probability_colorscale"]]
        assert positions[0] == 0.0
        assert positions[-1] == 1.0


class TestWeightMap:
    def test_returns_real_rows_for_a_real_lead(self):
        response = client.get("/api/weight-map", params={"lead": 24})
        assert response.status_code == 200
        rows = response.json()
        assert rows
        assert all(row["lead_hours"] == 24 for row in rows)
        assert {"region", "source", "weight", "is_fallback", "reason", "n_train_points"} <= set(
            rows[0]
        )

    def test_unknown_lead_returns_422_not_a_silent_default(self):
        response = client.get("/api/weight-map", params={"lead": 999})
        assert response.status_code == 422
        assert "999" in response.json()["detail"]


class TestSkillTrends:
    def test_rmse_mm_returns_rows_across_all_four_tiers_methods(self):
        response = client.get("/api/skill-trends", params={"metric": "rmse_mm"})
        assert response.status_code == 200
        rows = response.json()
        methods = {row["method"] for row in rows}
        assert "tier0_ensemble_mean" in methods
        assert "tier1_regional_blend" in methods
        assert "tier3_regime_conditioned" in methods
        assert all(row["metric"] == "rmse_mm" for row in rows)

    def test_unknown_metric_returns_422_not_a_silent_default(self):
        response = client.get("/api/skill-trends", params={"metric": "bogus"})
        assert response.status_code == 422
        assert "bogus" in response.json()["detail"]


class TestBlendedMap:
    def test_returns_the_real_committed_grid_shape_for_every_real_lead(self):
        for lead in REAL_LEADS:
            response = client.get("/api/blended-map", params={"lead": lead})
            assert response.status_code == 200
            body = response.json()
            assert len(body["latitude"]) == 129
            assert len(body["longitude"]) == 135
            assert len(body["bin_index"]) == 129
            assert len(body["bin_index"][0]) == 135
            assert body["lead_hours"] == lead
            # Real bin indices only, matching RAIN_BIN_LABELS's 5 categories.
            flat = [v for row in body["bin_index"] for v in row]
            assert min(flat) >= 0
            assert max(flat) <= len(RAIN_BIN_LABELS) - 1

    def test_unknown_lead_returns_422_not_a_silent_default(self):
        response = client.get("/api/blended-map", params={"lead": 999})
        assert response.status_code == 422


class TestExtremeProbability:
    def test_returns_probabilities_in_range_and_fallback_cells_have_zero_probability(self):
        for lead in REAL_LEADS:
            response = client.get("/api/extreme-probability", params={"lead": lead})
            assert response.status_code == 200
            body = response.json()
            assert len(body["probability"]) == 129
            assert len(body["probability"][0]) == 135
            assert body["lead_hours"] == lead

            probability = np.array(body["probability"])
            is_fallback = np.array(body["is_fallback"])
            assert np.all(probability >= 0.0) and np.all(probability <= 1.0)
            if is_fallback.any():
                assert np.allclose(probability[is_fallback], 0.0, atol=1e-6)

    def test_unknown_lead_returns_422_not_a_silent_default(self):
        response = client.get("/api/extreme-probability", params={"lead": 999})
        assert response.status_code == 422


class TestStreamlitPrototypeRetired:
    def test_streamlit_app_and_views_were_intentionally_removed(self):
        """dashboard/app.py and dashboard/views/*.py (the original
        Streamlit prototype) were removed once this API + dashboard-web/
        frontend reached real, checked parity with them (step 7 of
        workspace/frontend-prompts/, a decision routed to and made by the
        user via AskUserQuestion -- see README.md's "Streamlit's status").
        This asserts the retirement stayed intentional, not an accidental
        partial deletion: the views directory is gone, but
        dashboard.data_loading / dashboard.colors (this API's own real
        dependencies) must still be present and importable."""
        repo_root = Path(__file__).resolve().parents[1]
        assert not (repo_root / "dashboard" / "app.py").exists()
        assert not (repo_root / "dashboard" / "views").exists()

        import dashboard.colors  # noqa: F401
        import dashboard.data_loading  # noqa: F401


if __name__ == "__main__":
    pytest.main([__file__])
