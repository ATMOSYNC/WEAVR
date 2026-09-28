import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_daily_pipeline import (  # noqa: E402
    build_daily_blend,
    compute_rolling_and_drift,
    load_tier1_weights,
)


def _region_labels():
    lat = [10.0, 10.25]
    lon = [70.0, 70.25]
    values = np.array([["R1", "R1"], ["R2", "R2"]])
    return xr.DataArray(
        values, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"]
    )


def _forecast(value):
    lat = [10.0, 10.25]
    lon = [70.0, 70.25]
    return xr.DataArray(
        np.full((2, 2), value),
        coords={"latitude": lat, "longitude": lon},
        dims=["latitude", "longitude"],
    )


class TestBuildDailyBlend:
    def _weights(self):
        return {
            "R1": {"graphcast": 0.5, "hres": 0.3, "ifs_ens_mean": 0.2},
            "R2": {"graphcast": 0.2, "hres": 0.3, "ifs_ens_mean": 0.5},
        }

    def test_all_sources_present_blends_normally(self):
        region_labels = _region_labels()
        sources = {
            "graphcast": _forecast(10.0),
            "hres": _forecast(20.0),
            "ifs_ens_mean": _forecast(30.0),
        }

        blend, meta = build_daily_blend(self._weights(), region_labels, sources)

        assert meta["status"] == "ok"
        assert meta["dropped_sources"] == ""
        assert not meta["any_region_fallback"]
        # R1: 0.5*10 + 0.3*20 + 0.2*30 = 5+6+6 = 17
        assert float(blend.sel(latitude=10.0, longitude=70.0)) == pytest.approx(17.0)
        # R2: 0.2*10 + 0.3*20 + 0.5*30 = 2+6+15 = 23
        assert float(blend.sel(latitude=10.25, longitude=70.0)) == pytest.approx(23.0)

    def test_one_missing_source_runs_end_to_end_with_renormalized_blend(self):
        region_labels = _region_labels()
        sources = {
            "graphcast": _forecast(10.0),
            "hres": None,  # failed to fetch today
            "ifs_ens_mean": _forecast(30.0),
        }

        blend, meta = build_daily_blend(self._weights(), region_labels, sources)

        assert meta["status"] == "ok"
        assert meta["present_sources"] == "graphcast+ifs_ens_mean"
        assert meta["dropped_sources"] == "hres"
        # R1: original 0.5/0.2 renormalized to sum to 1 -> 5/7, 2/7
        expected_r1 = (0.5 / 0.7) * 10.0 + (0.2 / 0.7) * 30.0
        assert float(blend.sel(latitude=10.0, longitude=70.0)) == pytest.approx(expected_r1)

    def test_no_sources_present_returns_none_blend(self):
        region_labels = _region_labels()
        sources = {"graphcast": None, "hres": None, "ifs_ens_mean": None}

        blend, meta = build_daily_blend(self._weights(), region_labels, sources)

        assert blend is None
        assert meta["status"] == "no_sources_present"
        assert meta["any_region_fallback"] is True

    def test_single_surviving_source_flags_region_fallback(self):
        region_labels = _region_labels()
        sources = {"graphcast": _forecast(10.0), "hres": None, "ifs_ens_mean": None}

        blend, meta = build_daily_blend(self._weights(), region_labels, sources)

        assert meta["status"] == "ok"
        assert meta["any_region_fallback"] is True
        assert float(blend.sel(latitude=10.0, longitude=70.0)) == pytest.approx(10.0)


class TestComputeRollingAndDrift:
    def test_insufficient_held_out_history_returns_none(self):
        # window=14 default -> need > 14 entries total (>=16 incl. today)
        # for >=2 held-out baseline samples.
        history = [10.0] * 10

        result = compute_rolling_and_drift(history, today_score=10.5, window=14)

        assert result is None

    def test_stable_history_does_not_trigger_drift(self):
        rng = np.random.default_rng(0)
        history = list(rng.normal(loc=10.0, scale=0.5, size=20))

        result = compute_rolling_and_drift(history, today_score=10.3, window=14)

        assert result is not None
        assert result.is_drift is False

    def test_a_real_score_shift_surfaces_the_drift_flag_from_step_3(self):
        rng = np.random.default_rng(0)
        history = list(rng.normal(loc=10.0, scale=1.0, size=20))

        result = compute_rolling_and_drift(history, today_score=50.0, window=14)

        assert result is not None
        assert result.is_drift is True
        assert "threshold" in result.reason


class TestLoadTier1Weights:
    def test_parses_real_shaped_csv_into_nested_lead_region_source_dict(self, tmp_path):
        csv_path = tmp_path / "weights.csv"
        csv_path.write_text(
            "lead_hours,region,is_fallback,reason,n_train_points,"
            "weight_graphcast,weight_hres,weight_ifs_ens_mean\n"
            "24,CI,False,,100,0.3,0.0,0.7\n"
            "48,CI,False,,100,0.4,0.1,0.5\n"
        )

        weights = load_tier1_weights(str(csv_path))

        assert weights[24]["CI"] == {"graphcast": 0.3, "hres": 0.0, "ifs_ens_mean": 0.7}
        assert weights[48]["CI"]["graphcast"] == pytest.approx(0.4)
