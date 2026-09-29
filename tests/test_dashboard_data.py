import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.data_loading import (  # noqa: E402
    DEFAULT_TIER0_CSV,
    DEFAULT_TIER1_CSV,
    DEFAULT_TIER2_CSV,
    DEFAULT_TIER3_CSV,
    DEFAULT_WEIGHTS_CSV,
    load_probability_grid,
    load_skill_trends_data,
    load_weight_map_data,
)


class TestLoadWeightMapData:
    def test_melts_weight_columns_into_long_form(self, tmp_path):
        csv_path = tmp_path / "weights.csv"
        pd.DataFrame(
            [
                {
                    "lead_hours": 24,
                    "region": "CI",
                    "is_fallback": False,
                    "reason": "",
                    "n_train_points": 100,
                    "weight_graphcast": 0.3,
                    "weight_hres": 0.1,
                    "weight_ifs_ens_mean": 0.6,
                },
                {
                    "lead_hours": 24,
                    "region": "NE",
                    "is_fallback": True,
                    "reason": "insufficient training points",
                    "n_train_points": 2,
                    "weight_graphcast": 1 / 3,
                    "weight_hres": 1 / 3,
                    "weight_ifs_ens_mean": 1 / 3,
                },
            ]
        ).to_csv(csv_path, index=False)

        tidy = load_weight_map_data(csv_path)

        assert set(tidy.columns) == {
            "lead_hours",
            "region",
            "is_fallback",
            "reason",
            "n_train_points",
            "source",
            "weight",
        }
        assert set(tidy["source"]) == {"graphcast", "hres", "ifs_ens_mean"}
        assert len(tidy) == 6

        ci_graphcast = tidy[(tidy["region"] == "CI") & (tidy["source"] == "graphcast")]
        assert ci_graphcast["weight"].iloc[0] == pytest.approx(0.3)

        ne_rows = tidy[tidy["region"] == "NE"]
        assert ne_rows["is_fallback"].all()
        assert (ne_rows["reason"] == "insufficient training points").all()

    def test_real_committed_csv_loads_and_has_expected_shape(self):
        tidy = load_weight_map_data(DEFAULT_WEIGHTS_CSV)

        assert not tidy.empty
        assert set(tidy["source"]) == {"graphcast", "hres", "ifs_ens_mean"}
        assert tidy["lead_hours"].nunique() >= 1


class TestLoadSkillTrendsData:
    def _write(self, tmp_path, name, rows):
        path = tmp_path / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_tidies_all_four_tiers_without_duplicating_tier3s_repeated_columns(self, tmp_path):
        tier0 = self._write(
            tmp_path, "tier0.csv", [{"lead_hours": 24, "rmse_mm": 11.0}]
        )
        tier1 = self._write(
            tmp_path,
            "tier1.csv",
            [{"lead_hours": 24, "tier1_rmse_mm": 10.0, "equal_rmse_mm": 10.5}],
        )
        tier2 = self._write(
            tmp_path,
            "tier2.csv",
            [
                {
                    "lead_hours": 24,
                    "emos_graphcast_rmse_mm": 9.0,
                    "emos_graphcast_crps_mm": 3.0,
                    "emos_ifs_ens_rmse_mm": 9.5,
                    "emos_ifs_ens_crps_mm": 3.1,
                    "bma_rmse_mm": 9.2,
                    "bma_crps_mm": 3.2,
                }
            ],
        )
        tier3 = self._write(
            tmp_path,
            "tier3.csv",
            [
                {
                    "lead_hours": 24,
                    "regime_rmse_mm": 9.9,
                    "regime_crps_mm": 3.5,
                    # Duplicated reference columns (bit-for-bit identical to
                    # tier2's own values in the real committed CSVs) -- must
                    # NOT be re-plotted as separate tier3 rows.
                    "emos_graphcast_rmse_mm": 9.0,
                    "emos_graphcast_crps_mm": 3.0,
                    "emos_ifs_ens_rmse_mm": 9.5,
                    "emos_ifs_ens_crps_mm": 3.1,
                    "bma_rmse_mm": 9.2,
                    "bma_crps_mm": 3.2,
                }
            ],
        )

        tidy = load_skill_trends_data(tier0, tier1, tier2, tier3)

        assert set(tidy["tier"]) == {"tier0", "tier1", "tier2", "tier3"}

        tier3_rows = tidy[tidy["tier"] == "tier3"]
        assert set(tier3_rows["method"]) == {"tier3_regime_conditioned"}

        tier2_rows = tidy[tidy["tier"] == "tier2"]
        assert set(tier2_rows["method"]) == {
            "tier2_emos_graphcast",
            "tier2_emos_ifs_ens",
            "tier2_bma",
        }

        rmse = tidy[tidy["metric"] == "rmse_mm"]
        assert (
            rmse[(rmse["tier"] == "tier3") & (rmse["method"] == "tier3_regime_conditioned")][
                "value"
            ].iloc[0]
            == pytest.approx(9.9)
        )

        crps = tidy[tidy["metric"] == "crps_mm"]
        assert set(crps["tier"]) == {"tier2", "tier3"}, "tier0/tier1 have no CRPS"

    def test_real_committed_csvs_load_and_confirm_tier3_duplicates_tier2(self):
        tidy = load_skill_trends_data(
            DEFAULT_TIER0_CSV, DEFAULT_TIER1_CSV, DEFAULT_TIER2_CSV, DEFAULT_TIER3_CSV
        )

        assert not tidy.empty
        assert set(tidy["tier"]) == {"tier0", "tier1", "tier2", "tier3"}

        raw_tier2 = pd.read_csv(DEFAULT_TIER2_CSV)
        raw_tier3 = pd.read_csv(DEFAULT_TIER3_CSV)
        shared_cols = [
            c
            for c in raw_tier2.columns
            if c.startswith(("emos_graphcast_", "emos_ifs_ens_", "bma_")) and c in raw_tier3.columns
        ]
        assert shared_cols, "expected tier2/tier3 to share emos/bma reference columns"
        for col in shared_cols:
            assert raw_tier2[col].equals(raw_tier3[col]), (
                f"{col} differs between tier2 and tier3 -- "
                "load_skill_trends_data's no-duplicate assumption no longer holds"
            )


class TestLoadProbabilityGrid:
    def test_committed_grid_loads_default_threshold_204_5(self):
        grid = load_probability_grid(24, threshold=204.5)
        assert grid["lead_hours"] == 24
        assert grid["threshold"] == 204.5
        assert "probability" in grid
        assert "is_fallback" in grid
        assert "method" in grid

    def test_missing_threshold_keys_raises_key_error_not_silent_fallback(self, tmp_path):
        test_npz = tmp_path / "legacy_grid.npz"
        import numpy as np

        np.savez(
            test_npz,
            lead_hours=np.array([24]),
            latitude=np.array([10.0]),
            longitude=np.array([70.0]),
            probability_lead_24=np.array([[0.05]]),
            is_fallback_lead_24=np.array([[False]]),
            sample_time_lead_24=np.array("2020-07-01"),
        )
        # 204.5 succeeds by resolving the legacy/default keys
        grid_204 = load_probability_grid(24, test_npz, threshold=204.5)
        assert grid_204["threshold"] == 204.5
        assert grid_204["probability"][0, 0] == 0.05

        # 115.6 MUST raise KeyError and must NOT silently fall back to the 204.5 array
        with pytest.raises(KeyError, match="Threshold 115.6mm not found"):
            load_probability_grid(24, test_npz, threshold=115.6)

    def test_explicit_threshold_keys_load_successfully_when_present(self, tmp_path):
        test_npz = tmp_path / "multi_threshold_grid.npz"
        import numpy as np

        np.savez(
            test_npz,
            lead_hours=np.array([24]),
            latitude=np.array([10.0]),
            longitude=np.array([70.0]),
            probability_115p6_lead_24=np.array([[0.25]]),
            is_fallback_115p6_lead_24=np.array([[False]]),
            method_115p6_lead_24=np.array([["csgd+gpd_tail"]]),
            probability_204p5_lead_24=np.array([[0.05]]),
            is_fallback_204p5_lead_24=np.array([[False]]),
            method_204p5_lead_24=np.array([["csgd"]]),
            sample_time_lead_24=np.array("2020-07-01"),
        )
        grid_115 = load_probability_grid(24, test_npz, threshold=115.6)
        assert grid_115["threshold"] == 115.6
        assert grid_115["probability"][0, 0] == 0.25
        assert grid_115["method"][0, 0] == "csgd+gpd_tail"

        grid_204 = load_probability_grid(24, test_npz, threshold=204.5)
        assert grid_204["threshold"] == 204.5
        assert grid_204["probability"][0, 0] == 0.05
        assert grid_204["method"][0, 0] == "csgd"
