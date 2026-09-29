import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_scorecard import (  # noqa: E402
    CLAIM_SPECS,
    _bootstrap_categorical_difference,
    _pooled_csi,
    _pooled_sedi,
    _verdict_for_claim,
    build_scorecard,
    compare_methods,
    measure_block_length,
    write_verdicts,
)

from weavr import verify as V  # noqa: E402

PREREGISTRATION = Path(__file__).resolve().parents[1] / "docs" / "preregistration.md"


def _per_day(method: str, lead: int, n_days: int, mse: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-07-01", periods=n_days, freq="D"),
            "fold": "test",
            "n_cells": 100.0,
            "mse_mm2": rng.normal(mse, mse * 0.05, size=n_days),
            "mae_mm": rng.normal(np.sqrt(mse), 0.5, size=n_days),
            "crps_mm": rng.normal(np.sqrt(mse) / 2, 0.3, size=n_days),
            "method": method,
            "lead_hours": lead,
        }
    )
    for threshold in V.IMD_RAIN_THRESHOLDS_MM:
        frame[f"hits_{threshold}"] = 10.0
        frame[f"misses_{threshold}"] = 5.0
        frame[f"false_alarms_{threshold}"] = 5.0
        frame[f"correct_negatives_{threshold}"] = 80.0
    return frame


class TestPooledCategoricalScores:
    def test_pooled_csi_matches_the_definition(self):
        assert _pooled_csi(10, 5, 5) == pytest.approx(10 / 20)

    def test_pooled_csi_is_nan_when_nothing_happened(self):
        assert np.isnan(_pooled_csi(0, 0, 0))

    def test_pooled_sedi_matches_weavr_verify_sedi(self):
        # The scalar path must agree with the DataArray implementation, or
        # the scorecard and the results docs would disagree.
        import xarray as xr

        rng = np.random.default_rng(0)
        obs_values = rng.uniform(0, 120, size=(60, 8, 8))
        forecast_values = obs_values + rng.normal(0, 25, size=obs_values.shape)
        obs = xr.DataArray(obs_values, dims=["sample", "latitude", "longitude"])
        forecast = xr.DataArray(forecast_values, dims=obs.dims)
        threshold = 64.5

        counts = V.contingency_counts(forecast, obs, thresholds=(threshold,))[threshold]
        scalar = _pooled_sedi(
            float(counts["hits"]),
            float(counts["misses"]),
            float(counts["false_alarms"]),
            float(counts["correct_negatives"]),
        )

        assert scalar == pytest.approx(float(V.sedi(forecast, obs, threshold)))

    def test_pooled_sedi_is_nan_in_the_degenerate_cases(self):
        assert np.isnan(_pooled_sedi(0, 10, 0, 90))  # H = 0 and F = 0
        assert np.isnan(_pooled_sedi(10, 0, 5, 85))  # H = 1

    def test_categorical_difference_is_oriented_so_negative_means_a_is_better(self):
        # A has a higher CSI than B. CSI is positively oriented, so the
        # recorded difference must be NEGATIVE to match every loss metric.
        n = 20
        better = tuple(
            np.full(n, x, dtype=float) for x in (20.0, 2.0, 2.0, 76.0)
        )  # CSI = 20/24
        worse = tuple(np.full(n, x, dtype=float) for x in (5.0, 10.0, 10.0, 75.0))

        result = _bootstrap_categorical_difference(
            better, worse, "csi", block_days=7, n_resamples=200, seed=0
        )

        assert result.estimate < 0
        assert _pooled_csi(20, 2, 2) > _pooled_csi(5, 10, 10)

    def test_categorical_difference_of_identical_counts_is_zero(self):
        counts = tuple(np.full(20, x, dtype=float) for x in (10.0, 5.0, 5.0, 80.0))

        result = _bootstrap_categorical_difference(
            counts, tuple(c.copy() for c in counts), "csi", 7, 100, 0
        )

        assert result.estimate == pytest.approx(0.0)
        assert not result.significant


class TestCompareMethods:
    def test_produces_rows_for_every_available_metric(self):
        per_day = pd.concat(
            [_per_day("a", 24, 30, 100.0, seed=1), _per_day("b", 24, 30, 200.0, seed=2)],
            ignore_index=True,
        )

        rows = compare_methods(per_day, "a", "b", 24, 7, 200, 0)

        metrics = {row["metric"] for row in rows}
        assert {"rmse_mm", "mae_mm", "crps_mm", "csi", "sedi"} <= metrics

    def test_a_clearly_better_method_is_significant_on_rmse(self):
        per_day = pd.concat(
            [_per_day("good", 24, 40, 50.0, seed=3), _per_day("bad", 24, 40, 400.0, seed=4)],
            ignore_index=True,
        )

        rows = compare_methods(per_day, "good", "bad", 24, 7, 300, 0)
        rmse = next(r for r in rows if r["metric"] == "rmse_mm")

        assert rmse["diff"] < 0
        assert rmse["significant"]
        assert rmse["estimate_a"] < rmse["estimate_b"]

    def test_rmse_estimate_is_sqrt_of_mean_mse(self):
        frame_a = _per_day("a", 24, 30, 100.0, seed=5)
        per_day = pd.concat([frame_a, _per_day("b", 24, 30, 120.0, seed=6)], ignore_index=True)

        rows = compare_methods(per_day, "a", "b", 24, 7, 100, 0)
        rmse = next(r for r in rows if r["metric"] == "rmse_mm")

        assert rmse["estimate_a"] == pytest.approx(np.sqrt(frame_a["mse_mm2"].mean()))

    def test_methods_are_paired_on_shared_dates_only(self):
        frame_a = _per_day("a", 24, 30, 100.0, seed=7)
        frame_b = _per_day("b", 24, 30, 120.0, seed=8)
        frame_b = frame_b.iloc[10:]  # b is missing the first 10 days
        per_day = pd.concat([frame_a, frame_b], ignore_index=True)

        rows = compare_methods(per_day, "a", "b", 24, 7, 100, 0)

        assert all(row["n_days"] == 20 for row in rows if row["metric"] == "rmse_mm")

    def test_returns_nothing_when_a_method_is_absent(self):
        per_day = _per_day("a", 24, 30, 100.0)
        assert compare_methods(per_day, "a", "missing", 24, 7, 100, 0) == []

    def test_returns_nothing_with_fewer_than_two_shared_days(self):
        per_day = pd.concat(
            [_per_day("a", 24, 1, 100.0), _per_day("b", 24, 1, 120.0)], ignore_index=True
        )
        assert compare_methods(per_day, "a", "b", 24, 7, 100, 0) == []


class TestBuildScorecard:
    def test_compares_every_method_against_each_reference(self):
        per_day = pd.concat(
            [
                _per_day("tier0", 24, 20, 150.0, seed=9),
                _per_day("climatology", 24, 20, 300.0, seed=10),
                _per_day("best_single_member_on_train", 24, 20, 140.0, seed=11),
                _per_day("tier2_bma", 24, 20, 120.0, seed=12),
            ],
            ignore_index=True,
        )

        scorecard = build_scorecard(per_day, 7, 100, 0)

        pairs = set(zip(scorecard["method_a"], scorecard["method_b"], strict=True))
        assert ("tier2_bma", "tier0") in pairs
        assert ("tier2_bma", "climatology") in pairs
        assert ("tier2_bma", "best_single_member_on_train") in pairs
        # A reference is never compared against itself.
        assert not any(a == b for a, b in pairs)


class TestMeasureBlockLength:
    def test_reports_autocorrelation_per_method_and_lead(self):
        per_day = pd.concat(
            [_per_day("a", 24, 30, 100.0), _per_day("a", 48, 30, 110.0)], ignore_index=True
        )

        frame = measure_block_length(per_day)

        assert set(frame["lead_hours"]) == {24, 48}
        assert "lag1_autocorrelation_mse" in frame.columns
        assert (frame["n_days"] == 30).all()


class TestVerdicts:
    def _scorecard_row(self, lead, diff, significant, method_a="m", method_b="ref"):
        return {
            "metric": "rmse_mm",
            "threshold": "",
            "lead_hours": lead,
            "method_a": method_a,
            "method_b": method_b,
            "estimate_a": 10.0,
            "estimate_b": 11.0,
            "diff": diff,
            "ci_lo": -1.0,
            "ci_hi": -0.5,
            "dm_p": 0.01,
            "significant": significant,
            "n_days": 100,
            "block_days": 7,
        }

    def _spec(self):
        return {
            "claim": "HX",
            "summary": "test claim",
            "method_a": "m",
            "method_b": "ref",
            "metric": "rmse_mm",
            "threshold": "",
            "produced_by": "07",
        }

    def test_passes_when_improved_and_significant_at_three_leads(self):
        scorecard = pd.DataFrame(
            [self._scorecard_row(lead, -1.0, True) for lead in (24, 48, 72)]
            + [self._scorecard_row(lead, 0.5, False) for lead in (96, 120)]
        )

        verdict = _verdict_for_claim(self._spec(), scorecard, {"m", "ref"})

        assert verdict["status"] == "PASS"
        assert verdict["n_leads_significant"] == 3

    def test_fails_when_the_direction_itself_fails(self):
        # Improvement at only 2 leads: a real negative result.
        scorecard = pd.DataFrame(
            [self._scorecard_row(lead, -1.0, True) for lead in (24, 48)]
            + [self._scorecard_row(lead, 1.0, False) for lead in (72, 96, 120)]
        )

        verdict = _verdict_for_claim(self._spec(), scorecard, {"m", "ref"})

        assert verdict["status"] == "FAIL"

    def test_insufficient_data_when_the_direction_holds_but_cis_do_not_exclude_zero(self):
        # This is the expected outcome on today's weekly store, and it must
        # NOT be reported as FAIL -- unproven is not disproven.
        scorecard = pd.DataFrame(
            [self._scorecard_row(lead, -1.0, False) for lead in (24, 48, 72, 96, 120)]
        )

        verdict = _verdict_for_claim(self._spec(), scorecard, {"m", "ref"})

        assert verdict["status"] == "INSUFFICIENT_DATA"
        assert "not disproven" in verdict["reason"]

    def test_not_yet_tested_names_the_missing_method_and_step(self):
        verdict = _verdict_for_claim(self._spec(), pd.DataFrame(), set())

        assert verdict["status"] == "NOT_YET_TESTED"
        assert "step 07" in verdict["reason"]
        assert "m" in verdict["reason"]

    def test_report_only_claims_are_never_pass_or_fail(self):
        spec = {**self._spec(), "report_only": True, "claim": "H9"}

        verdict = _verdict_for_claim(spec, pd.DataFrame(), {"m", "ref"})

        assert verdict["status"] == "NOT_YET_TESTED"
        assert "No pass/fail" in verdict["reason"]

    def test_every_claim_in_the_specs_exists_in_the_preregistration_doc(self, tmp_path):
        # The doc is the source of truth; a claim implemented here but absent
        # there would be an unregistered claim masquerading as a registered
        # one. write_verdicts enforces this.
        per_day = _per_day("tier0", 24, 5, 100.0)
        write_verdicts(pd.DataFrame(), per_day, tmp_path, str(PREREGISTRATION))
        written = pd.read_csv(tmp_path / "preregistration_verdicts.csv")

        assert set(written["claim"]) == {spec["claim"] for spec in CLAIM_SPECS}

    def test_rejects_a_claim_missing_from_the_doc(self, tmp_path):
        fake = tmp_path / "preregistration.md"
        fake.write_text("# Nothing registered here\n")
        per_day = _per_day("tier0", 24, 5, 100.0)

        with pytest.raises(ValueError, match="no section"):
            write_verdicts(pd.DataFrame(), per_day, tmp_path, str(fake))

    def test_rejects_a_missing_preregistration_file(self, tmp_path):
        per_day = _per_day("tier0", 24, 5, 100.0)
        with pytest.raises(FileNotFoundError, match="meaningless without"):
            write_verdicts(pd.DataFrame(), per_day, tmp_path, str(tmp_path / "absent.md"))
