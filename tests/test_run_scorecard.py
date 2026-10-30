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
    _verdict_from_runner_table,
    _verdicts_with_legs,
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

    def test_pod_and_far_keep_the_loss_orientation(self):
        # A detects more (POD 20/22 vs 5/15) and raises fewer false alarms
        # (FAR 2/22 vs 10/15): both differences must read "A is better".
        better = tuple(np.full(20, x, dtype=float) for x in (20.0, 2.0, 2.0, 76.0))
        worse = tuple(np.full(20, x, dtype=float) for x in (5.0, 10.0, 10.0, 75.0))

        pod = _bootstrap_categorical_difference(better, worse, "pod", 7, 200, 0)
        far = _bootstrap_categorical_difference(better, worse, "far", 7, 200, 0)

        assert pod.estimate == pytest.approx(-(20 / 22 - 5 / 15))
        assert far.estimate == pytest.approx(2 / 22 - 10 / 15)
        assert pod.estimate < 0 and far.estimate < 0

    def test_unknown_score_is_rejected_not_scored_as_sedi(self):
        counts = tuple(np.full(20, x, dtype=float) for x in (10.0, 5.0, 5.0, 80.0))
        with pytest.raises(ValueError):
            _bootstrap_categorical_difference(counts, counts, "ets", 7, 100, 0)


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

    def test_a_method_without_a_point_forecast_gets_no_rmse_or_categorical_rows(self):
        # Tier 2b writes only probabilistic scores: no MSE, no contingency counts.
        probabilistic = _per_day("p", 24, 30, 100.0, seed=9)
        probabilistic["mse_mm2"] = np.nan
        for threshold in V.IMD_RAIN_THRESHOLDS_MM:
            for name in ("hits", "misses", "false_alarms", "correct_negatives"):
                probabilistic[f"{name}_{threshold}"] = np.nan
        per_day = pd.concat([probabilistic, _per_day("b", 24, 30, 120.0)], ignore_index=True)

        rows = compare_methods(per_day, "p", "b", 24, 7, 100, 0)

        metrics = {row["metric"] for row in rows}
        assert "crps_mm" in metrics
        assert not metrics & {"rmse_mm", "csi", "sedi"}
        assert all(np.isfinite(row["diff"]) for row in rows)

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

    def _leg_rows(self, metric, method_b, diffs, significant):
        return [
            {**self._scorecard_row(lead, d, significant, method_b=method_b),
             "metric": metric, "threshold": "64.5"}
            for lead, d in zip((24, 48, 72, 96, 120), diffs)
        ]

    def _two_leg_spec(self):
        return {
            **self._spec(),
            "metric": "brier",
            "threshold": "64.5",
            "legs": [
                {"claim": "HX-Brier", "summary": "leg 1", "method_b": "ref"},
                {"claim": "HX-SEDI", "summary": "leg 2", "method_b": "ref2", "metric": "sedi"},
            ],
        }

    def test_a_claim_fails_when_one_of_its_legs_fails(self):
        # H3: the Brier leg passing must not pass a claim whose SEDI leg fails.
        scorecard = pd.DataFrame(
            self._leg_rows("brier", "ref", [-1.0] * 5, True)
            + self._leg_rows("sedi", "ref2", [0.01, 0.01, -0.01, 0.01, -0.01], False)
        )

        rows = _verdicts_with_legs(self._two_leg_spec(), scorecard, {"m", "ref", "ref2"})

        assert [r["claim"] for r in rows] == ["HX-Brier", "HX-SEDI", "HX"]
        assert [r["status"] for r in rows] == ["PASS", "FAIL", "FAIL"]
        assert rows[-1]["reason"] == "Needs every leg: HX-Brier PASS; HX-SEDI FAIL."

    def test_a_claim_passes_only_when_every_leg_passes(self):
        both = pd.DataFrame(
            self._leg_rows("brier", "ref", [-1.0] * 5, True)
            + self._leg_rows("sedi", "ref2", [-0.1] * 5, True)
        )
        unproven = pd.DataFrame(
            self._leg_rows("brier", "ref", [-1.0] * 5, True)
            + self._leg_rows("sedi", "ref2", [-0.1] * 5, False)
        )
        available = {"m", "ref", "ref2"}

        assert _verdicts_with_legs(self._two_leg_spec(), both, available)[-1]["status"] == "PASS"
        assert (
            _verdicts_with_legs(self._two_leg_spec(), unproven, available)[-1]["status"]
            == "INSUFFICIENT_DATA"
        )
        untested = _verdicts_with_legs(self._two_leg_spec(), both, {"m", "ref"})
        assert [r["status"] for r in untested] == ["PASS", "NOT_YET_TESTED", "NOT_YET_TESTED"]

    def test_h3_is_judged_on_both_legs_of_the_preregistration(self, tmp_path):
        # docs/preregistration.md, H3: "Pass. Both conditions". The ledger used
        # to read the Brier leg alone.
        h3 = next(spec for spec in CLAIM_SPECS if spec["claim"] == "H3")
        legs = [{**h3, **leg} for leg in h3["legs"]]
        assert [(leg["method_b"], leg["metric"], leg["threshold"]) for leg in legs] == [
            ("climatology", "brier", "64.5"),
            ("best_single_member_on_train", "sedi", "64.5"),
        ]
        scorecard = pd.DataFrame(
            [{**r, "method_a": h3["method_a"], "method_b": "climatology"}
             for r in self._leg_rows("brier", "x", [-1.0] * 5, True)]
            + [{**r, "method_a": h3["method_a"], "method_b": "best_single_member_on_train"}
               for r in self._leg_rows("sedi", "x", [0.01] * 5, False)]
        )
        per_day = pd.concat(
            [_per_day(m, 24, 5, 100.0)
             for m in (h3["method_a"], "climatology", "best_single_member_on_train")]
        )
        write_verdicts(scorecard, per_day, tmp_path, str(PREREGISTRATION))
        written = pd.read_csv(tmp_path / "preregistration_verdicts.csv").set_index("claim")

        assert written.loc["H3-Brier", "status"] == "PASS"
        assert written.loc["H3-SEDI", "status"] == "FAIL"
        assert written.loc["H3", "status"] == "FAIL"

    def test_every_claim_in_the_specs_exists_in_the_preregistration_doc(self, tmp_path):
        # The doc is the source of truth; a claim implemented here but absent
        # there would be an unregistered claim masquerading as a registered
        # one. write_verdicts enforces this.
        per_day = _per_day("tier0", 24, 5, 100.0)
        write_verdicts(pd.DataFrame(), per_day, tmp_path, str(PREREGISTRATION))
        written = pd.read_csv(tmp_path / "preregistration_verdicts.csv")

        assert set(written["claim"]) == {spec["claim"] for spec in CLAIM_SPECS} | {
            leg["claim"] for spec in CLAIM_SPECS for leg in spec.get("legs", [])
        }

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



class TestVerdictFromRunnerTable:
    """H7 and H10 are read from their runners' paired tables (31 Oct, F2)."""

    H7 = {
        "claim": "H7-QM (x)", "summary": "s", "method_a": "a", "method_b": "b",
        "metric": "m", "threshold": "", "produced_by": "11",
        "runner_table": "qm.csv", "rule": "h7", "rows": {"scope": "tier1_blend"},
    }
    H10 = {
        "claim": "H10", "summary": "s", "method_a": "a", "method_b": "b",
        "metric": "crps_mm", "threshold": "", "produced_by": "13",
        "runner_table": "t2b.csv", "rule": "h10",
    }

    def _h7(self, tmp_path, tw_delta, tw_sig, sedi_delta, sedi_sig, brier_bad):
        pd.DataFrame(
            {
                "scope": "tier1_blend",
                "lead_hours": [24, 48, 72, 96, 120],
                "twcrps_64.5_delta": tw_delta,
                "twcrps_64.5_improved": tw_sig,
                "sedi_115.6_delta": sedi_delta,
                "sedi_115.6_improved": sedi_sig,
                "brier_7.5_degraded": brier_bad,
            }
        ).to_csv(tmp_path / "qm.csv", index=False)
        return _verdict_from_runner_table(self.H7, tmp_path)

    def test_h7_passes_on_either_leg(self, tmp_path):
        v = self._h7(tmp_path, [0.4] * 5, [False] * 5, [-0.2] * 5, [True] * 4 + [False],
                     [False] * 5)
        assert v["status"] == "PASS" and v["n_leads_significant"] == 4
        assert "SEDI" in v["reason"]

    def test_h7_fails_when_brier_is_worse_at_any_lead(self, tmp_path):
        v = self._h7(tmp_path, [-0.4] * 5, [True] * 5, [-0.2] * 5, [True] * 5,
                     [False] * 4 + [True])
        assert v["status"] == "FAIL" and "non-inferiority" in v["reason"]

    def test_h7_direction_without_ci_is_insufficient(self, tmp_path):
        v = self._h7(tmp_path, [-0.1] * 5, [False] * 5, [0.1] * 5, [False] * 5, [False] * 5)
        assert v["status"] == "INSUFFICIENT_DATA"

    def test_h10_needs_both_parents_beaten_at_a_lead(self, tmp_path):
        rows = []
        for lead in (24, 48, 72, 96, 120):
            for parent, diff in (("emos_csg", -0.02), ("bma", -0.01 if lead < 72 else 0.01)):
                rows.append({"lead_hours": lead, "arm": "quantile_avg", "is_nominated": True,
                             "vs_parent": parent, "crps_difference_mm": diff,
                             "beats_parent": diff < 0, "degenerate": False})
        pd.DataFrame(rows).to_csv(tmp_path / "t2b.csv", index=False)
        v = _verdict_from_runner_table(self.H10, tmp_path)
        assert v["status"] == "FAIL" and v["n_leads_improved"] == 2

    def test_missing_table_is_not_yet_tested(self, tmp_path):
        assert _verdict_from_runner_table(self.H10, tmp_path)["status"] == "NOT_YET_TESTED"
