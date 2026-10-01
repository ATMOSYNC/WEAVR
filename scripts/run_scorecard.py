#!/usr/bin/env python3
"""Build the ECMWF-style scorecard and the pre-registration verdicts.

Reads every per-day score file under `results/per_day/` (written by the tier
scripts via `weavr.score_io`) and produces two outputs:

- `results/scorecard.csv` -- one row per (metric, threshold, lead, method_a,
  method_b) with both estimates, their difference, a 95% paired
  block-bootstrap confidence interval, a Diebold-Mariano p-value, and whether
  the interval excludes 0. Every method is compared against the three
  references `docs/preregistration.md` cares about:
  `best_single_member_on_train` (step 01's honest baseline), `climatology`,
  and `tier0`.
- `results/preregistration_verdicts.csv` -- for each claim in
  `docs/preregistration.md`, a status of PASS / FAIL / INSUFFICIENT_DATA /
  NOT_YET_TESTED together with the evidence rows behind it.

**On today's weekly store this will report almost nothing as significant, and
that is the point.** With 3-4 test days per lead, the block bootstrap
degenerates to resampling the whole series (there is less than one 7-day
block), so intervals are enormous. Running it now proves the machinery works
and measures exactly how weak the old evidence base was; step 07 re-runs it
on daily, two-season data where the intervals mean something.

Categorical scores (CSI, SEDI) are rebuilt from per-day contingency **counts**
inside each bootstrap replicate rather than averaged from per-day scores --
a ratio of sums is not the sum of ratios, and averaging daily CSIs would give
a number that does not match the pooled CSI anyone would quote. The same
applies to RMSE, which is why `weavr.score_io` writes daily MSE.

Usage:
    python scripts/run_scorecard.py [--results-dir results]
        [--block-days N] [--n-resamples N] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weavr import verify as V  # noqa: E402
from weavr.score_io import read_per_day_scores  # noqa: E402
from weavr.significance import (  # noqa: E402
    DifferenceCI,
    diebold_mariano,
    lag1_autocorrelation,
    paired_difference_ci,
)

# The references every method is scored against, per the prompt and
# docs/preregistration.md's comparisons.
#
# The two single-source EMOS calibrations are here because the pre-registered
# H2 claim names them as its comparator: "the multi-source combiner beats the
# best single-source EMOS on CRPS". Without them in the reference set that
# comparison was never computed and H2 could only ever report
# NOT_YET_TESTED, whatever the numbers said.
#
# Comparing against *both* rather than picking one is deliberate, and it is
# what removes the need for train-fold selection. H2 asks whether the
# multi-source combiner beats the BEST single-source EMOS. A method that beats
# both of them necessarily beats whichever is best, so the claim is decided
# without choosing a comparator on the data being tested -- which is exactly
# the leakage the pre-registration was guarding against, and why
# `needs_train_scores` is no longer set on H2.
REFERENCE_METHODS = (
    "best_single_member_on_train",
    "climatology",
    "tier0",
    "tier2_emos_graphcast",
    "tier2_emos_ifs_ens",
)

# Metrics that are a plain mean of a per-day column.
MEAN_METRICS = {
    "mae_mm": "mae_mm",
    "crps_mm": "crps_mm",
    "twcrps_64.5_mm": "twcrps_64.5_mm",
}


def _rmse_from_mse(frame: pd.DataFrame) -> np.ndarray:
    return frame["mse_mm2"].to_numpy(dtype=float)


def _categorical_from_counts(
    frame: pd.DataFrame, threshold: float, score: str
) -> tuple[np.ndarray, ...]:
    """The four daily count columns for one threshold, as arrays."""
    return tuple(
        frame[f"{name}_{threshold}"].to_numpy(dtype=float)
        for name in ("hits", "misses", "false_alarms", "correct_negatives")
    )


def _pooled_csi(hits: float, misses: float, false_alarms: float) -> float:
    denominator = hits + misses + false_alarms
    return float(hits / denominator) if denominator > 0 else float("nan")


def _pooled_sedi(
    hits: float, misses: float, false_alarms: float, correct_negatives: float
) -> float:
    """SEDI from pooled counts, with Ferro & Stephenson's own degenerate cases.

    Mirrors `weavr.verify.sedi` exactly, but from scalars, so it can be
    evaluated inside a bootstrap replicate without rebuilding a DataArray.
    """
    hit_denominator = hits + misses
    false_denominator = false_alarms + correct_negatives
    if hit_denominator <= 0 or false_denominator <= 0:
        return float("nan")
    h = hits / hit_denominator
    f = false_alarms / false_denominator
    if h <= 0.0 or h >= 1.0 or f <= 0.0 or f >= 1.0:
        return float("nan")
    numerator = np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h)
    denominator = np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)
    return float(numerator / denominator) if denominator != 0 else float("nan")


def _bootstrap_categorical_difference(
    counts_a: tuple[np.ndarray, ...],
    counts_b: tuple[np.ndarray, ...],
    score: str,
    block_days: int,
    n_resamples: int,
    seed: int,
) -> DifferenceCI:
    """CI for a pooled categorical score difference, rebuilt per replicate.

    Positively-oriented scores (CSI, SEDI: higher is better) are negated so
    that, like every loss in this file, a **negative difference means method
    A is better**. Keeping one orientation everywhere is what lets the
    scorecard's `significant` column be read without a per-metric lookup.
    """
    from weavr.significance import block_bootstrap_indices, bootstrap_is_degenerate

    score_fn = _pooled_csi if score == "csi" else _pooled_sedi
    n_args = 3 if score == "csi" else 4

    def pooled(counts: tuple[np.ndarray, ...], index: np.ndarray | None = None) -> float:
        selected = [c if index is None else c[index] for c in counts[:n_args]]
        return score_fn(*[float(np.sum(c)) for c in selected])

    n_days = counts_a[0].size
    estimate = -(pooled(counts_a) - pooled(counts_b))

    # Same guard as paired_difference_ci: a block that spans the whole series
    # has one possible resample, so the interval would have zero width and
    # read as significant for any non-zero difference.
    if bootstrap_is_degenerate(n_days, block_days):
        return DifferenceCI(
            estimate=float(estimate),
            ci_lo=float("nan"),
            ci_hi=float("nan"),
            n_days=n_days,
            block_days=min(block_days, n_days),
            n_resamples=n_resamples,
            degenerate=True,
        )

    indices = block_bootstrap_indices(n_days, block_days, n_resamples, seed)
    replicates = np.array(
        [-(pooled(counts_a, idx) - pooled(counts_b, idx)) for idx in indices]
    )
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        ci_lo = ci_hi = float("nan")
    else:
        ci_lo, ci_hi = (float(x) for x in np.quantile(finite, [0.025, 0.975]))

    return DifferenceCI(
        estimate=float(estimate),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_days=n_days,
        block_days=min(block_days, n_days),
        n_resamples=n_resamples,
    )


def compare_methods(
    per_day: pd.DataFrame,
    method_a: str,
    method_b: str,
    lead: int,
    block_days: int,
    n_resamples: int,
    seed: int,
) -> list[dict]:
    """Every available metric comparison between two methods at one lead.

    Methods are paired on their shared dates only. If they were scored on
    different day sets (which should not happen, but would silently produce a
    meaningless comparison), the intersection is used and `n_days` records it.
    """
    frame_a = per_day[(per_day["method"] == method_a) & (per_day["lead_hours"] == lead)]
    frame_b = per_day[(per_day["method"] == method_b) & (per_day["lead_hours"] == lead)]
    if frame_a.empty or frame_b.empty:
        return []

    shared = sorted(set(frame_a["date"]) & set(frame_b["date"]))
    if len(shared) < 2:
        return []
    frame_a = frame_a[frame_a["date"].isin(shared)].sort_values("date")
    frame_b = frame_b[frame_b["date"].isin(shared)].sort_values("date")

    rows: list[dict] = []

    def record(
        metric: str,
        threshold: str,
        estimate_a: float,
        estimate_b: float,
        ci: DifferenceCI,
        dm_p: float,
    ) -> None:
        rows.append(
            {
                "metric": metric,
                "threshold": threshold,
                "lead_hours": lead,
                "method_a": method_a,
                "method_b": method_b,
                "estimate_a": estimate_a,
                "estimate_b": estimate_b,
                "diff": ci.estimate,
                "ci_lo": ci.ci_lo,
                "ci_hi": ci.ci_hi,
                "dm_p": dm_p,
                "significant": ci.significant,
                "degenerate_bootstrap": ci.degenerate,
                "n_days": ci.n_days,
                "block_days": ci.block_days,
            }
        )

    # RMSE, from daily MSE.
    mse_a, mse_b = _rmse_from_mse(frame_a), _rmse_from_mse(frame_b)
    ci = paired_difference_ci(
        mse_a, mse_b, block_days, n_resamples, seed, aggregate="rmse"
    )
    record(
        "rmse_mm",
        "",
        float(np.sqrt(np.mean(mse_a))),
        float(np.sqrt(np.mean(mse_b))),
        ci,
        diebold_mariano(mse_a, mse_b).p_value,
    )

    for metric, column in MEAN_METRICS.items():
        if column not in frame_a.columns or column not in frame_b.columns:
            continue
        values_a = frame_a[column].to_numpy(dtype=float)
        values_b = frame_b[column].to_numpy(dtype=float)
        if not (np.isfinite(values_a).any() and np.isfinite(values_b).any()):
            continue
        ci = paired_difference_ci(values_a, values_b, block_days, n_resamples, seed)
        record(
            metric,
            "",
            float(np.nanmean(values_a)),
            float(np.nanmean(values_b)),
            ci,
            diebold_mariano(values_a, values_b).p_value,
        )

    for threshold in V.IMD_RAIN_THRESHOLDS_MM:
        brier_column = f"brier_{threshold}"
        if brier_column in frame_a.columns and brier_column in frame_b.columns:
            values_a = frame_a[brier_column].to_numpy(dtype=float)
            values_b = frame_b[brier_column].to_numpy(dtype=float)
            if np.isfinite(values_a).any() and np.isfinite(values_b).any():
                ci = paired_difference_ci(values_a, values_b, block_days, n_resamples, seed)
                record(
                    "brier",
                    str(threshold),
                    float(np.nanmean(values_a)),
                    float(np.nanmean(values_b)),
                    ci,
                    diebold_mariano(values_a, values_b).p_value,
                )

        counts_a = _categorical_from_counts(frame_a, threshold, "csi")
        counts_b = _categorical_from_counts(frame_b, threshold, "csi")
        for score in ("csi", "sedi"):
            ci = _bootstrap_categorical_difference(
                counts_a, counts_b, score, block_days, n_resamples, seed
            )
            fn = _pooled_csi if score == "csi" else _pooled_sedi
            n_args = 3 if score == "csi" else 4
            estimate_a = fn(*[float(np.sum(c)) for c in counts_a[:n_args]])
            estimate_b = fn(*[float(np.sum(c)) for c in counts_b[:n_args]])
            record(score, str(threshold), estimate_a, estimate_b, ci, float("nan"))

    return rows


def build_scorecard(
    per_day: pd.DataFrame, block_days: int, n_resamples: int, seed: int
) -> pd.DataFrame:
    methods = sorted(per_day["method"].unique())
    leads = sorted(per_day["lead_hours"].unique())
    rows: list[dict] = []
    for lead in leads:
        for method in methods:
            for reference in REFERENCE_METHODS:
                if method == reference or reference not in methods:
                    continue
                rows.extend(
                    compare_methods(
                        per_day, method, reference, lead, block_days, n_resamples, seed
                    )
                )
    return pd.DataFrame(rows)


def measure_block_length(per_day: pd.DataFrame) -> pd.DataFrame:
    """Lag-1 autocorrelation of each method's daily MSE, per lead.

    `docs/preregistration.md` fixes the bootstrap block length at 7 days
    "unless step 07 measures a different autocorrelation length". This is
    that measurement, reported so the doc can be amended (in its own commit,
    before any verdict is computed) if the data says otherwise.
    """
    rows = []
    for (method, lead), group in per_day.groupby(["method", "lead_hours"]):
        ordered = group.sort_values("date")
        rows.append(
            {
                "method": method,
                "lead_hours": lead,
                "n_days": len(ordered),
                "lag1_autocorrelation_mse": lag1_autocorrelation(
                    ordered["mse_mm2"].to_numpy(dtype=float)
                ),
            }
        )
    return pd.DataFrame(rows)


# --- Pre-registration verdicts -----------------------------------------------

# One entry per claim in docs/preregistration.md. `produced_by` names the step
# whose output the claim needs, so a NOT_YET_TESTED verdict says *what is
# missing* rather than just "no data". Claims whose comparator must be chosen
# on the training fold carry `needs_train_scores`: today's per-day files are
# test-only, and step 07 is what adds the train side.
#
# The headline configuration (H1-RMSE, H1-CRPS, H3) is `tier2_bma`, declared
# here on **design** grounds before the verdicts exist, not by picking whichever
# tier happened to score best: choosing it after seeing the two-season results
# would select on the test set and quietly invalidate all three claims. The
# reasoning is only that `tier2_bma` is already the declared multi-source
# combiner for H2 immediately below, so H1/H3/H2 now all test the same object,
# and it is the calibrated layer, which H3's Brier/SEDI claim specifically
# needs (Tier 1 has no probabilistic calibration to score). If Tier 2 turns out
# to lose to Tier 1, that is the finding to report, not a reason to swap the
# declaration.
HEADLINE_CONFIGURATION = "tier2_bma"

CLAIM_SPECS: list[dict] = [
    {
        "claim": "H1-RMSE",
        "summary": "A WEAVR blend beats the best single member (chosen on train), on RMSE",
        "method_a": HEADLINE_CONFIGURATION,
        "method_b": "best_single_member_on_train",
        "metric": "rmse_mm",
        "threshold": "",
        "produced_by": "07 (headline declared; tier2_bma per-day scores absent)",
    },
    {
        "claim": "H1-CRPS",
        "summary": (
            "A WEAVR blend beats the best single member (chosen on train), on CRPS "
            "-- as written; not evaluable, a single member has no ensemble spread"
        ),
        "method_a": HEADLINE_CONFIGURATION,
        "method_b": "best_single_member_on_train",
        "metric": "crps_mm",
        "threshold": "",
        "produced_by": (
            "07: `best_single_member_on_train` is one deterministic member, so its "
            "per-day file carries no crps_mm column. See the H1-CRPS (point-mass "
            "identity) row below for the evaluable form of the same claim."
        ),
    },
    {
        "claim": "H1-CRPS (point-mass identity)",
        "summary": (
            "A WEAVR blend beats the best single member (chosen on train), on CRPS "
            "-- CRPS of a deterministic forecast is its MAE, exactly"
        ),
        "method_a": HEADLINE_CONFIGURATION,
        "method_b": "best_single_member_on_train",
        "metric": "mae_mm",
        "threshold": "",
        "produced_by": (
            "07: for a point mass at x, CRPS = E|X-x| - 0.5 E|X-X'| = |y-x|, "
            "which is the MAE. This is an identity, not a substitute metric, so "
            "the pre-registered claim is decided by the mae_mm comparison "
            "against the same comparator."
        ),
    },
    {
        "claim": "H2",
        "summary": (
            "The multi-source combiner beats the best single-source EMOS on CRPS "
            "-- evaluated against GraphCast-EMOS"
        ),
        "method_a": "tier2_bma",
        "method_b": "tier2_emos_graphcast",
        "metric": "crps_mm",
        "threshold": "",
        "produced_by": "07 (both single-source EMOSs are compared; see REFERENCE_METHODS)",
    },
    {
        "claim": "H2",
        "summary": (
            "The multi-source combiner beats the best single-source EMOS on CRPS "
            "-- evaluated against IFS-ENS-EMOS"
        ),
        "method_a": "tier2_bma",
        "method_b": "tier2_emos_ifs_ens",
        "metric": "crps_mm",
        "threshold": "",
        "produced_by": "07 (both single-source EMOSs are compared; see REFERENCE_METHODS)",
    },
    {
        "claim": "H3",
        "summary": "P(>=64.5mm) has BSS > 0 vs climatology and higher SEDI than the best member",
        "method_a": HEADLINE_CONFIGURATION,
        "method_b": "climatology",
        "metric": "brier",
        "threshold": "64.5",
        "produced_by": "07 (headline declared; tier2_bma probabilities absent)",
    },
    {
        "claim": "H4",
        "summary": "Adding NEPS-G improves CRPS at 24 h",
        "method_a": "neps_g_blend",
        "method_b": "blend_without_neps_g",
        "metric": "crps_mm",
        "threshold": "",
        "produced_by": "08",
    },
    {
        "claim": "H5",
        "summary": "WEAVR vs HEPPI's published NEPS-G EMOS on CRPS at 24 h",
        "method_a": "weavr_blend",
        "method_b": "heppi_neps_g_emos",
        "metric": "crps_mm",
        "threshold": "",
        "report_only": True,
        "produced_by": "08",
    },
    {
        "claim": "H6",
        "summary": "Independence-aware weights beat OLS fit_region_weights on RMSE",
        "method_a": "independence_weighted",
        "method_b": "tier1_regional",
        "metric": "rmse_mm",
        "threshold": "",
        "produced_by": "09",
    },
    {
        "claim": "H7",
        "summary": "Tail repair improves tw-CRPS or SEDI without harming Brier at 7.5mm",
        "method_a": "tail_repaired",
        "method_b": None,
        "metric": "twcrps_64.5_mm",
        "threshold": "",
        "produced_by": "11, 12",
    },
    {
        "claim": "H8",
        "summary": "District Orange/Red warnings beat the same rule on the best raw ensemble",
        "method_a": "district_warnings_weavr",
        "method_b": "district_warnings_raw",
        "metric": "csi",
        "threshold": "64.5",
        "produced_by": "14",
    },
    {
        "claim": "H9",
        "summary": "Event replay: first lead at which Orange and Red are issued",
        "method_a": "event_replay",
        "method_b": None,
        "metric": "",
        "threshold": "",
        "report_only": True,
        "produced_by": "19",
    },
    {
        "claim": "H10",
        "summary": "A stacked combiner beats both EMOS and BMA on CRPS",
        "method_a": "tier2b_stacked",
        "method_b": None,
        "metric": "crps_mm",
        "threshold": "",
        "produced_by": "13",
    },
    {
        "claim": "H11",
        "summary": "Online weights beat static Tier 1 weights on a sequential replay",
        "method_a": "online_weights",
        "method_b": "tier1_regional",
        "metric": "rmse_mm",
        "threshold": "",
        "produced_by": "21",
    },
]

MIN_LEADS_FOR_PASS = 3


def _verdict_for_claim(
    spec: dict, scorecard: pd.DataFrame, available: set[str]
) -> dict:
    """Apply docs/preregistration.md's default pass rule to one claim.

    The rule is: the improvement holds at >= 3 of 5 leads, AND the CI of the
    paired difference excludes 0 at those leads. Mapped to a status as:

    - **PASS** -- both halves hold.
    - **FAIL** -- the *direction* fails (improvement at fewer than 3 leads).
      That is a real negative result, not a power problem.
    - **INSUFFICIENT_DATA** -- the direction holds at >= 3 leads but the
      intervals do not exclude 0. The claim is unproven, not disproven, and
      saying "FAIL" here would misreport a sample-size limit as evidence
      against the claim.
    - **NOT_YET_TESTED** -- the step that produces the method has not run.
    """
    method_a, method_b = spec.get("method_a"), spec.get("method_b")
    base = {
        "claim": spec["claim"],
        "summary": spec["summary"],
        "method_a": method_a or "(not yet declared)",
        "method_b": method_b or "(not yet declared)",
        "metric": spec["metric"],
        "threshold": spec["threshold"],
        "n_leads_improved": 0,
        "n_leads_significant": 0,
        "leads_significant": "",
        "evidence_rows": 0,
    }

    if spec.get("report_only"):
        return {
            **base,
            "status": "NOT_YET_TESTED",
            "reason": (
                f"Report-either-way result, produced by step {spec['produced_by']}. "
                "No pass/fail is defined for it."
            ),
        }

    missing = [m for m in (method_a, method_b) if m is None or m not in available]
    if missing:
        named = ", ".join(str(m) for m in missing)
        reason = f"Awaiting step {spec['produced_by']}; not in results/per_day/: {named}."
        if spec.get("needs_train_scores"):
            reason += (
                " The comparator must be selected on the training fold, and today's "
                "per-day scores cover the test split only."
            )
        return {**base, "status": "NOT_YET_TESTED", "reason": reason}

    rows = scorecard[
        (scorecard["method_a"] == method_a)
        & (scorecard["method_b"] == method_b)
        & (scorecard["metric"] == spec["metric"])
        & (scorecard["threshold"].astype(str) == str(spec["threshold"]))
    ]
    if rows.empty:
        return {
            **base,
            "status": "NOT_YET_TESTED",
            "reason": "No scorecard comparison exists for this metric and method pair.",
        }

    improved = rows[rows["diff"] < 0]
    significant = improved[improved["significant"]]
    base.update(
        {
            "n_leads_improved": int(len(improved)),
            "n_leads_significant": int(len(significant)),
            "leads_significant": "+".join(
                str(int(x)) for x in sorted(significant["lead_hours"])
            ),
            "evidence_rows": int(len(rows)),
        }
    )

    if len(improved) < MIN_LEADS_FOR_PASS:
        return {
            **base,
            "status": "FAIL",
            "reason": (
                f"Improvement at only {len(improved)} of {len(rows)} leads; the rule "
                f"needs at least {MIN_LEADS_FOR_PASS}. The direction itself fails."
            ),
        }
    if len(significant) < MIN_LEADS_FOR_PASS:
        return {
            **base,
            "status": "INSUFFICIENT_DATA",
            "reason": (
                f"Improvement at {len(improved)} leads, but the CI excludes 0 at only "
                f"{len(significant)}. Unproven, not disproven -- "
                f"median n_days = {int(rows['n_days'].median())}."
            ),
        }
    return {
        **base,
        "status": "PASS",
        "reason": (
            f"Improvement at {len(improved)} of {len(rows)} leads, with the CI excluding "
            f"0 at {len(significant)}."
        ),
    }


def write_verdicts(
    scorecard: pd.DataFrame, per_day: pd.DataFrame, out_dir: Path, preregistration: str
) -> Path:
    """Write `results/preregistration_verdicts.csv`, one row per claim."""
    doc = Path(preregistration)
    if not doc.is_file():
        raise FileNotFoundError(
            f"{doc} not found. The verdicts are meaningless without the "
            "pre-registration they are judged against."
        )
    registered = {spec["claim"].split("-")[0] for spec in CLAIM_SPECS}
    text = doc.read_text()
    unregistered = [c for c in sorted(registered) if f"### {c} " not in text]
    if unregistered:
        raise ValueError(
            f"Claims {unregistered} are implemented here but have no section in {doc}. "
            "The doc is the source of truth; reconcile before publishing verdicts."
        )

    available = set(per_day["method"].unique())
    verdicts = pd.DataFrame(
        [_verdict_for_claim(spec, scorecard, available) for spec in CLAIM_SPECS]
    )
    path = out_dir / "preregistration_verdicts.csv"
    verdicts.to_csv(path, index=False)

    counts = verdicts["status"].value_counts().to_dict()
    print(f"Verdicts: {counts}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preregistration", default="docs/preregistration.md")
    args = parser.parse_args()

    per_day = read_per_day_scores(args.results_dir)
    if per_day.empty:
        print(
            f"No per-day scores under {args.results_dir}/per_day/. "
            "Run the tier scripts first."
        )
        return 1

    methods = sorted(per_day["method"].unique())
    leads = sorted(per_day["lead_hours"].unique())
    print(f"Per-day scores: {len(per_day)} rows, {len(methods)} methods, leads {leads}")
    print(f"Methods: {', '.join(methods)}")
    days_per_lead = per_day.groupby("lead_hours")["date"].nunique().to_dict()
    print(f"Distinct days per lead: {days_per_lead}")
    if min(days_per_lead.values()) <= args.block_days:
        print(
            f"\nNOTE: at some leads there are no more days than the {args.block_days}-day "
            "block length, so the moving-block bootstrap has only ONE possible resample "
            "and cannot produce an interval at all. Those comparisons are written with "
            "NaN bounds, degenerate_bootstrap=True and significant=False. That is the "
            "correct answer for a 3-4 day test set, and it is what step 07's daily, "
            "two-season data exists to fix."
        )

    scorecard = build_scorecard(per_day, args.block_days, args.n_resamples, args.seed)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = out_dir / "scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    print(f"\nWrote {scorecard_path} ({len(scorecard)} comparisons)")
    if not scorecard.empty:
        n_significant = int(scorecard["significant"].sum())
        n_degenerate = int(scorecard["degenerate_bootstrap"].sum())
        print(
            f"Significant (CI excludes 0): {n_significant} of {len(scorecard)} "
            f"({100 * n_significant / len(scorecard):.1f}%)"
        )
        print(
            f"No interval computable (too few days to resample): {n_degenerate} of "
            f"{len(scorecard)} ({100 * n_degenerate / len(scorecard):.1f}%)"
        )

    autocorrelation_path = out_dir / "block_length_diagnostic.csv"
    measure_block_length(per_day).to_csv(autocorrelation_path, index=False)
    print(f"Wrote {autocorrelation_path}")

    verdicts_path = write_verdicts(scorecard, per_day, out_dir, args.preregistration)
    print(f"Wrote {verdicts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
