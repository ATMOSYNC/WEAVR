# Step 04: Significance, honest baselines and the scorecard

**Type**: implementation prompt (new module, verification additions,
per-day score outputs).
**Plan items**: A4.
**Depends on**: step 01 (the single-source baselines) and step 02 (the
pre-registered claims this machinery will judge).

## Goal

Build the machinery that turns WEAVR's numbers into claims with error bars.
Every later step reuses it.

- Paired block-bootstrap confidence intervals and a Diebold–Mariano test.
- Climatology baselines and skill scores (CRPSS, BSS), plus
  reliability/PIT, SEDI and threshold-weighted CRPS. Research brief §3.5–3.6
  recommends these, and they are missing from `weavr.verify`.
- Every scoring script also writes **per-day** domain-wide scores, so
  differences can be bootstrapped over days.
- An ECMWF-style scorecard CSV, plus an automatic verdict for each
  pre-registered claim.

Prove it end-to-end on today's weekly data. Expect wide CIs and mostly
"insufficient evidence" verdicts; that is the honest point of running it
now. Step 07 re-runs it on the daily, two-season data.

## Why the machinery comes before the new data

- Step 07 should produce v2 results *and* their verdicts in one pass, not
  numbers first and CIs later.
- The tier scripts write only aggregated CSVs today. Differences can't be
  bootstrapped without per-day scores, so every scoring script needs a small
  change. Do it once, here.
- IMD/NCMRWF scientists read ECMWF-style scorecards daily. One scorecard
  slide replaces pages of prose.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Add src/weavr/significance.py (pure numpy/xarray, fully unit-tested):
   - block_bootstrap_indices(n_days, block_days, n_resamples, seed):
     moving-block resampling over the day axis, blocks kept contiguous in
     time, with the seed fixed for reproducibility.
   - paired_difference_ci(loss_a, loss_b, block_days=7, n_resamples=1000,
     seed=0, aggregate="mean" | "rmse"): the CI of aggregate(a) -
     aggregate(b), bootstrapping days jointly.
     For RMSE, bootstrap the daily MSE and take the sqrt of the mean inside
     each resample. Never average daily RMSEs.
   - diebold_mariano(loss_a, loss_b, horizon=1): the DM statistic with a
     Newey-West (HAC) variance using horizon-1 lags, and a two-sided
     p-value.
   - lag1_autocorrelation(series): used by step 07 to choose the block
     length.
   Tests: identical losses give a CI containing 0 and DM p ~ 1; a clear,
   consistent winner gives a CI excluding 0; the same seed gives identical
   output; blocks never wrap across the series end unless documented.

2. Extend src/weavr/verify.py. Each function gets a docstring citing its
   formula source (the module's existing convention):
   - sedi(forecast, obs, threshold): Ferro & Stephenson (2011).
     SEDI = [ln F - ln H - ln(1-F) + ln(1-H)] / [ln F + ln H + ln(1-F) + ln(1-H)].
     Handle H or F of 0 or 1 explicitly (return NaN and document it; don't
     crash, and don't return a silently wrong number).
   - twcrps_ensemble(ensemble, obs, threshold, member_dim): threshold-
     weighted CRPS with weight 1{z >= t}, using the chaining-function
     identity (Allen et al. 2023): twCRPS(F, y) = CRPS of max(members, t)
     vs max(y, t). Reuse crps(). Test against brute-force numerical
     integration on a small synthetic case.
   - pit_values(ensemble, obs) (rank histogram, random tie-breaking) and
     pit_values_csgd(mean, std, shift, obs) (randomized PIT on the point
     mass at 0).
   - reliability_table(prob, obs_binary, n_bins=10): forecast-probability
     bin centres, observed frequency and counts.
   - brier_skill_score(bs, bs_ref) and crpss(crps, crps_ref) helpers.

3. Add src/weavr/climatology.py:
   - climatological_ensemble(imd_climatology_store, target_dates,
     exclude_years, window_days=15): for each target date and grid cell,
     the members are IMD values from the same day-of-season +/- the
     window, across the archive years EXCLUDING the test year. Including
     the test year would leak the answer into the reference forecast.
   - climatological_probability(..., threshold): the event frequency of
     that same ensemble.
   Tests on a synthetic store, including one that proves exclusion works.

4. Per-day score outputs.
   - Add a shared helper (e.g. src/weavr/score_io.py:
     write_per_day_scores(method, lead, per_day: pandas.DataFrame, out_dir))
     writing results/per_day/<method>__lead<h>.csv. Columns: date, fold,
     and one per metric (mse_mm2, mae_mm, crps_mm where applicable,
     brier_<t>, twcrps_64.5, and hit/miss/false_alarm/correct_negative
     counts at each IMD threshold, so SEDI/CSI can be recomputed on
     resampled days).
   - Wire it into: run_single_source_baseline.py, run_tier0_baseline.py,
     run_tier1_regional_baseline.py, run_tier2_hierarchical_baseline.py
     (EMOS per source, BMA; score_csgd / score_bma already return
     unreduced per-cell DataArrays, so average them over lat/lon per
     sample), run_tier3_regime_conditioned_baseline.py and
     run_phase2_ensemble_baseline.py. Add a climatology "method" computed
     from item 3.
   - Keep every existing aggregated CSV exactly as it is. This is additive
     only.
   - Re-run each script on today's weekly store, and check the existing
     aggregated CSVs are unchanged (git diff shows no changes there).

5. Add scripts/run_scorecard.py, which reads results/per_day/ and writes:
   - results/scorecard.csv with rows (metric, threshold, lead, method_a,
     method_b): estimate_a, estimate_b, diff, ci_lo, ci_hi, dm_p, and
     significant (CI excludes 0).
     Required comparisons for every method: vs best_single_member_on_train
     (from step 01), vs climatology, vs tier0.
   - results/preregistration_verdicts.csv: for each claim in
     docs/preregistration.md, its status (PASS / FAIL / INSUFFICIENT_DATA /
     NOT_YET_TESTED) and the evidence rows. Implement the pass rules
     exactly as written there. If the doc is ambiguous for some claim,
     stop and ask; don't interpret it silently.

6. Write docs/scorecard-and-significance.md: the methods, the formulas and
   today's weekly-data results, stated plainly (e.g. "with 3-4 test days,
   no comparison is significant"). The point is that the machinery works
   and the old evidence base was too small, not a new claim.
   Add a README paragraph.
```

## Done when

- `weavr.significance`, the new `weavr.verify` functions and
  `weavr.climatology` are tested.
- Every scoring script writes per-day scores, with the existing aggregate
  CSVs unchanged.
- `results/scorecard.csv` and `results/preregistration_verdicts.csv` are
  produced from today's data.
- The doc states honestly what weekly data can and cannot support.
