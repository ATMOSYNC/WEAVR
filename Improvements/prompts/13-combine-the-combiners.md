# Step 13: Combine the combiners (EMOS-CSG + BMA)

**Type**: method + evaluation prompt.
**Plan items**: B5.
**Depends on**: step 07 (v2 EMOS and BMA outputs under LOYO).

## Goal

Phase 4 found EMOS-CSG and BMA genuinely trade off. Javanshiri (2021)
found the same: BMA has better reliability, EMOS better discrimination.
Slide 2 already claims WEAVR "uses the strongest combiner per bin", but no
such selector exists (plan §2 F8).

Build and test two ways to use both, under pre-registered H10:

- **(a) Per-bin × lead selection:** choose, on the *train* fold, whichever
  combiner has the lower CRPS for each (rain bin, lead).
- **(b) Quantile averaging (Vincentization)** of the two predictive
  distributions. Lichtendahl et al. (2013) show quantile averaging keeps
  sharpness, while the linear pool (averaging probabilities) tends to
  over-disperse.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Add src/weavr/stacking.py (pure, tested):
   - select_per_bin(train_scores: DataFrame[bin, lead, combiner, crps]) ->
     mapping (bin, lead) -> combiner. Tie-break rule documented; a bin
     with no train cells falls back to the pooled-best combiner, flagged.
   - predictive_quantiles_csgd(mean, std, shift, levels): use the
     CSGD's inverse CDF, handling the censoring point mass at 0.
   - predictive_quantiles_bma(result, forecasts, levels, rng,
     n_samples): empirical quantiles of weavr.bma.sample_bma_mixture
     samples.
   - quantile_average(q_a, q_b, weight=0.5) -> averaged quantiles
     (monotone by construction; assert it).
   - crps_from_quantiles(q, levels, obs): the quantile-score
     approximation CRPS ~ 2 * mean over levels of the pinball loss, with
     enough levels (e.g. 99) to be accurate. Test it against the
     closed-form csgd_crps on a known CSGD (error below a stated
     tolerance).
   Also implement the linear pool, as a comparison only.

2. Add scripts/run_tier2b_combined.py (LOYO, pooled). For each lead:
   EMOS-CSG (best source), BMA, per-bin selection, quantile average (the
   weight fit on train by CRPS over a small grid, or fixed at 0.5 if the
   train fit is unstable; document which) and the linear pool.
   Metrics: CRPS, Brier at the IMD thresholds, PIT/reliability summary.
   Write results/tier2b_combined.csv and per-day outputs. Compute H10.

3. If H10 PASSES: the winning method becomes the recommended
   probabilistic combiner in docs, and a flag in
   export_dashboard_example_grids.py. Slide 2's "strongest combiner per
   bin" claim may then return, with the CI. If H10 FAILS: report it, and
   keep slide 2 without that claim.

4. docs/tier2b-combined-results.md (methods, table, verdict, the
   reliability-vs-discrimination discussion tied back to Javanshiri 2021
   and Ji 2025 from research/), plus a README paragraph.
```

## Done when

- `weavr.stacking` is tested, including the CRPS-from-quantiles accuracy
  check.
- All four methods are compared under LOYO with CIs.
- H10 has a verdict that decides the fate of slide 2's claim.
