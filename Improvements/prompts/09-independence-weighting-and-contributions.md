# Step 09: Independence-aware weighting and per-model contribution maps

**Type**: method + evaluation prompt (the signature idea).
**Plan items**: B1.
**Depends on**: step 07 (v2 data and LOYO). Step 08 is recommended, so
NEPS-G's contribution can be shown at 24 h.

## Goal

Turn the plan's F3 finding ("three models ≈ 1.1 independent opinions") into
three deliverables:

1. **Diagnostics:** the error-correlation matrix and N_eff per region ×
   lead. Step 01 built the functions; this step makes them per-fold and
   train-only, and adds the dashboard-ready outputs.
2. **Shapley skill attribution:** how much of the blend's skill gain over
   climatology each model contributes, per region × lead. It answers "what
   does adding GraphCast / NEPS-G / HRES actually buy?"
3. **Shrinkage-covariance weights:** a minimum-variance combination whose
   error covariance is estimated with Ledoit–Wolf shrinkage. It is compared
   against today's OLS `fit_region_weights` under pre-registered H6.

## Why this is WEAVR's unique contribution

- The research brief lists correlation-aware weighting for AI+NWP weather
  blends as an **open question never tested over India** (§1.5, and
  "stand out" #3).
- Tier 1's weight map shows HRES = 0 in most cells without explaining why.
  A contribution map explains it: IFS-ENS already carries HRES's
  information.
- Even if H6 fails, (1) and (2) remain unique, explainable deliverables. A
  well-reported no-go on (3) is still a strong result.

## Technical notes to check, not assume

- `w ∝ Σ⁻¹1` (Bates & Granger 1969) is the minimum-variance combination for
  **unbiased** members. Either debias each source per region on the train
  fold (subtract its mean error) before estimating Σ, or justify
  otherwise.
- Check what `weavr.weighting.fit_weights_least_squares` actually fits
  (intercept or not; constrained or not) before claiming how the two
  methods differ. Unconstrained OLS already reacts to collinearity. The
  expected difference is **stability with short training data**, from the
  shrinkage.
- Ledoit–Wolf has a closed-form shrinkage intensity. Implement it in numpy
  (no new dependency), and test it against a hand-computed small case.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Extend src/weavr/independence.py:
   - ledoit_wolf_covariance(errors_2d) -> (sigma, shrinkage), closed form,
     shrinking towards a scaled-identity target; test it against a
     hand-computed case.
   - min_variance_weights(sigma) -> w proportional to inverse(sigma)
     times the ones vector, normalised, then passed through the existing
     weavr.weighting.clip_and_renormalize. Fall back to equal weights (with
     is_fallback and reason) when sigma is singular even after shrinkage
     or when train points are below MIN_TRAIN_SAMPLES. Mirror
     RegionWeightResult's shape, so blend_with_region_weights works
     unchanged.
   - fit_region_weights_shrinkage(...): same signature and return shape as
     fit_region_weights, with per-source per-region debiasing on the train
     fold (the bias is stored in the result and applied at blend time).
   - shapley_attribution(value_fn, sources): exact Shapley values over all
     2^N - 1 non-empty subsets, where value_fn(subset) is the held-out
     skill gain over climatology (define it as RMSE_climatology -
     RMSE_blend(subset), or the CRPS analogue for probabilistic blends).
     Test that values sum to v(all) and that symmetric sources get equal
     values.

2. Add scripts/run_independence_weighting.py (LOYO, both folds, pooled):
   - Per region x lead: N_eff and participation ratio (train only); OLS
     vs shrinkage-weight blends scored on test; the weights themselves;
     Shapley values for the deterministic Tier 1-style blend over
     {graphcast, hres, ifs_ens_mean}.
   - At 24 h for 2018/2019, repeat including neps_g (step 08's
     configuration A, and B for graphcast).
   - Write results/independence_weights_comparison.csv,
     results/shapley_contributions.csv and
     results/independence_by_region.csv, plus per-day outputs (step 04
     helper).
   - Compute H6's verdict per docs/preregistration.md.

3. If H6 PASSES: add the shrinkage weights as an option to
   scripts/run_daily_pipeline.py and export_dashboard_example_grids.py,
   behind a flag. Don't silently switch the default; say which combiner the
   blended map uses. If H6 FAILS: keep OLS, and report why (e.g. weights
   near-identical, or shrinkage over-regularises).

4. Write docs/independence-results.md:
   - the correlation matrices
   - N_eff by lead and region (and with NEPS-G at 24 h)
   - the Shapley tables, including the plain explanation of HRES's zero
     weights
   - OLS vs shrinkage with CIs
   - the H6 verdict
   Add the one-sentence finding for slides, e.g. "GraphCast, HRES and
   IFS-ENS carry about <N_eff> independent opinions; NEPS-G raises it to
   <x> at day 1". Use only the measured numbers. README paragraph.

5. API only (UI is step 16): add read-only endpoints to dashboard/api.py
   (/api/independence, /api/contributions), reading the new CSVs via pure
   functions added to dashboard/data_loading.py, with tests in
   tests/test_dashboard_api.py covering valid output and 422 on bad params.
```

## Done when

- `weavr.independence` has tested Ledoit–Wolf, minimum-variance weights,
  the shrinkage region fit and Shapley attribution.
- The LOYO comparison and Shapley/N_eff tables are committed.
- H6 has a verdict, acted on as stated.
- The API serves the data for step 16's views.
