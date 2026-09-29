# Step 01: Single-source baselines, independence diagnostic and claims audit

**Type**: measurement + docs prompt. Hand this to Claude Code or an engineer.
**Plan items**: A5, plus the plan's §2 F2, F3 and F8 findings made permanent.
**Depends on**: nothing. Do this first.

## Goal

Two uncomfortable findings in the plan's §2 currently exist only in a
planning doc. This step turns them into reproducible, committed parts of
WEAVR's results:

- **F2:** raw GraphCast beats both Tier 0 and Tier 1 on RMSE at 24–96 h, on
  the same test days.
- **F3:** the three rainfall sources' errors correlate at 0.71–0.93, so the
  effective number of models is about 1.1. This is also why HRES gets zero
  weight in 25 of 30 Tier 1 cells.

Then make every slide in `ppt/content/` say only what the repo proves today.

## Why this goes first

- A judge can compute F2 in five minutes with the repo's own functions. It
  is far better for WEAVR to publish it first, with the RMSE-favours-smooth
  caveat, than to be caught by it.
- Every later step compares against "the best single member, chosen on
  train data". That baseline has to exist as a committed results file.
- The slide over-claims (plan §2 F8) take minutes to fix and are expensive to
  be caught on.
- `docs/baseline-store.md` says "only the second window overlaps a full
  monsoon season". F5 found that is wrong. Correcting it now stops later
  steps from re-deriving it.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Add scripts/run_single_source_baseline.py.
   - For each lead in [24, 48, 72, 96, 120] and each rainfall source in
     data/baseline_2020_jjas.zarr (graphcast, hres, ifs_ens_mean), score the
     raw source ALONE, on exactly the split the tier scripts use today
     (seasonal_block_split, test_fraction=0.2).
   - Pangu has no rainfall variable. Say so; do not zero-fill it.
   - Reuse run_tier0_baseline's _align_to_imd_day and score_lead, so
     RMSE/bias/ACC/SEEPS/FSS/POD/FAR/CSI/ETS all come from the same code
     path. Don't write a second scoring path.
   - Write results/single_source_baseline.csv.
   - Also record, per lead, "best_single_member_on_train": the source with
     the lowest RMSE on the TRAIN split (never chosen on test), and that
     source's TEST scores.
   - Expected test RMSEs, from the plan's §2 F2 (verify, don't copy):
     graphcast 11.49 / 12.45 / 11.80 / 12.71 / 14.83;
     ifs_ens_mean 11.76 / 13.78 / 13.85 / 14.48 / 15.38;
     hres 13.50 / 16.42 / 17.97 / 19.29 / 17.65.
     If the re-run differs, report the real numbers and explain why.
   - Add a test that runs the script's pure pieces on a tiny synthetic store.

2. Add src/weavr/independence.py with pure functions:
   - error_correlation_matrix(errors: dict[str, xr.DataArray], dims) ->
     (sources, matrix): pairwise Pearson correlation of forecast-minus-obs
     errors, over cells finite in every source.
   - effective_number_of_models(corr) = N / (1 + (N-1) * mean_off_diagonal)
   - participation_ratio(corr) = (sum eigenvalues)^2 / sum(eigenvalues^2)
   Unit tests: independent synthetic errors give ~N; identical errors give
   1; a known 2-source correlation gives the closed-form value.

3. Add scripts/run_independence_diagnostic.py, writing
   results/independence_diagnostic.csv with the pairwise correlations, N_eff
   and participation ratio for:
   - each lead, domain-wide
   - each lead x region (weavr.regions.assign_regions)
   Compute on the TRAIN split only.
   WeatherBench 2 precipitation is in METRES: multiply by PRECIP_M_TO_MM
   first. Forgetting this gives a spurious correlation of exactly 1.0.
   Expected domain-wide N_eff is about 1.07-1.15 (plan §2 F3; that figure
   used all 18 samples, so the train-only numbers will differ slightly).

4. Write docs/single-source-and-independence-results.md. It must include:
   - both tables
   - a plain statement that no blend beats raw GraphCast on RMSE at
     24-96 h on today's 3-4-day test sets, and that Tier 2's single-source
     EMOS-on-GraphCast is the strongest RMSE result
   - the research brief's section 1.3 caveat, that RMSE favours smooth,
     MSE-trained AI output, so blends must also be judged on CRPS,
     heavy-rain POD/ETS and reliability
   - that HRES's zero weight in 25 of 30 Tier 1 cells is collinearity with
     IFS-ENS (quote the correlations), not evidence that HRES is useless
   Link it from README.md in the existing style.

5. Correct docs/baseline-store.md's "Only the second window overlaps a full
   monsoon season".
   - Re-check live that
     gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours_derived.zarr
     has total_precipitation_24hr covering 2018-06-01..2018-09-30.
   - Note that it uses lat/lon dimension names, not latitude/longitude.
   - Add a dated correction note stating what was checked. Build nothing
     for 2018 here; step 06 does that.

6. Outside the repo, edit ppt/content/slide1.md..slide4.md so every claim
   is true TODAY. Use the plan's section 2 F8 table:
   - NEPS-G/NCUM are not claimed as ingested. "Planned: NEPS-G via the
     HEPPI dataset" is fine.
   - Remove "provably better than any single source".
   - Remove "strongest combiner per bin" (step 13 may make it true).
   - Reword the heavy-bin claim so it doesn't assert unmeasured gains.
   - Mark the "70% chance of >115.6 mm in this district" example as
     illustrative.
   - Remove Chennai, or label it as a north-east-monsoon event outside
     today's June-September scope.
   Keep the slides' structure. Put a before/after list of every changed
   claim in the PR description.

7. Run ruff, mypy and pytest. Open one PR for items 1-5.
```

## Done when

- `results/single_source_baseline.csv` and
  `results/independence_diagnostic.csv` exist, produced by committed,
  tested scripts.
- The new doc and README state F2 and F3 plainly.
- `docs/baseline-store.md` is corrected.
- No slide in `ppt/content/` contains a claim the repo contradicts.
