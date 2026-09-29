# Step 22 (optional): Real AI ensembles — GenCast and FuXi

**Type**: data + evaluation prompt (long fetch).
**Plan items**: B2.
**Depends on**: step 05 (daily 2020 builders) and step 07 (the v2
evaluation pipeline). It is 2020-only, because neither archive covers
2018.

## Goal

WEAVR's only "AI ensemble" today is a lagged GraphCast pseudo-ensemble with
a spread-skill ratio of **4.0–6.9** against a calibrated target of about
1.05 (plan §2 F7), so it is severely under-dispersive. The plan's F6 found
in WeatherBench 2, at 0.25° with 24 h rainfall:

- **GenCast** (`gencast/2020-1440x721.zarr`): a real **56-member** AI
  ensemble (`sample` dimension), 2020-01-01 → 2020-12-31, leads
  12–360 h (Price et al. 2024, Nature).
- **FuXi** (`fuxi/2020-1440x721.zarr`): deterministic, with variable
  `total_precipitation_24hr_from_6hr`, 2020-01-01 → 2020-12-16, leads
  6–360 h (Chen et al. 2023).

Add both for JJAS 2020, then answer two research-brief open questions over
India:

- Is a real AI ensemble better calibrated than a lagged one, and than
  IFS-ENS?
- Do ERA5-trained AI models add independent information to each other, or
  mostly repeat it? (N_eff, using step 09's tools.)

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Probe both archives:
   - dimension and coordinate names (latitude/longitude vs lat/lon;
     GenCast's `sample` must become `member`), units (metres?), latitude
     direction, lead dtype, and chunk shape
   - how 24 h accumulation is defined (GenCast's total_precipitation_24hr;
     FuXi's "_from_6hr" name suggests a derived sum; check its window
     against IMD's 03-03 UTC day)
   Time one chunk fetch for each. Extrapolate the daily JJAS cost (5
   leads). If GenCast's full 56 members exceed ~12 h even with workers,
   route via AskUserQuestion: all 56 / a fixed subset (e.g. the first 20;
   state the CRPS bias from fewer members) / 2-day cadence. Include the
   measured numbers.

2. Extend the step-06 per-(source, year) archive map, and build
   data/gencast_2020_jjas_daily.zarr (member dimension) and add fuxi to a
   2020 daily store (or its own store; follow step 06's layout decision).
   Validate them like step 05 did.

3. Add scripts/run_ai_ensemble_comparison.py (2020 only, so
   seasonal_block_split, labelled as such; state that LOYO isn't possible
   for these sources):
   - Spread-skill ratio vs calibrated_spread_skill_ratio(n_members), CRPS,
     Brier at IMD thresholds, and PIT for: GenCast, IFS-ENS and lagged
     GraphCast.
   - EMOS-CSG on GenCast (reuse weavr.emos) vs raw GenCast.
   - BMA with {graphcast lagged, ifs_ens, gencast, hres, fuxi} components
     vs the v2 BMA.
   - Independence: error correlations and N_eff for {graphcast, fuxi,
     gencast mean, hres, ifs_ens_mean}, plus step 09's Shapley
     attribution with 5 sources (31 subsets).
   Write results/ai_ensembles_2020.csv and per-day outputs, with CIs.

4. docs/ai-ensembles-results.md: plain answers to the two questions above,
   including a finding that the AI models are highly mutually correlated
   if that's what the data shows. README paragraph.

5. Dashboard: extend the independence view's data (step 16) with the
   5-source 2020 result as a separate, labelled selection (2020-only).
   Browser-verify.
```

## Done when

- GenCast and FuXi JJAS 2020 daily stores exist and are validated.
- The calibration comparison, 5-source independence analysis and BMA
  extension are reported with CIs.
- The independence view can show the 5-source 2020 case.
