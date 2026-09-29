# Step 11: Tail repair, part 1 — quantile-map each source before blending

**Type**: method + evaluation prompt.
**Plan items**: B3 part 1.
**Depends on**: step 04 (tw-CRPS, SEDI, CIs) and step 07 (v2 data, LOYO).

## Goal

AI models smooth heavy rain away. Research brief §3.4 says GraphCast, AIFS
and GenCast under-predict rain above 50 mm/day over India. The plan's F7
shows WEAVR's POD at ≥115.6 mm is about 0 beyond day 1.

Add **per-source, per-region quantile mapping (QM)** that maps each source's
forecast distribution onto IMD's observed distribution before blending or
EMOS. Then test whether it restores heavy-rain detection without hurting
everything else. This is pre-registered H7, part 1.

## Why QM, and why this way

- HEPPI's authors found QM better at correcting the **shape** of the
  precipitation distribution, and EMOS better at reliability and heavy rain
  (research brief, Update 26 Sep (3)). So test QM *feeding* EMOS and the
  deterministic blend, not replacing EMOS.
- `HEPPI/Generalized_QM.m` implements several QM variants: empirical,
  gamma, GEV, double-gamma, and "GEVG" (GEV above the 90th percentile,
  gamma below). HEPPI's `NCMRWF_UQM_forecast.nc` is a real QM output on this
  grid. Use both as the method reference and as a validation target.
- There is a real methodological choice. The observed CDF can come from
  the **same period** as the forecast train data (the standard, matched
  approach). Or it can be anchored on the **15-year IMD JJAS climatology**
  (far more tail data, but it assumes the train season is typical).
  Evaluate both as variants and let the pre-registered criterion decide;
  don't pick in advance.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Read HEPPI/Generalized_QM.m and record which variants it implements and
   how it handles dry days and values beyond the training range. Put a
   short summary in the new doc.

2. Add src/weavr/quantile_mapping.py (pure, tested):
   - fit_quantile_map(forecast_train, obs_reference, quantiles,
     wet_threshold=0.1) -> QuantileMap. It must handle:
     - wet-day frequency: map the forecast's dry fraction to the observed
       dry fraction, so drizzle isn't inflated
     - the upper tail beyond the top fitted quantile: document the rule
       (e.g. constant additive correction or a scaled ratio), since this
       choice matters most for extremes
   - apply_quantile_map(qm, forecast) -> corrected forecast, never
     negative.
   - Per (source, region, lead) fitting helper, reusing weavr.regions.
   - Variant switch: reference="same_period" (train-fold IMD obs, same
     region) or "climatology" (IMD JJAS 2006-2020 archive EXCLUDING the
     test year).
   - For ensembles, apply the same map to every member.
   Tests: identity when forecast and obs distributions match; a known
   shifted gamma is recovered; dry-fraction matching; monotonicity; no
   negatives.

3. Validation against HEPPI (directional, not a literal reproduction):
   fit this module's empirical QM on NEPS-G raw 2018 -> apply to 2019, and
   compare quantiles of the result against HEPPI-UQM's 2019 output in the
   same regions. Report the agreement plainly. HEPPI's exact variant and
   training setup may differ; say what you found in Generalized_QM.m.

4. Add scripts/run_tail_repair_qm.py (LOYO, both folds, pooled):
   For each source and each variant, evaluate:
   a. raw vs QM-corrected single sources
   b. the Tier 1 regional blend built from QM-corrected sources (weights
      refit on corrected train data)
   c. EMOS-CSG (ifs_ens and graphcast lagged) fit on QM-corrected members
   Metrics: tw-CRPS at t = 64.5 mm, SEDI and POD/FAR/CSI at
   64.5/115.6/204.5 mm, Brier at 7.5 mm, RMSE, CRPS. CIs via step 04.
   Write results/tail_repair_qm.csv and per-day outputs.

5. Compute H7 (QM part) per docs/preregistration.md. It passes only if
   tw-CRPS or SEDI@115.6 improves at >= 3/5 leads with the CI excluding 0,
   AND Brier@7.5 does not get significantly worse. Report which variant
   (same_period / climatology) did what.

6. Write docs/tail-repair-results.md (section 1: QM): the method, the HEPPI
   comparison, the results table and the H7 part-1 verdict, in plain
   language. If QM hurts RMSE (inflating intensity usually does), state
   the trade-off explicitly. README paragraph.
```

## Done when

- `weavr.quantile_mapping` is tested.
- Both reference variants are evaluated under LOYO with CIs.
- The HEPPI-UQM directional check is reported.
- The H7 QM verdict is recorded.
