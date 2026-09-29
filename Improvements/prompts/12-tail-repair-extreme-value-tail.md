# Step 12: Tail repair, part 2 — an extreme-value tail for the bins that can't be fit

**Type**: method + evaluation + dashboard-update prompt.
**Plan items**: B3 part 2.
**Depends on**: step 11 (use its best QM variant if H7-QM passed; otherwise
build on raw inputs and say so).

## Goal

Today, cells whose forecast falls in an unfittable rain bin get a point-mass
fallback. That shows as a grey "not a real probability" overlay on the
extreme-probability map. Even with v2 data, `extremely_heavy` (≥204.5 mm)
is probably still thin; step 07 re-measured this, so use its real counts.

Replace the fallback with a **pooled extreme-value tail** anchored on
15 years of IMD extremes:

- `P(Y > u | forecast)` comes from the nearest *fittable* EMOS-CSG bin.
- The excess above `u` follows a Generalized Pareto distribution (GPD) with
  a shared shape and region-specific scales.
- Spliced: `P(Y > y) = P_CSGD(Y > u) · (1 + ξ (y − u)/σ_r)^(−1/ξ)` for
  `y > u`.

This gives a real, flagged `P(≥115.6)` and `P(≥204.5)` where today there is
none. Evaluate it under H7 part 2.

## Why this is sound, and its limits

- Extreme-value theory is the standard statistical tool for tails with few
  samples. HEPPI's own QM script has a "GEVG" option (GEV above the 90th
  percentile), so this lineage is already present in NEPS-G work.
- The 15-year IMD JJAS archive (1,830 days × about 18k land cells) has far
  more exceedances of 115.6 and 204.5 mm than any forecast training set.
  Pooling across regions (shared shape ξ) is what makes the fit stable.
- Honest limit: the GPD describes the *climatological* excess distribution.
  Conditioning it on the forecast happens only through `P_CSGD(Y > u)`.
  Say this in the view caption.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Add src/weavr/tail.py (pure, tested):
   - fit_pooled_gpd(exceedances_by_region, threshold_u) -> TailFit: one
     shared shape xi, and one scale per region (weavr.regions zones),
     fitted by maximum likelihood (scipy.optimize) on IMD archive
     exceedances above u, EXCLUDING the test year. Report standard errors
     (or a bootstrap CI) for xi.
   - Choose u by a documented rule, checked with a mean-residual-life /
     parameter-stability plot saved under docs/figures/: e.g. u = 64.5 mm
     (IMD's heavy threshold) or a high regional quantile.
   - spliced_exceedance_probability(p_exceed_u, tail_fit, region, y):
     continuous at u, monotone, in [0, 1]. Test these properties, and test
     it against Monte Carlo sampling of the spliced distribution.

2. Integrate with EMOS-CSG. Add exceedance_probability_with_tail(...):
   - For cells whose forecast bin is fittable and u lies below that bin's
     useful range, use the CSGD directly.
   - For cells whose bin is unfittable, or for thresholds above u, use the
     nearest fittable bin's CSGD for P(Y > u) and the GPD for the excess.
   - Every cell gets a method flag: "csgd" | "csgd+gpd_tail" |
     "fallback", keeping is_fallback semantics only for truly unfittable
     cases (e.g. no fittable bin at all).

3. Add scripts/run_tail_repair_evt.py (LOYO, pooled). Evaluate
   P(>=115.6) and P(>=204.5):
   - reliability tables with bootstrap CIs (expect wide ones, and say so)
   - SEDI
   - Brier score and Brier skill score vs climatology
   - tw-CRPS at t = 64.5 and 115.6, where computable from the spliced
     CDF (numerical integration, tested)
   Compare against: the current point-mass fallback, the step-11 QM
   variant, and raw IFS-ENS member-counting probabilities.
   Write results/tail_repair_evt.csv and per-day outputs.
   Compute the H7 verdict (EVT part).

4. Dashboard.
   - Extend scripts/export_dashboard_example_grids.py to export
     P(>=115.6) and P(>=204.5) with the per-cell method flag, and extend
     /api/extreme-probability with a threshold parameter
     (115.6 | 204.5, returning 422 on anything else) plus the method
     array. Update tests/test_dashboard_api.py.
   - Update dashboard-web/js/charts/extremeProbability.js: a threshold
     toggle, and a distinct marking for "csgd+gpd_tail" cells (e.g. a thin
     hatch or dotted outline, NOT the grey fallback) with a legend entry.
     Grey stays only for true fallbacks.
   - Update the caption to state the tail method and its limit.
   - Browser-verify on a fresh port across all leads and both thresholds.
     Pixel-sample the canvas (the technique from frontend step 5) to
     confirm the three cell classes render distinctly.

5. docs/tail-repair-results.md, section 2 (EVT): the fit (xi with CI, the
   threshold choice and its diagnostics figure), reliability, the verdict,
   and the honest limit. README paragraph.
```

## Done when

- `weavr.tail` is tested (continuity, monotonicity, a Monte Carlo check).
- The extreme-probability map shows real, flagged tail probabilities where
  it used to show fallback grey, with a threshold toggle.
- The H7 EVT verdict is recorded with reliability evidence.
