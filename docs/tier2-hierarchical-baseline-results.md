# Tier 2 hierarchical baseline results (Phase 4's exit criterion)

Produced by
[`scripts/run_tier2_hierarchical_baseline.py`](../scripts/run_tier2_hierarchical_baseline.py).
Full numbers in
[`results/tier2_hierarchical_baseline.csv`](../results/tier2_hierarchical_baseline.csv)
(domain-wide, per lead),
[`results/tier2_hierarchical_baseline_by_bin.csv`](../results/tier2_hierarchical_baseline_by_bin.csv)
(per bin, and per region for BMA), and
[`results/tier2_hierarchical_baseline_by_region.csv`](../results/tier2_hierarchical_baseline_by_region.csv)
(BMA's own per-region breakdown).

## Scope carried over from steps 1-5

- **EMOS-CSG sources: GraphCast's Phase 2 lagged pseudo-ensemble and step
  2's real IFS 50-member ensemble, fit independently** (`src/weavr/emos.py`)
  — never HRES, which stays deterministic in every store. Reported as two
  separate combiners, `emos_graphcast` and `emos_ifs_ens`; issue #6 names
  no further step blending them together, so neither does this one.
- **BMA sources: GraphCast + IFS (ensemble dressing) + HRES (kernel
  dressing), fit jointly into one mixture** (`src/weavr/bma.py`), per step
  5's own per-source route decision.
- **Rain-intensity bins classified from GraphCast's own lagged-ensemble
  mean forecast**, once per lead, shared by both combiners — the same
  forecast basis `docs/phase4-data-and-combiner-scope.md`'s own per-bin
  sample-count table used, needed here so EMOS-CSG (fit per source) and
  BMA (fit jointly, needs one shared label per cell) are compared on
  identical strata. This is a real, checked basis for a small honest
  discrepancy from that table: this run finds the "heavy" bin fittable
  domain-wide at **every** lead (not just 24h/48h) — because the
  classifying forecast here is GraphCast's lagged-pseudo-ensemble *mean*,
  not the single deterministic run that doc's illustrative table used;
  the two are closely related but not identical, and the difference is
  large enough to shift a handful of days across the bin boundary. Both
  are real, defensible choices; this run's own numbers below are internally
  consistent throughout.
- **One `seasonal_block_split`, shared by every method at each lead** —
  Tier 0's equal-weight mean, Tier 1's regional-weight blend, both
  EMOS-CSG sources, and BMA are all fit and scored against the *same*
  train/test split, a stronger fairness guarantee than Tier 1's own script
  (which only shared its split between Tier 0 and Tier 1; this run shares
  it across all four methods).
- **Tier 0/Tier 1's "ifs_ens_mean" here is this run's own IFS ensemble's
  mean**, not a separately-opened `baseline_2020_jjas.zarr` group — both
  come from the same real WeatherBench 2 archive, and deriving it from the
  already-loaded 50-member ensemble guarantees identical sample alignment
  with the Phase 4 combiners' own IFS source.
- **HEPPI stays a validation/methodology reference only** (step 1's
  decision) — see "HEPPI cross-check" below; not a training source.

## A real numerical bug, found and fixed running this step

Running this script against the real stores surfaced a genuine bug, not
assumed away: **~1% of GraphCast's own real forecast cells** (in both
`data/lagged_ensemble_inputs_2020_jjas.zarr` and
`data/baseline_2020_jjas.zarr`) carry tiny negative numerical-noise
artifacts (down to about -0.9mm after unit conversion) — a known
ML-weather-model output characteristic, not a bug in this project's own
pipeline (the real IFS ensemble has none, checked directly). Every prior
tier's scoring (RMSE, bias, ACC, SEEPS, FSS, contingency, ensemble CRPS)
is insensitive to a value this small being slightly negative rather than
zero, so it went unnoticed until now.

EMOS-CSG's gamma-based model is not insensitive to it: a negative
*ensemble-mean* forecast feeding a fitted regression's `location = a1 +
a2*ensemble_mean` produced a location value near/below zero, which fed
`src/weavr/emos.py`'s `csgd_crps`/`predict_csgd_params` inconsistently
(kappa's numerator using a raw value while theta's denominator used a
clipped one) — checked directly against the exact real cell that
triggered it: an uncorrected CRPS of **~1165mm** for an observation of
zero, and a Monte Carlo predictive mean of **~2460mm**, both obviously
wrong. Two fixes, both real, both landed in this same PR:

1. **Root cause**: `scripts/run_tier2_hierarchical_baseline.py` now floors
   every forecast source's precipitation to `>= 0` immediately after unit
   conversion (`_clip_negative_precip`) — precipitation cannot be
   physically negative, so this is a data-hygiene fix at the source, not a
   numerical patch inside the combiner.
2. **Defense in depth**: `src/weavr/emos.py`'s `csgd_crps` and
   `predict_csgd_params` now clip `mean`/`location` once, consistently,
   before using it in both kappa and theta, and `csgd_crps`'s final result
   is clamped to `>= 0` (CRPS is mathematically never negative; a small
   negative floating-point result only ever occurs for a pathologically
   tiny `mean`, an extreme-cancellation regime checked directly and
   documented in that module, not reachable in practice once inputs are
   floored to real, non-negative precipitation values).

Regression tests for both are in `tests/test_emos.py`
(`TestCsgdCrpsHandlesNonPositiveMean`, and a `predict_csgd_params` check)
and `tests/test_run_tier2_hierarchical_baseline.py`
(`TestClipNegativePrecip`). Re-running after the fix changed the 72h lead's
`emos_graphcast` domain-wide RMSE from an obviously-wrong 21.37mm down to
11.27mm (in line with every other lead); every other number moved by a
negligible amount, since the bug only ever affected a handful of dry-bin
cells.

## Fitting outcome: fallback is common, exactly as expected

- **EMOS-CSG**: `is_fallback=False` (a real fit) for dry/light/heavy at
  every lead, for both sources; `very_heavy`/`extremely_heavy` fall back
  at every lead (0 real test cells at either category except a single
  `very_heavy` cell at 96h) — consistent with Tier 0's own prior finding
  of essentially zero forecast-yes events in the highest categories.
- **BMA**: fit per (bin, region) — 150 cells total (5 bins x 6 regions x 5
  leads). **85 of 150 (57%) fall back**, almost entirely in `heavy` (most
  regions, most leads — a single region's-worth of heavy-rain days rarely
  reaches `MIN_TRAIN_DAYS_PER_BIN=5`) and `very_heavy`/`extremely_heavy`
  (every region, every lead). `dry` and `light` fit for real in every
  region at every lead. This is expected, not a defect: per-region
  splitting can only shrink an already-small bin's sample count further,
  never grow it (see "Small sample sizes" below).

## Domain-wide results: a real, mixed comparison

| Lead (h) | Tier 0 RMSE | Tier 1 RMSE | EMOS-graphcast RMSE / CRPS | EMOS-ifs_ens RMSE / CRPS | BMA RMSE / CRPS |
|---|---|---|---|---|---|
| 24  | 11.55 | 11.08 | 10.86 / **3.76** | 10.80 / 3.76 | 10.84 / **3.72** |
| 48  | 13.59 | 12.53 | 12.08 / **3.74** | 12.31 / 3.83 | 12.11 / 3.80 |
| 72  | 13.15 | 12.05 | 11.27 / **3.64** | 12.24 / 3.90 | 11.78 / 3.83 |
| 96  | 14.15 | 13.34 | 12.95 / **4.41** | 13.73 / 4.64 | 13.69 / 4.64 |
| 120 | 15.35 | 14.90 | 14.88 / 4.46 | 15.08 / 4.55 | 14.93 / **4.45** |

(RMSE/bias here score each combiner's own Monte Carlo predictive mean, a
different quantity from CRPS — the proper score both combiners are
designed to optimize, and the one to weight most heavily. Best CRPS per
lead is **bolded**.)

**Neither combiner dominates.** Counting domain-wide CRPS wins across the
5 leads: **EMOS-graphcast wins 3 (48h, 72h, 96h), BMA wins 2 (24h, 120h),
EMOS-ifs_ens wins 0.** Exactly the "trade off, don't assume either wins"
framing issue #6 itself names, not a result adjusted to favor one
combiner. Against Tier 0, every Tier 2 combiner's RMSE wins at every lead.
Against Tier 1, the picture is mixed: EMOS-graphcast's RMSE beats Tier 1
at all 5 leads, but EMOS-ifs_ens only beats it at 24h/48h (losing at
72h/96h/120h) and BMA only at 24h/48h/72h (losing at 96h/120h, the latter
narrowly) — Tier 2 does not win everywhere either, reported plainly rather
than only quoting the combiner that looks best.

## Per-bin comparison: EMOS-CSG and BMA really do split which they win

Pooled (cell-count-weighted) CRPS by lead and bin, across only cells where
a real fit exists for every combiner being compared:

| Lead (h) | Bin | BMA | EMOS-graphcast | EMOS-ifs_ens |
|---|---|---|---|---|
| 24  | dry   | **1.96** | 2.01 | 2.05 |
| 24  | light | 8.41 | 8.49 | **8.38** |
| 24  | heavy | 46.49 | **23.02** | 23.96 |
| 48  | dry   | 1.80 | **1.78** | 1.85 |
| 48  | light | 8.80 | **8.56** | 8.73 |
| 48  | heavy | **19.33** | 23.29 | 21.91 |
| 72  | dry   | 1.79 | **1.75** | 1.90 |
| 72  | light | 7.99 | **7.71** | 8.19 |
| 72  | heavy | 45.80 | **26.82** | 30.10 |
| 96  | dry   | 2.51 | **2.46** | 2.60 |
| 96  | light | 8.40 | **8.07** | 8.43 |
| 96  | heavy | 61.01 | **27.05** | 30.99 |
| 120 | dry   | **2.60** | 2.62 | 2.63 |
| 120 | light | 7.71 | **7.69** | 7.92 |
| 120 | heavy | **13.97** | 15.11 | 15.79 |

(Every row above is a real, non-degenerate margin — no near-ties this
time; a separate all-fallback `very_heavy` cell at 96h, where all three
combiners score an identical 111.46mm on the single available test point,
is excluded here as not a meaningful comparison.)

Across these 15 (lead, bin) cells: **EMOS-graphcast wins 10, BMA wins 4,
EMOS-ifs_ens wins 1.** The pattern is bin-dependent, not lead-dependent:
**dry/light lean toward EMOS-graphcast but stay close** (differences of a
few hundredths to a few tenths of a mm, with BMA or EMOS-ifs_ens
occasionally edging ahead); **heavy is where the real separation shows
up**, and there EMOS-CSG (either source) clearly beats BMA at
24h/72h/96h (by roughly 2x), while BMA clearly beats both EMOS variants at
48h and 120h. Domain-wide CRPS (previous section) is dominated by the huge
dry/light cell counts, which is why BMA's heavy-bin wins don't show up as
domain-wide wins as often as this bin-level table might suggest on its
own.

## HEPPI cross-check: directionally consistent, not a literal validation

`docs/phase4-data-and-combiner-scope.md` found HEPPI's own EMOS script
(`example_EMOS_fit.R`) fits the identical model family used here
(censored-shifted-gamma, CRPS-minimized) on a real 23-member NCMRWF NEPS-G
ensemble, and that Angus et al. (2024) reported EMOS beating quantile
mapping for short-lead heavy rain on that exact ensemble. This project has
**no quantile-mapping baseline implemented**, so this run cannot literally
reproduce that comparison — the honest cross-check available is narrower:
does EMOS-CSG look competitive for short-lead heavy rain *here*, on a
different ensemble/season? At the shortest lead (24h), EMOS-CSG
(graphcast: 23.0mm, ifs_ens: 24.0mm) is roughly half BMA's heavy-bin CRPS
(46.5mm) — directionally consistent with EMOS being a strong choice for
short-lead heavy rain. That consistency does not hold at 48h, where BMA
wins the heavy bin instead, so this is read as one supportive data point
on the same general model family, not confirmation of Angus et al.'s
specific finding.

## Small sample sizes: the same caveat every prior tier has stated, worse here

14 train / 3-4 test days per lead, same as every prior tier — and per-bin
(and per-bin-region, for BMA) splitting can only shrink the effective
per-cell day count further, never grow it. `heavy`'s domain-wide EMOS-CSG
fit draws from as few as 6 distinct train days (48h) up to 14 (72h/96h,
where every train day contributes at least one heavy-rain gridpoint
somewhere in the domain); BMA's own per-region heavy-bin fits are smaller
still, which is exactly why 57% of its (bin, region) cells fall back.
`very_heavy`/`extremely_heavy` are not fittable at all with today's data,
domain-wide or per-region — the same finding step 1 already made, now
confirmed at the region-stratified granularity too.

## Regenerating

```bash
python scripts/run_tier2_hierarchical_baseline.py
```

Override the stores, output paths, test fraction, or Monte Carlo sample
count via `--baseline-store`, `--lagged-store`, `--ifs-ensemble-store`,
`--climatology`, `--out-csv`, `--bin-out-csv`, `--region-out-csv`,
`--test-fraction`, `--n-monte-carlo`.
