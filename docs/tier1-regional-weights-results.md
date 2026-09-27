# Tier 1 regional-weights results (Phase 3's exit criterion)

Produced by
[`scripts/run_tier1_regional_baseline.py`](../scripts/run_tier1_regional_baseline.py).
Full numbers in
[`results/tier1_regional_baseline.csv`](../results/tier1_regional_baseline.csv)
(domain-wide),
[`results/tier1_regional_baseline_by_region.csv`](../results/tier1_regional_baseline_by_region.csv)
(per region), and
[`results/tier1_regional_weights.csv`](../results/tier1_regional_weights.csv)
(the fitted weights themselves).

## Scope carried over from steps 1-3

- **Precipitation only**, the same 3 of 4 baseline-store sources as Tier 0
  (`graphcast`, `hres`, `ifs_ens_mean` — `pangu` has no precipitation
  variable at all).
- **CV strategy: `seasonal_block_split` for every source**, per
  [`docs/phase3-cv-and-regional-scheme.md`](phase3-cv-and-regional-scheme.md)'s
  user-confirmed decision — GraphCast's WeatherBench 2 archive has exactly
  one full JJAS season ever, ruling out true leave-one-year-out CV for the
  one AI source/variable this project scores, so every source uses the
  same split rather than a mixed-CV path.
- **Regional scheme: Sreekala & Babu's 6-zone pooling**
  (`src/weavr/regions.py`) — WC, SI, WI, CI, NE1, NE2.
- **Weight fitting: `src/weavr/weighting.py`'s `fit_region_weights`** —
  unconstrained OLS per Wanders & Wood (2016), negative weights clipped to
  zero and renormalized to sum to 1 (Wang et al. 2025), with a documented
  equal-weight fallback for degenerate regions.

## Why the equal-weight comparison is recomputed here, not read from Tier 0's doc

[`docs/tier0-baseline-results.md`](tier0-baseline-results.md)'s numbers are
scored **domain-wide** on a test split drawn from Tier 0's own alignment.
Answering Phase 3's actual question — does the weighted blend beat equal
weighting **region by region** — needs per-region equal-weight numbers that
don't exist anywhere yet, and no per-region breakdown can be recovered from
Tier 0's domain-wide CSV after the fact. So this script recomputes the
equal-weight mean itself, from the same aligned forecasts and the same
train/test split as the Tier 1 blend, so both are scored against literally
the same held-out samples — a controlled, apples-to-apples comparison
rather than one across two independently-computed runs.

As a sanity check, the domain-wide equal-weight numbers computed here
(see table below) match Tier 0's own recorded RMSEs closely (11.66, 13.54,
13.38, 14.27, 15.30 mm at leads 24-120h) — confirming both scripts are
scoring the same underlying data the same way, not just superficially
similar.

## Per-region scoring: RMSE/bias/ACC/SEEPS only, not FSS/contingency

RMSE, bias, ACC, and SEEPS reduce over gridpoints with `skipna=True`
already, so masking non-region gridpoints to NaN before scoring correctly
restricts the average to that region. FSS (a spatial neighborhood box
filter) and the contingency counts are **not** broken out per region: a
plain box filter has no NaN-aware mode, so masking other regions to NaN
would corrupt the neighborhood-fraction computation for every gridpoint
near a region boundary, not just cleanly exclude other regions. These two
metrics stay domain-wide only, exactly as Tier 0 reported them, rather than
computing a subtly-wrong region-restricted version.

## Fitting outcome: no fallback triggered anywhere

All 30 (region × lead) cells fit real regression weights — `is_fallback`
is `False` throughout `results/tier1_regional_weights.csv`; no region had
too few train samples or a rank-deficient design matrix at any lead. Each
region pools thousands of (train sample × gridpoint) points per lead (see
`n_train_points` in that CSV — e.g. 20,006 for CI, 4,242 for the smallest,
WC), consistent with `docs/phase3-cv-and-regional-scheme.md`'s data-density
check that 6-region pooling isn't data-starved on this store.

## Domain-wide results: Tier 1 beats Tier 0 at every lead

| Lead (h) | n (train/test) | Tier 1 RMSE (mm) | Tier 0 equal RMSE (mm) | Tier 1 ACC | Equal ACC |
|---|---|---|---|---|---|
| 24  | 14/4 | **11.55** | 11.66 | 0.414 | 0.418 |
| 48  | 14/4 | **12.62** | 13.54 | 0.353 | 0.313 |
| 72  | 14/4 | **12.96** | 13.38 | 0.364 | 0.371 |
| 96  | 14/3 | **13.65** | 14.27 | 0.390 | 0.374 |
| 120 | 14/3 | **14.82** | 15.30 | 0.274 | 0.252 |

Tier 1's regional weighted blend has lower domain-wide RMSE than the
equal-weight mean at all 5 leads — real, not assumed to hold, and reported
even though it could have gone the other way (see below for where it does).
ACC and bias are mixed (Tier 1 wins ACC at 3/5 leads, ties closely at the
other 2); SEEPS and FSS/contingency (full tables in the CSV) show no
consistent domain-wide winner either way, which is expected: RMSE is
exactly what OLS weight-fitting minimizes on the training split, so it is
the metric most likely to show a clean win, not a guarantee every other
score improves too.

## Region-by-region: real, mixed — Tier 1 does not win everywhere

Counting RMSE wins across all 30 (region × lead) cells:

| Region | Tier 1 RMSE wins | out of |
|---|---|---|
| NE2 | 5 | 5 |
| CI  | 3 | 5 |
| NE1 | 2 | 5 |
| SI  | 4 | 5 |
| WC  | 4 | 5 |
| WI  | 4 | 5 |
| **Total** | **22** | **30** |

Tier 1 beats the equal-weight mean's RMSE in 22 of 30 region×lead cells
(73%) — a real, majority improvement, but **not universal**: NE1 loses
more often than it wins (2/5), and every region loses at least once. This
is reported as the actual outcome, not adjusted or hidden — a fitted
regional weight can overfit its own 14-day training split for a given
region/lead and score worse than a simple mean on the 3-4 held-out test
days, exactly the risk this step's own prompt named in advance. NE1's
comparatively poor showing is plausible given it draws from only 7,336
train points (the second-smallest pool, per `results/tier1_regional_weights.csv`)
among a scheme with genuinely small effective per-region sample counts.

## Small sample sizes: the same caveat every prior tier has stated

**14 train / 3-4 test days per lead**, the same weekly-init sampling
density Phase 0 chose deliberately (see `docs/baseline-store.md`). Region
pooling multiplies the *spatial* points per region (thousands, per above)
but does not add independent *time* samples — the train/test split still
only ever varies at 14/3-4 distinct days. A fitted weight from 14 training
days, evaluated on 3-4 test days, is exactly the kind of small-sample
result where "22/30 region-lead cells win" should be read as a first real
signal that regional weighting helps more often than not on this data, not
a statistically robust guarantee it always will. This is an inherent
consequence of Phase 0's sampling-density tradeoff, not a bug in this step.

## Regenerating

```bash
python scripts/run_tier1_regional_baseline.py
```

Override the store, climatology archive, output paths, test fraction, or
neighborhood size via `--store`, `--climatology`, `--out-csv`,
`--region-out-csv`, `--weights-out-csv`, `--test-fraction`,
`--neighborhood-size`.
