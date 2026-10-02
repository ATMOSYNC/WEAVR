# Archive: v1 weekly single-season results (2020 JJAS)

These are the original baseline evaluation results produced from the single
weekly 2020 JJAS store (`data/baseline_2020_jjas.zarr`), using
`seasonal_block_split` (18 samples total, 14 train / 4 test). They are kept
here as a labelled historical record; the live `results/` directory now holds
v2 results from the daily two-season stores.

## Source commit

The v1 results were last committed / updated across multiple PRs; the tier
scores (e.g. `tier0_baseline.csv`) were introduced in commit `0e1ccd9`
("Run the Tier 0 equal-weight baseline", PR #17) and subsequently updated
through the Phase 3–5 PRs on the `ATMOSYNC/WEAVR` main branch.

## What these files are

- `*.csv` — aggregate per-lead tier metrics (RMSE, CRPS, SEEPS, FSS, POD, FAR,
  CSI, ETS) for Tier 0 through Tier 3, phase 2 ensemble, and single sources.
- `per_day/` — per-day per-cell scores used for significance testing.
- `tier1_regional_weights.csv` — Tier 1 regional blend weights fitted on the
  2020 weekly store.
- `preregistration_verdicts.csv`, `scorecard.csv` — v1 H1–H3 verdicts from
  the single-season evidence base.

## Evaluation setup

| Setting | Value |
|---|---|
| Store | `data/baseline_2020_jjas.zarr` (weekly, gitignored) |
| Season | 2020 JJAS (Jun–Sep) |
| Split | `seasonal_block_split` (18 samples: 14 train / 4 test) |
| Leads | 24, 48, 72, 96, 120 h |
| Grid | 129 × 135 (0.25° India, 6.5–38.5°N, 66.5–100°E) |

## Superseded by

`results/` (v2) — re-run on the daily 2020 store with LOYO cross-validation
via `weavr.splits.iter_evaluation_folds`. See `docs/v2-evidence-base-results.md`.
