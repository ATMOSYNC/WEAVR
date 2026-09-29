# Steps completed so far

Five steps are merged into `main` of `ATMOSYNC/WEAVR`. Prompts for them live
in this folder. Numbers are as measured at merge time; the docs named below
are the source of truth.

| Step | What it delivered | PR | Main artefacts |
|---|---|---|---|
| **01** | Single-source baselines (GraphCast, HRES, IFS-ENS mean alone), independence diagnostic, claims audit (10 over-claims fixed in `ppt/content/slide1–4.md`) | #53 | `docs/single-source-and-independence-results.md`, `src/weavr/independence.py`, `results/single_source_baseline.csv`, `results/independence_diagnostic.csv` |
| **02** | Pre-registration of headline claims H1–H11 (pass rule: ≥3 of 5 leads AND 95% CI excludes 0) | #54 | `docs/preregistration.md` |
| **03** | Recovered HEPPI dates | merged | `docs/heppi-date-map.csv`, `docs/heppi-reference-dataset.md` |
| **04** | Significance and scorecard machinery: paired block bootstrap, Diebold–Mariano, SEDI, twCRPS, PIT/reliability, BSS/CRPSS, leave-year-out climatology, per-day score files, scorecard runner | #56 | `src/weavr/significance.py`, `climatology.py`, `score_io.py`, `verify.py` additions, `scripts/run_scorecard.py`, `docs/scorecard-and-significance.md` |
| **05** | Daily-cadence 2020 JJAS stores (00 UTC inits, 122 days, was 18) | #57 | `scripts/build_*.py`, `scripts/validate_daily_stores.py`, `docs/baseline-store.md` |

## Findings that everything remaining must respect

- **No blend beats raw GraphCast on RMSE at 24–96 h.** Only Tier 2's EMOS
  single-source calibrations do.
- **N_eff ≈ 1.1 of 3 sources** (error correlations 0.62–0.90). HRES's zero
  weight in 25/30 Tier 1 cells is collinearity with IFS-ENS, not uselessness.
- **GraphCast detects no 115.6 mm event at any lead; no source detects any
  204.5 mm event.** RMSE and warning skill disagree, so don't optimise RMSE
  alone.
- **Zero of 1,691 scorecard comparisons have computable CIs** on the current
  3–4-day test sets (bootstrap is degenerate). That is what step 07's extra
  data exists to fix.
- **Units:** WeatherBench 2 precipitation is in metres; ×1000 before
  comparing with IMD mm.

## State of the data (gitignored, in `data/`)

| Store | Status |
|---|---|
| `baseline_2020_jjas_daily.zarr` | built, validated, 122 inits |
| `lagged_ensemble_inputs_2020_jjas_daily.zarr` | built, validated, 122 nominal inits |
| `ifs_ens_2020_jjas_daily.zarr` | **deliberately not built** (126 GB pull; skipped by decision) |

**Mixed-cadence caveat — must be stated wherever step 07 reports a Tier 2
number:** Tier 0, Tier 1 and single-source results get 122 days (via
`ifs_ens_mean` in the daily baseline store). Tier 2's EMOS-on-IFS-ENS and BMA
remain on the weekly 18-init `data/ifs_ens_2020_jjas.zarr`.

Minor known wart: the daily lagged store manifest's `_meta.init_cadence_days`
records 7 (the default flag) even though `nominal_times_from` records the
daily store.

## Repo conventions established during 01–05 (still binding)

- Each results method is written by exactly one script; per-day scores go to
  `results/per_day/<method>__lead<h>.csv` as counts and MSE, never ratios.
- RMSE is computed from daily MSE, never by averaging daily RMSEs.
- With n ≤ block length the bootstrap is degenerate: report NaN bounds,
  `significant=False`.
- Doc-only PRs run no CI (`paths-ignore`); squash merges; rebase before PRs.
