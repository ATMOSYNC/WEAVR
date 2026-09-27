# WEAVR

Hybrid AI–NWP multi-model forecast blending (SIH PS 26081).

See [docs/phase-plan.md](docs/phase-plan.md) for the build plan, tracked as
GitHub issues per phase. See [docs/data-sources.md](docs/data-sources.md),
[docs/grid-and-time-convention.md](docs/grid-and-time-convention.md), and
[docs/baseline-store.md](docs/baseline-store.md) for the Phase 0 data
pipeline: `scripts/build_baseline_store.py` builds the baseline JJAS
forecast + IMD observation store. See
[docs/heppi-reference-dataset.md](docs/heppi-reference-dataset.md) for a
third-party NCMRWF-ensemble + IMD reference dataset (not redistributed here)
useful for Phase 1 bias-correction work. See
[docs/phase-1-data-requirements.md](docs/phase-1-data-requirements.md) for
what each Phase 1 verification metric needs and what was extended:
`scripts/build_seeps_climatology.py` builds the multi-year IMD-only
climatology archive SEEPS/ACC-for-precipitation need.
[src/weavr/verify.py](src/weavr/verify.py) implements the Phase 1
verification protocol itself: RMSE/bias/ACC, CRPS/Brier, SEEPS, FSS, and
POD/FAR/CSI/ETS at IMD's own rain thresholds.
[src/weavr/splits.py](src/weavr/splits.py) is the only sanctioned way to
split time series data in this codebase: `leave_one_year_out` (needs 2+
years, not yet true for the current single-season baseline store) and
`seasonal_block_split` (usable now) — never a random split. See
[docs/tier0-baseline-results.md](docs/tier0-baseline-results.md) for Phase
1's exit criterion: `scripts/run_tier0_baseline.py` scores the equal-weight
mean forecast against IMD gauges — the benchmark every later phase must beat.
See [docs/baseline-store.md](docs/baseline-store.md)'s Phase 2 addition for
`scripts/build_lagged_ensemble_store.py`, which pulls the extra GraphCast/
Pangu init times a ±4-starts/12h-spaced lagged ensemble needs, turning these
deterministic AI models into probabilistic ensembles comparable to NWP
ensemble members.
[src/weavr/ensemble.py](src/weavr/ensemble.py) is the pure-function
counterpart: `build_lagged_ensemble` assembles one nominal forecast's
lagged ensemble from an already-open source dataset, producing the
`member`-dimensioned shape `verify.py`'s `crps`/`brier_score` expect. See
[docs/phase2-ensemble-baseline-results.md](docs/phase2-ensemble-baseline-results.md)
for Phase 2's exit criterion: `scripts/run_phase2_ensemble_baseline.py`
scores GraphCast's lagged precipitation ensemble with CRPS/Brier against
IMD gauges — the probabilistic counterpart to Phase 1's Tier 0 baseline —
and documents its dispersion characteristics (`weavr.verify.spread_skill_ratio`,
`calibrated_spread_skill_ratio`): the lagged ensemble is measured to be
strongly under-dispersive at every lead, a finding later phases' weighting
decisions need to account for. See
[docs/phase3-cv-and-regional-scheme.md](docs/phase3-cv-and-regional-scheme.md)
for Phase 3's scoping decisions, and
[src/weavr/regions.py](src/weavr/regions.py) for `assign_regions`, which
pools weavr's common India grid into Sreekala & Babu's six homogeneous
monsoon rainfall zones (WC/SI/WI/CI/NE1/NE2) for per-region skill weighting.
[src/weavr/weighting.py](src/weavr/weighting.py) fits those per-region
weights (`fit_region_weights`): unconstrained regression per Wanders &
Wood (2016), negative weights clipped to zero (Wang et al. 2025) and
renormalized to sum to 1, with a documented equal-weight fallback for
regions with too few train samples or a rank-deficient fit. See
[docs/tier1-regional-weights-results.md](docs/tier1-regional-weights-results.md)
for Phase 3's exit criterion: `scripts/run_tier1_regional_baseline.py`
fits and scores the real regional blend against IMD gauges and compares
it directly, region by region and lead by lead, against Tier 0's
equal-weight mean — the regional blend wins domain-wide at every lead and
in 22 of 30 region×lead cells, not universally, reported plainly either way.
See [docs/phase4-data-and-combiner-scope.md](docs/phase4-data-and-combiner-scope.md)
for Phase 4's scoping decisions: `seasonal_block_split` substitutes for an
unreachable rolling training window, GraphCast's lagged pseudo-ensemble
plus a newly-decided real IFS 50-member ensemble pull back the EMOS-CSG/BMA
combiners (HEPPI is a validation reference only, not training data), and
the real per-bin sample counts found the extreme-rain bin (204.5mm+) is
never fittable at any lead. See `docs/baseline-store.md`'s Phase 4 addition
for `scripts/build_ifs_ensemble_store.py`, which pulls that real IFS
50-member ensemble into `data/ifs_ens_2020_jjas.zarr` — ~55 minutes of real
GCS transfer time, resumable across the idle-sleep-induced connection
stalls that interrupted the live run.
[src/weavr/rain_bins.py](src/weavr/rain_bins.py) implements the first
Phase 4 checkbox itself: `classify_rain_bin` labels each forecast value
into IMD's rain-intensity bins (dry/light/heavy/very_heavy/extremely_heavy
— the upper three are IMD's own published category names, at its own
64.5/115.6/204.5mm boundaries), distinct from `verify.py`'s obs-vs-forecast
contingency scoring at the same thresholds, with missing (NaN) forecast
values explicitly excluded rather than silently landing in the last bin
(`numpy.digitize`'s own default behavior).
[src/weavr/emos.py](src/weavr/emos.py) implements issue #6's named default
combiner, EMOS-CSG: `fit_emos_csg` fits a censored-shifted-gamma
distribution (Scheuerer & Hamill 2015; Baran & Nemoda 2016) per rain-
intensity bin, per ensemble source, minimizing CRPS via a from-scratch,
numerically-verified closed form (`csgd_crps`) rather than the paper's own
published equation, which did not reproduce the correct value when checked
against brute-force numerical integration. Falls back to a point-mass-at-
zero prediction, flagged via `CensoredShiftedGammaResult.is_fallback`, for
a bin with too few train days or a degenerate (all-dry) climatology.
[src/weavr/bma.py](src/weavr/bma.py) implements issue #6's named
comparison combiner, hierarchical BMA: `fit_hierarchical_bma` fits a
mixture, per rain-intensity bin and per `weavr.regions` zone, of each
source's own point-mass-plus-power-transformed-gamma predictive
distribution (Sloughter et al. 2007 -- not Raftery et al. 2005's Gaussian
form), with mixture weights fit by EM. Ensemble sources (GraphCast, IFS)
regress their component's variance on real per-cell ensemble spread;
HRES, with no ensemble, uses Sloughter et al.'s own fixed-variance
"kernel dressing" route instead. `score_bma` estimates CRPS via Monte
Carlo sampling (no closed form exists for a general mixture), reusing
`weavr.verify.crps`'s ensemble machinery -- the same predict/score
interface shape as `weavr.emos.score_csgd`, so Phase 4's final comparison
can call both combiners identically.
See [docs/tier2-hierarchical-baseline-results.md](docs/tier2-hierarchical-baseline-results.md)
for Phase 4's exit criterion:
`scripts/run_tier2_hierarchical_baseline.py` fits and scores both real
combiners against the real stores on one shared train/test split, and
reports an honest, mixed comparison -- EMOS-CSG and BMA really do trade
off which they win (EMOS-CSG dominates the "heavy" rain-intensity bin at
most leads, BMA wins domain-wide at 2 of 5 leads), and Tier 2 does not
beat Tier 0/Tier 1 everywhere either. Running it against the real data
also caught and fixed a real numerical bug in `weavr.emos.csgd_crps`
(tiny negative-precipitation numerical-noise artifacts in GraphCast's own
real forecast data could produce a physically nonsensical CRPS), documented
in that results doc.
See [docs/phase5-regime-covariate-scope.md](docs/phase5-regime-covariate-scope.md)
for Phase 5's scoping: which regime covariates issue #7 names are real and
actually obtainable for this project's 2020 JJAS-only data -- monsoon
active/break (Rajeevan, Gadgil & Bhate 2010, derivable from `imd_observed`
directly) and MJO phase (Kiladis et al. 2014's OMI) and monsoon-depression
presence (a real ERA5-derived low-pressure-system track catalogue) are in
scope; Neal et al.'s 30-weather-pattern classification (covers only
1979-2016, not this project's 2020 season) and western-disturbance
presence (real, but ~50x rarer in JJAS than winter -- near-zero signal in
a JJAS-only season) are not.
[src/weavr/regimes.py](src/weavr/regimes.py) implements the first
covariate directly (`classify_monsoon_active_break`, spot-checked against
real, independently reported 2020 monsoon activity); the other two are
built by [scripts/fetch_omi_mjo_index.py](scripts/fetch_omi_mjo_index.py)
and
[scripts/build_monsoon_depression_index.py](scripts/build_monsoon_depression_index.py)
-- real per-category counts for all three (14 active / 3 break days; an
8-phase MJO spread; 14 real depression-or-stronger days) are in
`docs/phase5-regime-covariate-scope.md`'s own step 2 update.
See [docs/phase5-regime-conditioned-results.md](docs/phase5-regime-conditioned-results.md)
for step 3's real assessment of whether a regime covariate plausibly
explains any of Phase 4's own error pattern (checked directly against the
real Tier 2 train/test data, not assumed): the held-out test window used
to measure Phase 4's heavy-bin EMOS-vs-BMA split turns out to have almost
no regime diversity at any lead, ruling out a regime explanation for that
specific alternation, while training data shows only a weak, noisy
correlation between monsoon-active days and elevated heavy-rain cell
counts. [src/weavr/regime_weighting.py](src/weavr/regime_weighting.py)
implements the simplest defensible regime-stratified fit anyway (reusing
`weavr.weighting.fit_region_weights`'s own OLS/fallback machinery,
conditioned on monsoon active/break instead of spatial region); given this
weak evidence and a training set with only 2 "active" days per lead, the
user decided via `AskUserQuestion` not to build the optional heavier
GBM/ViT-style gating model issue #7 names.
[scripts/run_tier3_regime_conditioned_baseline.py](scripts/run_tier3_regime_conditioned_baseline.py)
applies issue #7's own go/no-go criterion (decided before running it: beat
Phase 4's real EMOS-CSG/BMA on domain-wide CRPS at >= 3 of 5 leads),
recomputing Phase 4's own combiners on the identical split. **Result: NO
GO** -- the regime-conditioned model wins 0 of 5 leads domain-wide, and 0
of 15 (lead, bin) cells, reported plainly per
`docs/phase5-regime-conditioned-results.md`'s own step 4 (which also names
why: weak underlying regime signal, plus a structural CRPS disadvantage
from being a deterministic point-forecast blend rather than a fitted
predictive distribution like EMOS-CSG/BMA). Phase 5's regime-conditioned
model is not adopted.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest --cov=weavr
```

CI runs lint, type-check, tests (3.11/3.12), and publishes a build artifact
on every push/PR to `main`.