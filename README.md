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

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest --cov=weavr
```

CI runs lint, type-check, tests (3.11/3.12), and publishes a build artifact
on every push/PR to `main`.