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
useful for Phase 1 bias-correction work.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest --cov=weavr
```

CI runs lint, type-check, tests (3.11/3.12), and publishes a build artifact
on every push/PR to `main`.