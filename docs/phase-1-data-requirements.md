# Phase 1 data requirements — checked against real data, not assumed

Goes through every verification metric in the [Phase 1
issue](https://github.com/ATMOSYNC/WEAVR/issues/3) and states what it
needs, whether `data/baseline_2020_jjas.zarr` (Phase 0's store) already
provides it, and what was actually done about any gap — checked live
against the real WeatherBench 2 catalog and real IMD pulls, the same
discipline Phase 0 used to find ECMWF's rate limit and Pangu's descending
latitude, rather than assumed from the metric names alone.

## Deterministic: RMSE, bias, ACC

**RMSE, bias**: need only a forecast/obs pair at matching valid times —
Phase 0's sparse (weekly-init, 5-lead) sampling is fine, since these are
computed independently per lead time. No extension needed.

**ACC** additionally needs a climatological reference to compute the
anomaly against. Checked WeatherBench 2's precomputed climatology catalog
(`gs://weatherbench2/datasets/era5-daily-climatology/`) — it exists, but
every variant is natively **1.5°** (240×121 grid), coarser than weavr's
locked 0.25° common grid. `regrid_to_common` would (correctly) raise
`SourceTooCoarseError` on it rather than silently upsampling. Building a
native-0.25° climatology from raw ERA5
(`gs://weatherbench2/datasets/era5/...-0p25deg...zarr`) ourselves is real,
separate pipeline work — multi-year aggregation across a large store — not
a small extension of this step.

**Decision (asked, not assumed)**: skip ACC for 2m_temperature in Tier 0.
Documented as deferred until a proper native-resolution climatology
pipeline exists — not silently dropped from the results table later, see
`docs/tier0-baseline-results.md` once step 4 runs. ACC for precipitation is
still available via IMD's own multi-year record (see SEEPS below, same
climatology archive serves both).

## Probabilistic: CRPS, Brier

Both need a predictive *distribution*, not a single value. The current
store's `ifs_ens_mean` group is already collapsed to a mean field by
WeatherBench 2 — structurally incapable of producing these scores.

Checked live: WeatherBench 2 **does** serve the full 50-member IFS ensemble,
uncollapsed (`gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr`,
`number` dim = 50). But its native chunking bundles all 50 members into one
I/O chunk per (init_time, lead_time) pair — measured **~47s per chunk**
fetched to disk (vs. ~2-3s for the deterministic sources' chunks, see
`docs/baseline-store.md`). At Phase 0's sampling density (weekly inits × 5
leads × 2 variables ≈ 180 chunks), a naive pull would take roughly 2-2.5
hours and ~30-35GB.

**Decision (asked, not assumed)**: defer CRPS/Brier to Phase 2. Phase 2
builds lagged ensembles from the deterministic AI models anyway (see
`docs/phase-plan.md`) — that's the point where probabilistic scoring
machinery needs to exist regardless, so building it now against a
single-source ensemble that took 2+ hours to fetch would be built twice.
Tier 0's results (step 4) report only deterministic and categorical scores;
`docs/tier0-baseline-results.md` states this omission explicitly rather
than silently leaving CRPS/Brier columns out with no explanation.

## Categorical: SEEPS, FSS/pFSS, POD/FAR/CSI/ETS

**POD/FAR/CSI/ETS**: fixed IMD thresholds (7.5/64.5/115.6/204.5mm), computed
per lead time from the existing sparse sampling — no extension needed.

**FSS/pFSS**: spatial, needs the full grid at each verified lead time.
Nothing in Phase 0's store subsamples space — already fine.

**SEEPS**: needs a per-gridpoint local climatology (a distribution of
historical daily rainfall at that point) to define its dry/light/heavy
terciles. One season (JJAS 2020 alone) is far too thin for this. Checked
live: IMD-only pulls have none of the WeatherBench-2 per-chunk GCS latency
that forced Phase 0's sparse sampling — one **full year** measured **~22s**
via `imdlib` (vs. ~2-3s *per chunk* for a single forecast source/lead/init
combination). Pulling many extra years costs no real sampling-density
tradeoff.

**Decision**: no ask needed here, this one had no real tradeoff — built
`scripts/build_seeps_climatology.py`, pulling IMD-only rainfall for JJAS
2006-2020 (15 years; every forecast this project will ever score is JJAS,
so the other eight months of each year aren't fetched) into
`data/imd_seeps_climatology_jjas.zarr`. Same idempotent-manifest pattern as
`build_baseline_store.py`. This archive also backs ACC-for-precipitation,
since it's the same kind of long-record climatological reference.

A real bug turned up while actually running this, not while designing it:
the first version concatenated years along `time` via `append_dim`, and two
years (2009, 2018) hit transient network timeouts on the first pass, got
retried afterward, and landed **out of chronological order** at the end of
the time axis instead of where they belonged. Fixed by writing one zarr
group per year instead (the same per-source-group pattern
`build_baseline_store.py` already uses) and adding `load_climatology()` to
open every year-group and concatenate+sort them — so retry order can never
leak into the data again. Covered by a regression test that reproduces the
exact ordering hazard (write a later year first, an earlier year second,
confirm the loaded result is still chronological).

## Cross-cutting: forecast valid-time vs. IMD day

Not a data-volume gap, but worth restating here so step 2 doesn't miss it:
`docs/baseline-store.md` already flags that forecast groups in the baseline
store are indexed by `(init_time, prediction_timedelta)`, not bucketed to
an IMD observation day. Every metric above that compares a forecast to an
IMD-observed day needs a `valid_time = init_time + prediction_timedelta`
computation, then a join to the correct IMD day, before scoring — that's
alignment code, not additional data, so it belongs in step 2
(`src/weavr/verify.py`) alongside the scoring functions that need it, not
in this ingestion step.

## Summary of what changed

| Metric | Extension needed? | Outcome |
|---|---|---|
| RMSE, bias | No | Use Phase 0 store as-is |
| ACC (temperature) | Native climatology, out of scope | Deferred — documented, not silently dropped |
| ACC (precipitation) | IMD climatology | Covered by `build_seeps_climatology.py`'s output |
| CRPS, Brier | Full IFS ensemble, ~2-2.5hr pull | Deferred to Phase 2 — documented |
| POD/FAR/CSI/ETS | No | Use Phase 0 store as-is |
| FSS/pFSS | No | Use Phase 0 store as-is |
| SEEPS | IMD-only multi-year climatology | Built — cheap, no tradeoff |

> **v2 (2026-10-01).** The numbers in this file are v1 (weekly, 2020 only,
> `n_train=14`, `n_test=3-4`) and are superseded by
> [`docs/v2-evidence-base-results.md`](v2-evidence-base-results.md), which is
> measured on the two-season daily LOYO base with real confidence intervals.
> This file is retained as history.
