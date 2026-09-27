# Tier 0 baseline results (Phase 1's exit criterion)

Produced by [`scripts/run_tier0_baseline.py`](../scripts/run_tier0_baseline.py).
This is the number every later tier (Phase 3's regional weights, Phase 4's
hierarchical BMA/EMOS, ...) must beat to justify its own added complexity.
Full numbers in [`results/tier0_baseline.csv`](../results/tier0_baseline.csv).

## What was averaged

The equal-weight mean of `total_precipitation_24hr` across **graphcast,
hres, ifs_ens_mean** — the 3 of 4 forecast sources in
`data/baseline_2020_jjas.zarr` that carry this variable. **pangu is
excluded from the mean**, not treated as contributing zero: its
WeatherBench 2 archive has no precipitation variable at all (documented
already in `docs/baseline-store.md`).

## Scope: precipitation only, not temperature

The store's only IMD ground truth is `imd_observed.rain` —
`build_baseline_store.py` never pulled an IMD temperature product, so there
is literally no observed temperature to verify `2m_temperature` against in
this store. IMD does publish a gridded temperature product, but at **1°**
native resolution, coarser than weavr's locked 0.25° grid — `regrid_to_common`
would (correctly) refuse it as upsampling, the same guard already blocking a
temperature climatology for ACC (`docs/phase-1-data-requirements.md`). Tier 0
therefore scores precipitation only. This is a scope limitation to revisit
if a matching-resolution temperature ground truth source is ever added, not
a silently narrowed deliverable.

## A real bug caught before trusting any number

WeatherBench 2's `total_precipitation_24hr` is in **meters** (the ECMWF/GRIB
convention); IMD's `rain` is in **millimeters**. Scoring them directly
against each other produces numbers that look plausible, not obviously
broken — e.g. a "bias" of roughly `-obs_mean`, because the forecast is
effectively ~0 in the wrong units next to real millimeter values. Caught by
checking actual magnitudes before trusting a score (`forecast.mean() ~
0.006`, `obs.mean() ~ 5.7`) rather than assuming matching units, and fixed
by converting forecast precipitation ×1000 before scoring
(`PRECIP_M_TO_MM` in the script).

## Split used

`leave_one_year_out` is not usable: the paired forecast+obs store covers
one season only (checked directly, per `docs/phase-1-data-requirements.md`
and `src/weavr/splits.py`'s own design). Every lead time uses
`seasonal_block_split(test_fraction=0.2)` instead — a trailing ~20% block of
each lead's valid days held out as test, computed from the actual sorted
timestamps present (see `src/weavr/splits.py`). The script states this
explicitly per run rather than silently falling back.

**Sample sizes are small** (14 train / 3-4 test per lead, out of Phase 0's
deliberately sparse weekly-init sampling — see `docs/baseline-store.md` for
why) — every number below should be read as a first, real baseline
established for later tiers to beat, not as a statistically robust skill
estimate. This is an inherent consequence of Phase 0's sampling-density
tradeoff, not a bug in this step.

## CRPS/Brier: not computed

No ensemble-shaped forecast source exists in this store (`ifs_ens_mean` is
already collapsed to a mean field) — deferred to Phase 2, which needs the
same probabilistic-scoring machinery for lagged AI ensembles anyway. See
`docs/phase-1-data-requirements.md` for the full reasoning (measured
~2-2.5h/~30GB cost to pull the full IFS ensemble at this sampling density).

## ACC climatology

Computed as the per-gridpoint **mean** rainfall across all 1830 JJAS days
(2006-2020) in `data/imd_seeps_climatology_jjas.zarr` — a single
climatological value per gridpoint, not a finer day-of-year climatology
(which would need either more years or an explicit smoothing choice this
step didn't make). A simplification worth revisiting if ACC needs finer
temporal resolution than "the JJAS-season average" later.

## Results

FSS computed at a single neighborhood size (5 grid cells, ~1.25° edge — a
representative mesoscale window; not swept over multiple sizes here).

| Lead (h) | n (train/test) | RMSE (mm) | Bias (mm) | ACC | SEEPS |
|---|---|---|---|---|---|
| 24  | 14/4 | 11.66 | +0.79 | 0.42 | 1.38 |
| 48  | 14/4 | 13.54 | +1.60 | 0.31 | 1.57 |
| 72  | 14/4 | 13.38 | +1.56 | 0.37 | 1.43 |
| 96  | 14/3 | 14.27 | +0.17 | 0.37 | 1.34 |
| 120 | 14/3 | 15.30 | +0.70 | 0.25 | 1.65 |

RMSE increases and ACC broadly decreases with lead time — the expected
direction of forecast skill decay, a sanity check the pipeline is doing
something real, not an artifact. SEEPS stays well below its no-skill
reference value of 2 at every lead (see `src/weavr/verify.py`'s docstring
for what that reference means), consistent with real skill.

### Contingency scores (POD / FAR / CSI / ETS) at IMD thresholds

| Lead (h) | 7.5mm | 64.5mm | 115.6mm | 204.5mm |
|---|---|---|---|---|
| 24  | POD .61 FAR .51 CSI .37 ETS .25 | POD .17 FAR .69 CSI .12 ETS .12 | POD .12 FAR .71 CSI .09 ETS .09 | POD 0 FAR NaN CSI 0 ETS 0 |
| 48  | POD .66 FAR .56 CSI .36 ETS .24 | POD .02 FAR .97 CSI .01 ETS .01 | POD 0 FAR 1.0 CSI 0 ETS ≈0 | POD 0 FAR NaN CSI 0 ETS 0 |
| 72  | POD .68 FAR .55 CSI .37 ETS .24 | POD .13 FAR .88 CSI .07 ETS .06 | POD .04 FAR .95 CSI .02 ETS .02 | POD 0 FAR NaN CSI 0 ETS 0 |
| 96  | POD .62 FAR .52 CSI .37 ETS .23 | POD .10 FAR .78 CSI .07 ETS .07 | POD 0 FAR 1.0 CSI 0 ETS ≈0 | POD 0 FAR NaN CSI 0 ETS 0 |
| 120 | POD .62 FAR .63 CSI .30 ETS .16 | POD .05 FAR .79 CSI .04 ETS .04 | POD 0 FAR 1.0 CSI 0 ETS ≈0 | POD 0 FAR NaN CSI 0 ETS 0 |

**204.5mm (IMD's most extreme category) never has a single forecast-yes
event across any lead's tiny test fold** — FAR is correctly NaN (0/0, per
`contingency_scores`'s documented degenerate-case behavior), not a bug.
With only 3-4 test days per lead, this reflects too little data to evaluate
the rarest category, not necessarily zero skill at extreme rain — worth
revisiting once more seasons exist and `leave_one_year_out` becomes usable.

### FSS at 5-cell neighborhood

| Lead (h) | 7.5mm | 64.5mm | 115.6mm | 204.5mm |
|---|---|---|---|---|
| 24  | 0.252 | 0.201 | 0.068 | 0.0 |
| 48  | 0.221 | 0.055 | 0.003 | 0.0 |
| 72  | 0.235 | 0.148 | 0.108 | 0.0 |
| 96  | 0.240 | 0.275 | 0.004 | 0.0 |
| 120 | 0.209 | 0.179 | 0.0 | 0.0 |

204.5mm's FSS is 0.0 (not NaN) at every lead because *some* neighborhood
exceeds it in the observed field even though the forecast never does —
`reference` (the denominator) is nonzero, so this is a real (very low, as
expected for an event the equal-weight mean never predicts) score, unlike
the contingency table's NaN case above.

## Regenerating

```bash
python scripts/run_tier0_baseline.py
```

Override the store, climatology archive, output path, test fraction, or
neighborhood size via `--store`, `--climatology`, `--out-csv`,
`--test-fraction`, `--neighborhood-size`.
