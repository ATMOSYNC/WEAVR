# Phase 4 scoping: data and combiner feasibility

Answers, checked against the real repo state (not assumed from issue #6's
phrasing), to the open questions in
[`solving issues/05-phase-4-hierarchical-bma-emos/01-check-data-and-combiner-feasibility.md`](../../solving%20issues/05-phase-4-hierarchical-bma-emos/01-check-data-and-combiner-feasibility.md)
before any bin-classification or combiner code is written.

## Rolling training window: not reachable on today's data, `seasonal_block_split` used instead

Directly inspected `data/baseline_2020_jjas.zarr` and
`data/lagged_ensemble_inputs_2020_jjas.zarr`: every source (`graphcast`,
`hres`, `ifs_ens_mean`, `pangu`) has exactly **18 timestamps**
(2020-06-01 .. 2020-09-28, weekly), 5 `prediction_timedelta` leads, and no
rolling calendar to slide a 60–90 day window across — this is a static,
single-season store, not a daily-refreshed archive. A literal rolling
window is not achievable until Phase 6's operational, daily-refreshed
pipeline exists (issue #8's own "daily-refreshed rolling verification"
checkbox). **Decision** (forced by the data, not a real tradeoff — the
data genuinely doesn't support the alternative): Phase 4 uses
`seasonal_block_split(test_fraction=0.2)`, exactly the convention every
prior tier (Tier 0, Phase 2, Tier 1) already used, and states this
explicitly rather than pretending a rolling refit cadence exists on static
data. Revisit once Phase 6 makes daily refitting real.

## Ensemble source: GraphCast alone confirmed as today's only real ensemble — decision made to pull the real IFS ensemble

Directly inspected every baseline-store group's dims: `graphcast`, `hres`,
`ifs_ens_mean`, and `pangu` are **all** `(time, prediction_timedelta,
latitude, longitude)` in `data/baseline_2020_jjas.zarr` — no `member`
dimension anywhere in that store. `data/lagged_ensemble_inputs_2020_jjas.zarr`
has only `graphcast` and `pangu` groups (Phase 2's lagged pseudo-ensembles),
and only `graphcast` carries `total_precipitation_24hr` (pangu has no
precipitation variable at all, per `docs/baseline-store.md`). **Confirmed:
GraphCast's Phase 2 lagged pseudo-ensemble is the only ensemble-shaped
precipitation forecast that exists in this project today.**

This isn't a new gap — `docs/baseline-store.md`'s own `ifs_ens_mean` row
already flagged it: *"ensemble mean, not the full 50-member ensemble —
chosen to keep store size reasonable for a baseline; full-ensemble access
is a Phase 1+ decision if BMA/EMOS needs individual members."* And
`docs/phase-1-data-requirements.md` already measured the real cost of
getting it: WeatherBench 2 serves the full, uncollapsed 50-member IFS
ensemble (`gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr`),
confirmed live, at ~47s/chunk vs. ~2-3s for deterministic sources — roughly
**2-2.5 hours and 30-35GB** at Phase 0's weekly sampling density, deferred
at the time specifically pending a phase that needed individual members.
Phase 4 is that phase.

This is a genuine cost/benefit tradeoff (a real, nontrivial one-time fetch
cost vs. GraphCast-only scope vs. a second synthetic lagged pseudo-ensemble
for HRES), so it was put to the user directly via `AskUserQuestion` rather
than decided silently. **Decision: pull the real IFS 50-member ensemble.**
Phase 4 will therefore fit combiners against **two real ensemble sources**
— GraphCast's Phase 2 lagged pseudo-ensemble and the newly-pulled real IFS
50-member ensemble — rather than GraphCast alone or a second synthetic
approximation. Building the actual fetch (a `build_ifs_ensemble_store.py`
analog to Phase 2's `build_lagged_ensemble_store.py`) is real engineering
work, done as its own step, not folded silently into the combiner-fitting
step — see this doc's "Folder restructuring" section below.

## HEPPI's role: methodology/validation reference only, not training data

Directly opened the real files in `HEPPI/` (outside this repo, per
`docs/heppi-reference-dataset.md`):

- `IMD_observed.nc` and `NCMRWF_orig_forecast.nc` both have a bare
  `Dimensions without coordinates: ... forecast` axis (334 samples,
  23 ensemble members) — **no time/date coordinate anywhere in the file**,
  confirmed directly, not re-quoted from the doc.
- `HEPPI_readme.txt` (the original author's own description) documents
  "334 total timesteps across two monsoon seasons" but likewise gives no
  calendar mapping.
- `example_EMOS_fit.R` (the original author's own EMOS script) reads dates
  from a **separate** `dat_vals`/`IMD_dates.mat`-style file that is **not**
  part of this download — confirming the calendar was real at the time but
  isn't recoverable from what we have.
- The grid **is** confirmed to match exactly: `IMD_lat`/`IMD_lon` in
  `IMD_observed.nc` equal `weavr.grid.COMMON_LAT`/`COMMON_LON` bit-for-bit
  (lat 6.5–38.5, lon 66.5–100.0, 129×135).

**Decision** (forced by the missing calendar, not a real tradeoff): HEPPI
cannot be joined by date to any of weavr's own dated stores, so it is used
only as a **methodology/validation reference**, never as Phase 4's
training data. Concretely: `example_EMOS_fit.R` (line 74) itself calls
`ensembleMOS(..., control = controlMOScsg0(scoringRule = c("crps")), model
= "csg0", ...)` — the R `ensembleMOS` package's censored-shifted-gamma,
CRPS-minimized model — direct, real confirmation that "EMOS-CSG,
CRPS-minimized" is exactly what the original HEPPI/NEPS-G project fit on
this same grid, not a coincidental naming match. Step 5's results doc will
note this directional consistency; it will not attempt a literal
reproduction of Angus et al.'s numbers (different ensemble, different
seasons, no shared dates to align on).

## Real per-bin sample counts: dry/light/moderate are fittable everywhere, heavy is marginal, extreme is not fittable at all

Using `weavr.verify.IMD_RAIN_THRESHOLDS_MM` (7.5/64.5/115.6/204.5mm) as bin
edges and GraphCast's forecast values (the "classify each day's
ensemble-mean forecast" quantity, checked per gridpoint-day, matching how
finely rainfall actually varies spatially within one day), pooled over the
14 `seasonal_block_split` train days at each lead:

| Lead (h) | dry (points / days) | light (points / days) | moderate (points / days) | heavy (points / days) | extreme (points / days) |
|---|---|---|---|---|---|
| 24  | 166,317 / 14 | 75,714 / 14 | 1,548 / 14 | 227 / 9 | 4 / 3 |
| 48  | 157,920 / 14 | 84,758 / 14 |   997 / 14 | 123 / 6 | 12 / 1 |
| 72  | 154,492 / 14 | 88,716 / 14 |   583 / 12 |  19 / 2 |  0 / 0 |
| 96  | 152,912 / 14 | 90,346 / 14 |   538 / 13 |  14 / 3 |  0 / 0 |
| 120 | 155,665 / 14 | 87,627 / 14 |   490 / 10 |  28 / 2 |  0 / 0 |

("points" = pooled (train day × gridpoint) cells landing in that bin;
"days" = distinct train days contributing at least one such cell.)

**Decision**: a bin is fit for real only if it has **at least 5 distinct
contributing train days** — chosen because a combiner fit from fewer
independent days is fitting temporal noise, not a regime, the same
reasoning Phase 3's `MIN_TRAIN_SAMPLES` used for its own (different)
granularity. Against this threshold: **dry, light, and moderate are
fittable at every lead**; **heavy is fittable only at 24h (9 days) and 48h
(6 days)**, falling back at 72h/96h/120h (2-3 days); **extreme is never
fittable at any lead** (0-3 days). This is a new constant
(`MIN_TRAIN_DAYS_PER_BIN = 5`), not inherited from Phase 3, since Phase 4's
bins are a temporal/intensity regime split, a genuinely different
granularity than Phase 3's spatial regions. Steps 3/4's fallback (for both
EMOS-CSG and BMA) is the same documented, flagged pattern
`src/weavr/weighting.py` established:
`is_fallback=True`/`reason` set, with a plain climatological point-mass
fallback distribution (not a fitted CSG/mixture) for a bin that can't be
fit for real.

## EMOS-CSG and BMA: the real cited methodologies, checked before implementing

- **EMOS-CSG**: Scheuerer & Hamill (2015), *Monthly Weather Review* 143,
  4578–4596 (DOI 10.1175/mwr-d-15-0061.1) — fits a censored, shifted gamma
  distribution (left-censored at zero, so it can produce exact-zero
  forecasts) whose location/scale are connected to the ensemble's mean and
  spread via a nonhomogeneous regression, with parameters estimated by
  **minimizing mean CRPS** over the training period (not maximum
  likelihood). Baran & Nemoda (2016), *Environmetrics* 27, 280–292
  (arXiv:1512.04068) is the companion paper issue #6 also names, confirming
  the same CSG structure and reporting it beats both the raw ensemble and
  a BMA model on their own precipitation test cases — real literature
  context step 5's honest comparison should cite, not assume Phase 4's own
  result will match.
- **Hierarchical BMA**: Raftery, Gneiting, Balabdaoui & Polakowski (2005),
  *Monthly Weather Review* 133, 1155–1174 — the general BMA recipe: a
  predictive PDF as a weighted mixture of each component model's own
  bias-corrected predictive distribution, mixture weights fit by EM to
  maximize training-period log-likelihood. That paper's own component
  distributions are Gaussian (built for temperature/pressure) — **not**
  directly applicable to precipitation's mixed discrete/continuous
  structure. The actual precipitation-specific extension is **Sloughter,
  Raftery, Gneiting & Fraley (2007)**, *Monthly Weather Review* 135,
  3209–3220: each component's predictive PDF is a mixture of a point mass
  at zero and a power-transformed gamma distribution. **Decision** (a
  single correct answer, not a tradeoff): step 4 implements Sloughter et
  al. (2007)'s precipitation-BMA form, not the original 2005 Gaussian
  form, since this project is precipitation-only throughout (same scoping
  logic as every prior tier).

## Folder restructuring: a new step 2 for the real IFS ensemble pull

The `AskUserQuestion` decision above (pull the real IFS ensemble) is real
engineering work — a store-building script, not something to fold silently
into the combiner-fitting step. `05-phase-4-hierarchical-bma-emos/` is
restructured to insert it as its own step, renumbering what were steps
2-4:

1. `01-check-data-and-combiner-feasibility.md` (this step, unchanged)
2. **New**: `02-build-ifs-ensemble-store.md` — pulls the real 50-member
   IFS ensemble (`gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr`),
   at the same 18-timestamp/5-lead sampling density as the rest of the
   baseline store, into `data/ifs_ens_2020_jjas.zarr`.
3. `03-implement-rain-intensity-bins.md` (was 02)
4. `04-implement-emos-csg-combiner.md` (was 03) — now fits against
   **both** GraphCast's lagged pseudo-ensemble and the new real IFS
   ensemble.
5. `05-implement-hierarchical-bma-combiner.md` (was 04) — implements
   Sloughter et al. (2007)'s precipitation-BMA form, with GraphCast and
   IFS as its two real-ensemble components.
6. `06-run-tier2-hierarchical-baseline.md` (was 05)

## Summary of decisions for steps 2-6

| Decision | Choice | Basis |
|---|---|---|
| Training window | `seasonal_block_split(test_fraction=0.2)` | Forced — no rolling calendar exists in any static store |
| Ensemble source | GraphCast's lagged pseudo-ensemble **+** a newly-pulled real IFS 50-member ensemble | User-confirmed tradeoff; IFS pull was pre-flagged in Phase 0/1, cost already measured (~2-2.5h/~30-35GB) |
| HEPPI's role | Methodology/validation reference only | Forced — no calendar axis to join it to weavr's dated stores |
| Bin fittability | dry/light/moderate: every lead; heavy: 24h/48h only; extreme: never | Real per-bin train-day counts, `MIN_TRAIN_DAYS_PER_BIN = 5` |
| EMOS-CSG recipe | Scheuerer & Hamill (2015) CSG, CRPS-minimized | The real cited paper's own method |
| BMA recipe | Sloughter et al. (2007) precipitation-BMA (point mass + gamma) | Raftery et al. (2005)'s own component form (Gaussian) doesn't fit precipitation; this is the real precip-specific extension |
