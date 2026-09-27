# Phase 2 ensemble baseline results (Phase 2's exit criterion)

Produced by [`scripts/run_phase2_ensemble_baseline.py`](../scripts/run_phase2_ensemble_baseline.py).
The probabilistic counterpart to Phase 1's Tier 0 baseline
([`docs/tier0-baseline-results.md`](tier0-baseline-results.md)): where Tier 0
scored a deterministic equal-weight mean, this scores the ±4-starts,
12h-spaced lagged pseudo-ensembles [`src/weavr/ensemble.py`](../src/weavr/ensemble.py)
builds from GraphCast/Pangu, using CRPS and Brier — the two metrics Phase 1
could not compute at all, because no ensemble-shaped source existed yet.
Full numbers in [`results/phase2_ensemble_baseline.csv`](../results/phase2_ensemble_baseline.csv).

## What was scored, and what wasn't

**Precipitation only, scored.** Exactly Tier 0's own limitation, still true
here: the baseline store's only IMD ground truth is `imd_observed.rain`, and
IMD's own gridded temperature product is 1° native — coarser than weavr's
locked 0.25° grid, so it can't back a temperature climatology or observation
to score CRPS against. Building a temperature *ensemble* (Phase 2's actual
job) doesn't change this — there is still nothing in this repo's data to
verify it against.

**GraphCast and Pangu temperature ensembles are still built and counted**
(90 each, across 18 weekly nominal times × 5 lead hours) — confirming
`build_lagged_ensemble` works uniformly across every real AI source/variable
combination this project has, using the actual fetched data, not just
synthetic tests. They are not scored.

Of the two AI sources, only **GraphCast** carries `total_precipitation_24hr`
— Pangu's WeatherBench 2 archive has no precipitation variable at all
(documented already in `docs/baseline-store.md`), so it contributes no
precipitation ensemble to score, same as it contributed nothing to Tier 0's
equal-weight mean.

## No GCS re-fetch

This script does not pull anything from GCS. `scripts/build_lagged_ensemble_store.py`
already fetched every `(time, prediction_timedelta)` pair a lagged ensemble
needs into `data/lagged_ensemble_inputs_2020_jjas.zarr` — including working
around a real stall found running that fetch (see `docs/baseline-store.md`).
`_reconstruct_raw_source` rebuilds an in-memory Dataset shaped like the raw
WeatherBench 2 archives directly from that already-fetched local store, and
`build_lagged_ensemble` is called against it exactly as it would be against
a live source — verified to round-trip exactly (a reconstructed ensemble was
checked element-for-element against the dense store's own already-assembled
slice for the same nominal forecast, and matched). This is the first time
`build_lagged_ensemble` has been exercised against real data rather than
synthetic test fixtures (`tests/test_ensemble.py` covers the synthetic case).

## A real bug caught by the tests, not assumed away

`_exceedance_probability` computes the fraction of ensemble members at or
above an IMD threshold, to feed `brier_score`. The first version did
`(ensemble >= threshold).mean(dim="member", skipna=True)` — but `NaN >=
threshold` evaluates to `False` in numpy, not `NaN`. A missing member (the
short-window NaN-padding `src/weavr/ensemble.py` documents, real at lead=24h
and lead=48h) was silently counting as "did not exceed" instead of being
excluded, understating exceedance probability exactly at the leads with
fewer real members. Caught by
`tests/test_run_phase2_ensemble_baseline.py::TestExceedanceProbability::test_ignores_nan_members`,
not by inspection. Fixed by re-masking the boolean comparison back to `NaN`
wherever the source member was `NaN` before averaging, so `skipna=True`
excludes only genuinely missing members. **The numbers below are the
corrected ones** — the bug was fixed and the real scoring run repeated
before this doc was written, not documented as a known issue and left in.

## Sample counts do not grow

Collapsing 9 lagged members into one ensemble still verifies against
exactly one IMD day per nominal forecast — the *count* of independent test
days is identical to Tier 0's, not larger. `n_train`/`n_test` below match
Tier 0's own pattern exactly (14 train throughout; 4 test at 24/48/72h, 3 at
96/120h) because both scripts share the same 18 weekly nominal times and the
same JJAS-boundary edge effect: at 96h/120h lead, the last nominal week
(2020-09-28) validates on 2020-10-01/10-02 — past the store's JJAS
observation range (ends 2020-09-30) — so that sample is correctly dropped,
not a new issue introduced here. Ensemble members add *spread*, useful for
CRPS/Brier's probabilistic scoring, not more independent verification days.

## Precipitation units

The same `total_precipitation_24hr` meters-to-millimeters conversion
`scripts/run_tier0_baseline.py` established (`PRECIP_M_TO_MM = 1000.0`) is
imported directly here, not re-derived — confirmed applied by inspecting the
CRPS magnitudes below (a few mm, the same order as Tier 0's RMSE), not
assumed correct because the import succeeded.

## Split used

Same as Tier 0: `leave_one_year_out` is not usable (one season only), so
every lead uses `seasonal_block_split(test_fraction=0.2)` — a trailing ~20%
block of each lead's valid days held out as test.

## Results

| Lead (h) | n (train/test) | CRPS (mm) | Brier 7.5mm | Brier 64.5mm | Brier 115.6mm | Brier 204.5mm |
|---|---|---|---|---|---|---|
| 24  | 14/4 | 5.34 | 0.259 | 0.0028 | 0.00025 | 1.4e-05 |
| 48  | 14/4 | 5.56 | 0.255 | 0.0039 | 0.00054 | 1.4e-05 |
| 72  | 14/4 | 5.12 | 0.251 | 0.0032 | 0.00043 | 2.9e-05 |
| 96  | 14/3 | 6.03 | 0.295 | 0.0043 | 0.00067 | 1.9e-05 |
| 120 | 14/3 | 6.57 | 0.290 | 0.0044 | 0.00090 | 7.7e-05 |

CRPS is a few mm — the same order of magnitude as Tier 0's RMSE (11.7-15.3mm),
and CRPS is expected to sit below the deterministic-mean RMSE for a
reasonably-spread ensemble, which it does throughout. Brier scores shrink by
roughly two orders of magnitude per threshold step, matching how rare each
successive IMD category genuinely is in this small sample (204.5mm — IMD's
most extreme category — is a near-zero-probability event for the
equal-weight-style lagged ensemble at every lead, consistent with Tier 0's
own finding that this threshold saw zero forecast-yes events in its test
folds).

## Ensemble dispersion characteristics (issue #4's third checkbox)

Issue #4 also asks to "note dispersion characteristics (lagged AI ensembles
tend to be under-dispersive) for later weighting decisions" — measured here
against the real data, not restated from the literature as a given.

**The diagnostic**: for a *perfectly calibrated* ensemble of `M` members
(one where the observation is statistically exchangeable with the
members), `RMSE(ensemble mean) / spread` is expected to be
`sqrt((M+1)/M)` — not 1.0 — a finite-ensemble-size correction (Fortin et
al. 2014; Leutbecher & Palmer 2008), verified here by simulation before
being used as the comparison target (`weavr.verify.calibrated_spread_skill_ratio`,
`tests/test_verify.py::TestCalibratedSpreadSkillRatio`). A measured ratio
well *above* this calibrated target indicates under-dispersion: the
ensemble's spread is narrower than its actual error.

### A real member-count bug found while computing this

Building this diagnostic surfaced a bug in the member counts this doc and
`docs/baseline-store.md` had previously reported for precipitation. See
`docs/baseline-store.md`'s "second real bug" note for the full story: the
original verification checked `2m_temperature`'s NaN pattern and
(incorrectly) assumed `total_precipitation_24hr` followed the same 6/9,
8/9, 9/9 pattern. It doesn't — WeatherBench 2's `total_precipitation_24hr`
is NaN for any lead below 24h regardless of source (a 24-hour accumulation
isn't defined until 24h has elapsed), on top of the temperature-only
6h-index-floor constraint. The real, corrected pattern for
`total_precipitation_24hr` is **5 of 9 members at lead=24h, 7 of 9 at
lead=48h, the full 9 at lead=72h+** — not 6/9, 8/9, 9/9. This also
uncovered and fixed a related bug in `src/weavr/ensemble.py`'s
`build_lagged_ensemble`, which counted a member as valid whenever `.sel()`
succeeded without checking whether the returned data was itself all-NaN.
**The CRPS/Brier numbers above were not affected** — they were always
computed from the real fetched data via `skipna=True`-based reductions,
which already correctly excluded these NaN cells; only the member *count*
reported in code/docs was wrong, not any score.

### Precipitation: spread vs. skill

| Lead (h) | n members | Spread (mm) | RMSE of mean (mm) | Ratio | Calibrated target | Verdict |
|---|---|---|---|---|---|---|
| 24  | 5 | 2.31 | 11.09 | 4.79 | 1.095 | strongly under-dispersive |
| 48  | 7 | 2.31 | 12.43 | 5.37 | 1.069 | strongly under-dispersive |
| 72  | 9 | 2.81 | 11.31 | 4.02 | 1.054 | strongly under-dispersive |
| 96  | 9 | 2.29 | 12.64 | 5.51 | 1.054 | strongly under-dispersive |
| 120 | 9 | 2.19 | 14.98 | 6.85 | 1.054 | strongly under-dispersive |

The measured ratio is **4-7x the calibrated target at every lead** — this
lagged ensemble is severely under-dispersive, confirming the literature's
general expectation (Brenowitz et al.) in this specific data, not just
restating it. The spread itself (~2.2-2.8mm) barely changes with lead time
while the mean's actual error (RMSE, ~11-15mm) grows — a lagged ensemble
built from a single deterministic model's own short-range self-consistency
captures very little of the real day-to-day forecast uncertainty at any
lead, since all 5-9 members are the *same* model perturbed only by a few
hours of initialization time, not independent samples of the atmosphere's
own uncertainty. This is exactly why issue #4 flags this for "later
weighting decisions" — a Phase 3/4 combiner that treats this spread as a
genuine probability distribution without correcting for it (e.g. variance
inflation, or blending with genuinely independent NWP ensemble members)
would be significantly overconfident.

**Small-sample caveat applies here too**: this ratio is computed from the
same 3-4 test days per lead the CRPS/Brier numbers used — a rough,
indicative measurement of a real and severe effect, not a precise
calibration estimate.

### Temperature: spread magnitude only (no ground truth to compare against)

No matching-resolution IMD temperature observation exists (see "What was
scored, and what wasn't" above), so there is no error to compare spread
against — no ratio, no under/over-dispersion verdict is possible here.
These are raw spread magnitudes only, computed across all 18 nominal weeks
(no train/test split, since nothing is being scored):

| Lead (h) | GraphCast spread (K) | Pangu spread (K) |
|---|---|---|
| 24  | 0.297 | 0.440 |
| 48  | 0.308 | 0.446 |
| 72  | 0.308 | 0.455 |
| 96  | 0.317 | 0.447 |
| 120 | 0.344 | 0.462 |

Both are small fractions of a kelvin and barely grow with lead time —
consistent with the same qualitative pattern precipitation showed (a
lagged ensemble's spread stays roughly flat regardless of lead, since it
reflects a few hours of initialization offset, not the atmosphere's actual
growing forecast uncertainty) — but this is a descriptive observation, not
a calibration claim, since no ground truth exists to check it against.

## How to read these numbers relative to Tier 0

**This is not a direct comparison on the same metric.** Tier 0 reported
RMSE/bias/ACC/SEEPS/FSS/contingency scores for a deterministic forecast;
this reports CRPS/Brier for a probabilistic one — a new capability that did
not exist before, not an apples-to-apples improvement or regression against
Tier 0's numbers. A later phase (BMA/EMOS fitting, Phase 4) is where these
two families of scores get used together to judge whether a probabilistic
blend actually adds value over Tier 0's simple mean.

## Regenerating

```bash
python scripts/run_phase2_ensemble_baseline.py
```

Override the lagged-ensemble store, baseline store, output path, or test
fraction via `--lagged-store`, `--baseline-store`, `--out-csv`,
`--test-fraction`.
