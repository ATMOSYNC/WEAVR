# Baseline store (Phase 0 exit criterion)

Built by [`scripts/build_baseline_store.py`](../scripts/build_baseline_store.py).
This is the literal exit criterion for Phase 0: one full monsoon season of
aligned forecasts + IMD observations sitting in one Zarr store.

## Year: 2020, not "most recent"

The step-5 prompt suggested "the most recent complete year, e.g. 2024 or
2025." That turned out to be wrong once checked against what's actually in
the WeatherBench 2 bucket: GraphCast's public archive there only has **two**
evaluation windows — `2017-11-16..2019-02-01` and `2019-11-16..2021-02-01` —
confirmed by listing `gs://weatherbench2/datasets/graphcast/` directly
rather than assuming recency. 2020 JJAS (June-September) is the season this
store was built for.

### Correction, 2026-09-29: 2020 is not the only usable season

This section previously said "Only the second window overlaps a full monsoon
season" and concluded that 2020 JJAS was "**the only year** where GraphCast,
Pangu, HRES, and IFS ENS all have data simultaneously." **Both statements
are wrong**, and they were wrong in a way that cost the project a second
season of training data — the whole reason Phase 3 could never use
`leave_one_year_out`.

Re-checked live against the bucket, not inferred from the window's start
date:

```
gs://weatherbench2/datasets/graphcast/2018/
  date_range_2017-11-16_2019-02-01_12_hours_derived.zarr
```

- `total_precipitation_24hr` **is present**, with dims
  `(time, prediction_timedelta, lat, lon)`.
- It covers **243 initialisations from 2018-06-01 to 2018-09-30** at a
  12-hourly cadence — the whole of JJAS 2018, not a fragment of it.
- The values are real: 2018-08-15 00Z + 24 h over the India domain is
  100% finite, with a maximum of 168.9 mm (the Kerala floods week).
- Lead times run from 6 h in 6-hourly steps.

The 2018 derived store names its spatial dimensions **`lat`/`lon`**. A live
recheck of the 2020 derived path used by this repository found `lat`/`lon`
there too, correcting the earlier claim that it used `latitude`/`longitude`.
The builders normalize these names before geographic slicing and regridding.

The other three sources cover JJAS 2018 as well — checked the same way, all
with **243 initialisations** over 2018-06-01..2018-09-30:

| Source | Archive | JJAS 2018 |
|---|---|---|
| `graphcast` | `graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours_derived.zarr` | 243 inits, has `total_precipitation_24hr` |
| `pangu` | `pangu/2018-2022_0012_0p25.zarr` | 243 inits, temperature only (no precipitation anywhere in this archive, as in 2020) |
| `hres` | `hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr` | 243 inits, has `total_precipitation_24hr` |
| `ifs_ens` | `ifs_ens/2018-2022-1440x721.zarr` | 243 inits, has `total_precipitation_24hr` |

So a 2018 JJAS store is buildable with exactly the same source set as 2020,
and a two-season store would make `leave_one_year_out` usable for the first
time.

Nothing is built for 2018 here; this note only corrects the record so the
2018 season isn't ruled out a second time on the strength of a sentence.

## What's in the store

One Zarr store (`data/baseline_2020_jjas.zarr`, not committed — see
`.gitignore`), with one group per source:

| Group | Variables | Native resolution | Notes |
|---|---|---|---|
| `graphcast` | `2m_temperature`, `total_precipitation_24hr` | 0.25° (native) | uses the `_derived.zarr` store, which has 24hr-accumulated precip pre-computed by the WeatherBench 2 authors |
| `pangu` | `2m_temperature` only | 0.25° (native) | **no precipitation variable exists in this archive at all** — not zero-filled, genuinely absent |
| `hres` | `2m_temperature`, `total_precipitation_24hr` | 0.25° (native) | |
| `ifs_ens_mean` | `2m_temperature`, `total_precipitation_24hr` | 0.25° (native) | ensemble **mean**, not the full 50-member ensemble — chosen to keep store size reasonable for a baseline; full-ensemble access is a Phase 1+ decision if BMA/EMOS needs individual members |
| `imd_observed` | `rain` | 0.25° (native) | ground truth, via `imdlib` |

All groups are regridded through `weavr.grid.regrid_to_common` onto the
locked 0.25° India grid. Since every source above is already natively
0.25° (confirmed by inspection, not assumed), the regrid step is mostly a
validation pass — it's still routed through the shared function rather than
skipped, so a future source at a different resolution is caught by
`SourceTooCoarseError` instead of silently mismatching.

## A real gotcha this caught

**Latitude direction differs by source.** GraphCast, HRES, and IFS ENS all
have ascending latitude (`-90..90`); **Pangu's native store has descending
latitude (`90..-90`)**. `xarray`'s `.sel(latitude=slice(lo, hi))` does not
error on a descending coordinate — it silently returns zero points. This
was caught by a smoke test before the full run (Pangu's slice produced an
empty array, which then failed downstream with an unrelated-looking numpy
reshape error). Fixed in `_lat_slice_for()` in the script, which checks the
coordinate's own direction and swaps the slice bounds accordingly. Covered
by a unit test.

**Zarr v2-era encoding breaks the zarr-python v3 writer.** Reading a
WeatherBench 2 store returns per-variable `.encoding` carrying the
*original full-grid* chunk shape and a `numcodecs.Blosc` compressor spec
(zarr v2 style). Writing that straight back out via the installed
zarr-python v3 fails with `Expected a BytesBytesCodec. Got
numcodecs.blosc.Blosc`, and would in any case apply full-globe chunk sizes
to a much smaller India-sliced array. Fixed by clearing `.encoding` on every
variable/coord before `to_zarr()`, letting the writer pick fresh chunking
and a v3-native codec. Covered by a unit test.

## Why forecast data isn't run through `resample_to_imd_day`

`weavr.grid.resample_to_imd_day` is applied to `imd_observed` for
consistency, but IMD's own product from `imdlib` is already daily on IMD's
own convention — applying it there is a correctness check, not real
resampling work.

Forecast data is intentionally **not** run through it. That function
resamples a plain, continuous `time` axis; forecast data here is indexed by
`(init_time, prediction_timedelta)`, and each lead's actual valid time
(`init_time + prediction_timedelta`) needs to be bucketed into an IMD day —
a genuinely different problem (matching forecast valid-time to observation
day) than resampling a continuous series. Forcing `resample_to_imd_day`
onto the wrong axis here would have produced a result that looked correct
but wasn't. This valid-time-to-IMD-day bucketing is real work, left
explicitly for Phase 1's verification harness, where it belongs alongside
the scoring logic that actually needs it.

## Idempotency

The script writes `data/baseline_2020_jjas.manifest.json` alongside the
store, recording per-group status. Re-running skips any group already
marked `"ok"` unless `--force` is passed — a partial failure (e.g. one
source's GCS request timing out) doesn't force re-downloading groups that
already succeeded.

## Regenerating

```bash
python scripts/build_baseline_store.py \
    --start 2020-06-01 --end 2020-09-30 --max-lead-hours 120
```

Override the output location with `--out` or `WEAVR_BASELINE_STORE_PATH`.

## Phase 1 addition: SEEPS/ACC climatology archive

This store alone isn't enough for every Phase 1 verification metric — see
`docs/phase-1-data-requirements.md` for the full, checked-against-real-data
breakdown. Two things came out of that check:

- **SEEPS and precipitation ACC** need a multi-year per-gridpoint
  climatology, which one JJAS season can't provide. Built separately by
  `scripts/build_seeps_climatology.py` into
  `data/imd_seeps_climatology_jjas.zarr` (IMD-only, JJAS 2006-2020, 15
  years) — a separate script and store because the shape of what it builds
  (many years of one variable, no forecast pairing) doesn't fit this
  store's per-source-group layout. IMD-only pulls have no chunk-latency
  problem, so extending this dimension cost no tractability tradeoff.
- **CRPS/Brier and ACC-for-temperature** were checked and found to need,
  respectively, a ~2-2.5 hour/~30-35GB full-ensemble pull and a separate
  native-resolution climatology pipeline — both deliberately deferred
  rather than pulled now (documented in `docs/phase-1-data-requirements.md`
  with the reasoning). This store's `ifs_ens_mean` group therefore still
  cannot support CRPS/Brier; that isn't a bug to fix here, it's a known,
  documented limitation until Phase 2.

## Phase 2 addition: lagged-ensemble input store

Built separately by
[`scripts/build_lagged_ensemble_store.py`](../scripts/build_lagged_ensemble_store.py)
into `data/lagged_ensemble_inputs_2020_jjas.zarr` — a separate script and
store, not an extension of `build_baseline_store.py`, because the shape of
what it builds (a per-nominal-forecast cluster of extra init times, each
queried at a lead adjusted to land on a shared valid time) doesn't fit this
store's flat `(time, prediction_timedelta)` sampling grid without
distortion — the same reasoning that made `build_seeps_climatology.py` a
separate script in Phase 1.

**Scope: GraphCast and Pangu only.** These are the phase-plan's named "AI
models" — the ones with a single deterministic forecast per init time and
no ensemble spread of their own. HRES and `ifs_ens_mean` are untouched.

**Checked, not assumed, live against the real GCS archives:**

- Both GraphCast's and Pangu's WeatherBench 2 stores are confirmed
  12-hourly at native resolution (listing the actual zarr time index, not
  trusting the module docstring's claim).
- Both archives span well outside the ±48h window needed around every one
  of Phase 1's 18 weekly JJAS-2020 sample points — GraphCast covers
  2019-11-16..2021-01-31, Pangu covers 2018-01-01..2022-12-31. **No sample
  point runs off either archive's edge in the time dimension.**
- **Member availability is limited by lead range, not by archive edge, and
  only at the shortest lead times — differently per variable.** Both
  archives' `prediction_timedelta` axis starts at 6h (no 0h or negative lead
  exists). A lagged member initialized *after* the nominal init time needs a
  *shorter* lead to reach the same valid time; at nominal lead 24h, offsets
  of +36h/+48h would need leads of -12h/-24h, which don't exist for either
  variable. This alone gives **`2m_temperature`: lead=24h gets 6 of 9
  members, lead=48h gets 8 of 9, lead=72h+ gets the full 9.**
  **`total_precipitation_24hr` has a second, stricter constraint**, found
  only by checking the actual fetched *values* (not assumed to match
  temperature's pattern, which was the original mistake here — see the
  "second real bug" note below): a 24-hour precipitation accumulation is
  NaN in WeatherBench 2's own data for any lead below 24h, even though the
  index entry exists at every 6h step. This gives
  **`total_precipitation_24hr`: lead=24h gets 5 of 9 members, lead=48h gets
  7 of 9, lead=72h+ gets the full 9** — not padded with a substitute value,
  left as fewer real members.
- **Fetch cost, measured live**: a single-chunk fetch measured ~2.3-2.6s for
  both sources (consistent with Phase 0's ~0.75-2s estimate for this chunk
  shape). The exact set of needed (nominal_time, lead, offset) combinations
  — not a cartesian time×lead superset, which would have cost ~3x more — is
  2214 chunk fetches total across both sources' variables. Short of the
  ~2-2.5h/~30-35GB IFS full-ensemble pull Phase 1 deferred — tractable to
  just run, not a scope-cutting tradeoff worth interrupting for.

**A real bug found running the full pull, not just estimated:** the first
implementation issued a single vectorized `xr.Dataset.sel()` call covering
all ~1476 (time, lead) pairs for a source at once. On the real run (not the
small smoke test, which stayed under the threshold that triggers this) that
call **stalled indefinitely** — confirmed live, not assumed: network
byte-counters (`nettop`) sat completely flat for 45+ seconds while the
process was still alive, and a stack sample (`sample`) showed the async I/O
thread parked in a socket wait (`kevent`) with the main thread blocked
waiting on it, i.e. no forward progress, not just a slow request. Likely a
pathological interaction between zarr v3's async I/O layer and a very large
single vectorized selection under heavy concurrent load. **Fixed** by
batching the fetch into one smaller vectorized `.sel()`/`.load()` call per
nominal week (~41 combos each) instead of one call for all ~738 combos —
verified the fix on a 4-week smoke test (bytes-in climbing steadily, no
stall) before re-running the full pull, which then completed well inside
20 minutes for both sources combined — comfortably under the naive
~92-minute serial estimate, since per-week batching gives real, stable
concurrency instead of either a stall or an unparallelized serial crawl.

**A second real bug, found later while implementing Phase 2 step 4's
dispersion diagnostics, not during this step's original execution:** this
doc originally claimed `total_precipitation_24hr` followed the same 6/9,
8/9, 9/9 pattern as `2m_temperature`, because the original verification
checked the NaN pattern using `2m_temperature` as a stand-in for "the
graphcast group" and never re-checked it against the precipitation
variable specifically. Measuring the real dispersion numbers for
precipitation surfaced a mismatch (5 members counted, not the expected 6)
that traced back to WeatherBench 2's `total_precipitation_24hr` being NaN
below 24h lead (see above) — a genuinely different, stricter constraint
this doc had silently conflated with temperature's. This also uncovered a
related bug in `src/weavr/ensemble.py`'s `build_lagged_ensemble`, which
counted a member as valid whenever `.sel()` didn't raise, without checking
whether the returned data was itself all-NaN — fixed there, with a
regression test (`tests/test_ensemble.py::TestBuildLaggedEnsembleAllNanValues`)
using a synthetic source that reproduces this exact index-exists-but-data-
is-NaN shape. The store's own fetched *data* was never wrong (it correctly
contains real NaNs at these cells); only this doc's and `ensemble.py`'s
*description/count* of validity was.

Store layout: one group per source (`graphcast`, `pangu`), each indexed by
`(nominal_time, lead_hours, member_offset_hours, latitude, longitude)` —
`nominal_time`/`lead_hours` match Phase 1's existing weekly sample points
and lead hours exactly, `member_offset_hours` is
`[-48, -36, -24, -12, 0, 12, 24, 36, 48]`. Cells with no valid lead in the
source (the lead=24h/48h gaps above) are `NaN`, not a fetch failure.

## Regenerating the lagged-ensemble input store

```bash
python scripts/build_lagged_ensemble_store.py
```

Override the output location, window, lead hours, init cadence, or lag
parameters via `--out`, `--start`/`--end`, `--lead-hours`,
`--init-cadence-days`, `--n-lags`, `--lag-spacing-hours`.

## Phase 4 addition: the real IFS 50-member ensemble store

This row's own earlier note ("full-ensemble access is a Phase 1+ decision
if BMA/EMOS needs individual members") is now exercised: Phase 4's
EMOS-CSG/BMA combiners need a real per-member ensemble spread, not the
collapsed `ifs_ens_mean` field, so `scripts/build_ifs_ensemble_store.py`
pulls WeatherBench 2's real, uncollapsed 50-member IFS ensemble
(`gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr`) for
`total_precipitation_24hr` only, at exactly `data/baseline_2020_jjas.zarr`'s
own 18 weekly JJAS-2020 timestamps (read directly from that store, not
re-derived) and 5 lead hours, into `data/ifs_ens_2020_jjas.zarr`
(`(time, member, prediction_timedelta, latitude, longitude)`, `member` 1-50,
renamed from the raw archive's own `number` coordinate for consistency with
`weavr.verify`'s `member_dim="member"` convention).

**Real cost, measured live**: each `(time, prediction_timedelta)` pair is
one native Zarr chunk bundling all 50 members and the full global grid
(confirmed directly: chunk shape `(1, 50, 1, 721, 1440)`) — a single-chunk
probe measured **~34s and ~208MB** for `total_precipitation_24hr` alone.
Fetching one variable only (not the 2-variable estimate
`docs/phase-1-data-requirements.md` originally scoped) puts real cost at
**18 timestamps × 5 leads = 90 chunks, ~183.8s/timestamp average, ~55
minutes of real GCS transfer time, ~277MB on disk** (Zarr compression
shrinks this well below the raw ~19GB transferred) — smaller than the
original ~2-2.5h/~30-35GB two-variable estimate, as expected.

**A real operational problem found running the full pull, not estimated:**
this machine's idle-sleep silently killed the fetch's live GCS connections
three separate times mid-run (confirmed via `nettop`: `bytes_in` frozen
byte-for-byte across repeated checks, not just slow) — the script's own
resumability (each successful timestamp cached to a small per-timestamp
NetCDF file in `data/ifs_ens_2020_jjas.zarr.staging/`, with the manifest
skipping already-fetched timestamps on retry) meant each stall only cost
the one in-flight timestamp's progress, not the whole run, and a real
keep-awake hold requested partway through stopped it recurring. No
production code was affected by this — it's an operational note about
running a multi-minute unattended fetch on a laptop, not a bug in the
fetch logic itself, but worth recording since it will recur for any
similarly long-running local fetch this project runs later.

Every timestamp fetched cleanly (no all-NaN cells, no missing members) —
unlike GraphCast's lagged pseudo-ensemble, the real IFS archive has no
lead-range gap to produce fewer-than-full members at short leads.

```bash
python scripts/build_ifs_ensemble_store.py
```

Override the output location, baseline store to align against, or lead
hours via `--out`, `--baseline-store`, `--lead-hours`. Idempotent and
resumable: a failed or interrupted run can simply be re-invoked, and
already-fetched timestamps (cached in the `.staging` directory next to the
output store) are skipped, not re-fetched.

## Phase 6 addition: model-version metadata in every manifest

Checked before writing any drift-detection logic (`solving issues/
07-phase-6-robustness-and-operations/03-implement-rolling-verification-and-drift-detection.md`'s
own step 1): none of this store's, the lagged-ensemble store's, or the IFS
ensemble store's manifests recorded which archive vintage was actually
pulled for a source — two different WeatherBench 2 archive updates of the
same named source (e.g. `graphcast`) would have looked identical in the
manifest. WeatherBench 2 has no separate version number; the closest real
identifier is the archive's own `zarr_path`, which changes when WB2
republishes a model under a new date-range/path. **Fixed**: every forecast
source's manifest entry in this store and the lagged-ensemble store now
also records `source_archive_path` (`build_baseline_store.py`'s
`build_forecast_group`, `build_lagged_ensemble_store.py`'s own per-source
build function); `build_ifs_ensemble_store.py`'s manifest (a single source,
indexed per-timestamp rather than per-source-group) records the same thing
once at the manifest's top level, `_source_archive_path`. `imd_observed`
is unchanged — it's ground truth, not a forecast model, so "model version"
doesn't apply to it.

This alone isn't a live "model upgraded" trigger — these three scripts
build the frozen 2020 JJAS research stores, which will never see a version
change again. See [src/weavr/drift.py](../src/weavr/drift.py)'s own module
docstring for why the live daily pipeline's actual drift-detection hook
(covering AIFS/IFS/HRES, per `docs/phase6-operational-scope.md`) turned
out to have no explicit version signal to hook either, and uses a real
statistical drift check instead.

## Daily cadence addition (2020)

Step 05 of the improvement plan
(`Improvements/prompts/completed/05-daily-cadence-2020-stores.md`).

The Phase 0/1 stores sample JJAS 2020 **weekly** -- 18 initialisations --
because `build_baseline_store.py` feared "every 12-hourly init x every
6-hourly lead" would stall. But WEAVR only ever scores **5 lead times**, so
daily 00 UTC sampling is 122 x 5 = 610 chunk fetches per source-variable,
not thousands. This section adds **daily** stores alongside the weekly ones,
which are left untouched so every existing result stays reproducible.

Why it matters: step 04 measured that with 3-4 test days per lead, **not one
of 1,691 method comparisons had a computable confidence interval**
(`docs/scorecard-and-significance.md`). 18 -> 122 samples is the fix.

### What was built

| Store | Status |
|---|---|
| `data/baseline_2020_jjas_daily.zarr` | **built**, 122 inits, validated |
| `data/lagged_ensemble_inputs_2020_jjas_daily.zarr` | **built**, 122 nominal inits, validated |
| `data/ifs_ens_2020_jjas_daily.zarr` | **not built** -- see below |

All three builders are parametrised, deduplicated and tested here; the data
itself is gitignored, so the IFS-ENS store can be produced at any time by
running the command below without further code changes.

### Measured fetch costs

Timed live against the real archives before launching anything long
(one chunk = one init time at one lead, the archive's own chunk granularity):

| Source | s/chunk | Chunk size | 610 chunks, serial |
|---|---|---|---|
| GraphCast `total_precipitation_24hr` | 2.57 | 4.2 MB | 0.44 h |
| HRES `total_precipitation_24hr` | 2.07 | 4.2 MB | 0.35 h |
| IFS-ENS `total_precipitation_24hr` (50 members) | 33.92 | **207.6 MB** | **5.75 h** |

**The builds must run one at a time.** Running the lagged and IFS-ENS builds
concurrently was tried and measured: the lagged build's batch time degraded
from 57 s to 97 s to **331 s**, and the IFS-ENS build completed zero
timestamps in 12 minutes. Stopping IFS-ENS restored the lagged build to 57 s
per batch immediately. The bottleneck is shared bandwidth, so running two
fetches in parallel is strictly worse than running them in sequence.

### Baseline store: 00 UTC cycle, batched loads

- Built with `--init-cadence-days 1` into a new `--out`; the weekly store is
  untouched.
- **00 UTC only.** The source archives are 12-hourly, so a daily stride must
  pick one cycle or it would alternate 00/12 UTC and silently mix two
  different forecast cycles into one series. `_sample_init_times` filters to
  00 UTC before striding, and the manifest records
  `"init_cadence_days": 1` and `"init_cycle": "00:00 UTC"`.
- **Batched loads.** 122 init times are loaded 25 at a time rather than in
  one call; the single-call form that worked for 18 weekly inits did not
  survive 122.
- Validated: all five groups carry 122 daily steps spanning
  2020-06-01..2020-09-30 with **zero gaps**, every variable finite, no
  all-NaN days.

### Lagged-ensemble store: cross-day deduplication

Each nominal init needs +/-4 lags at 12 h spacing. With *daily* nominal
inits those overlap heavily between neighbouring days: nominal day T at lead
48 h with offset +24 h targets exactly the same `(source_time, source_lead)`
pair as nominal day T+1 at lead 24 h with offset 0.

Deduplicating before fetching, rather than after, is what makes the daily
build affordable:

- 122 nominal dates x 41 valid combos = **5,002 combos**
- unique `(source_time, source_lead)` pairs = **1,735**
- **65.3% fewer fetches**

Without it the build would download the same chunk up to three times.
`tests/test_build_lagged_ensemble_store.py` pins the property directly:
overlapping lags between consecutive days must produce one fetch each.

Member availability must reproduce the weekly store's documented structure --
short leads legitimately carry fewer members, because a lagged member needs a
*shorter* source lead to reach the same valid time, and a 24 h accumulation
is undefined below 24 h:

- `2m_temperature`: 6 / 8 / 9 / 9 / 9 members at leads 24 / 48 / 72 / 96 / 120 h
- `total_precipitation_24hr`: 5 / 7 / 9 / 9 / 9

`scripts/validate_daily_stores.py` checks exactly this against the built
store, and fails if it does not hold.

Measured cost, from the real run that produced this store: **57 s per 50-pair
batch**, 35 batches per source, two sources -- about **70 minutes** end to
end. Validated: 122 nominal inits over 2020-06-01..2020-09-30 with zero gaps,
no all-NaN samples, and member counts matching the documented pattern exactly
at every lead. Rebuild it with:

```bash
python scripts/build_lagged_ensemble_store.py \
    --out data/lagged_ensemble_inputs_2020_jjas_daily.zarr \
    --nominal-times-from data/baseline_2020_jjas_daily.zarr --workers 4
```

### IFS-ENS daily store: deliberately not built

`data/ifs_ens_2020_jjas_daily.zarr` **does not exist**, by decision, not by
failure.

The archive chunks this variable as `(1, 50, 1, 721, 1440)` -- **all 50
members in a single chunk**. Fetching 122 inits x 5 leads therefore means
downloading **126 GB** of full-global-grid slabs to keep a ~2 GB India
subset, and at the measured 6.1 MB/s single-stream throughput that is
between 3.5 and 9.5 hours.

Two consequences worth recording, because both are counter-intuitive:

- **Subsetting members does not help.** A "fetch only 20 of 50 members"
  fallback saves storage but *no download time at all*, because the 50
  members arrive in one indivisible chunk. The only lever that reduces the
  download is fewer init times.
- **More workers may not help either.** The cost is bandwidth, not latency,
  so a worker pool splits the same pipe rather than widening it.

What this costs downstream: the daily baseline store still contains
`ifs_ens_mean`, so **Tier 0, Tier 1 and the single-source baselines get the
full 122 days**. Only Tier 2's EMOS-on-IFS-ENS and the BMA combiner need the
50 individual members, and those continue to use the weekly
`data/ifs_ens_2020_jjas.zarr` (18 inits) until someone runs the daily pull.
Step 07 must state that mixed cadence explicitly wherever it reports a Tier 2
number, rather than letting a reader assume every tier had 122 days.

The builder is fully parametrised and ready whenever the pull is wanted:

```bash
python scripts/build_ifs_ensemble_store.py \
    --out data/ifs_ens_2020_jjas_daily.zarr \
    --init-times-from data/baseline_2020_jjas_daily.zarr \
    --workers 4
```

`--init-times-from` replaced the previously hard-coded 18 weekly timestamps
(the old default is preserved, so the weekly build stays reproducible), and
each timestamp is fetched under a 300 s timeout with up to 3 retries and
written atomically into `.staging/`, so an interrupted run resumes without
re-fetching.

### Validating

```bash
python scripts/validate_daily_stores.py --no-require-ifs-ens
```

`scripts/validate_daily_stores.py` fails on -- not merely prints -- init
count, season span, gaps, leads, all-NaN variables, all-NaN days, lagged
member counts per lead, and IFS-ENS member count. The member-count check
takes the **minimum across every nominal time** rather than sampling a few,
since a partial fetch is precisely the failure that sampling would miss.
`--no-require-ifs-ens` records the skip above as a documented decision
instead of a failure.

## Second season (2018)

Issue #68 adds JJAS 2018 as a separate, resumable season. Per-season Zarr
stores keep their own manifests; appending another year into a live Zarr
store would make retries and partial failures harder to audit.
`weavr.stores.open_multi_season(paths, group)` checks the latitude and
longitude grids and concatenates matching source groups in chronological
order. It rejects overlapping dates and incompatible variables.

### Live archive probe

The 2018 GraphCast derived archive was opened lazily before downloading a
season. The 2020 derived archive was opened for comparison. Both have
`(time, prediction_timedelta, lat, lon)` for precipitation, ascending
latitude from -90 to 90 degrees, longitude from 0 to 359.75 degrees, integer
lead values with `units=hours`, and one full-global-grid chunk per
initialization and lead (`1 x 1 x 721 x 1440`, about 4.2 MB raw). The 2018
archive has 884 total initializations; the 2020 archive has 886. The derived
precipitation variable and dataset have no `units` attribute in either
archive. A 2018-08-15 00 UTC, +24 h India slice loaded in **2.53 seconds**;
its maximum was 0.16893 m (168.9 mm). Precipitation is treated as metres,
following the WeatherBench 2 convention already used by the verification
code, and multiplied by 1000 when compared with IMD millimetres.

WeatherBench 2's [data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#graphcast)
states that the **2018 forecasts use a GraphCast model trained through
2017**, while the **2020 forecasts use a model trained through 2019**. This
is a model checkpoint change, not merely a different evaluation window.
The source archives and coordinate renames are selected by
`weavr.archives.ARCHIVES[(source, year)]`; the 2018 GraphCast path is
`graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours_derived.zarr`.
HRES, IFS-ENS mean, and Pangu use the same multi-year archives in both
seasons. Pangu remains temperature-only.

### Build and validation

```bash
python scripts/build_baseline_store.py --year 2018
python scripts/build_lagged_ensemble_store.py --year 2018 --workers 4
python scripts/validate_daily_stores.py --year 2018 --no-require-ifs-ens
```

The official IMD binary endpoint timed out during this build. The yearly
0.25-degree IMD rainfall NetCDF was taken from the [public Zenodo archive](https://zenodo.org/records/11195106),
whose full ZIP matched its published MD5
`cda7001f29fe8480a56d05026362ff3f`. The extracted
`RF25_ind2018_rfp25.nc` has 365 daily records, the expected 129 x 135
India grid, rainfall in millimetres, and missing values decoded as NaN.
The baseline builder accepts that yearly file explicitly:

```bash
python scripts/build_baseline_store.py --year 2018 \
    --imd-nc-path data/imd_cache/RF25_ind2018_rfp25.nc
```

The five 2018 baseline groups built successfully into a 228 MiB store.
Validation confirmed 122 daily initializations or observation days from
June 1 through September 30, no gaps, all five requested leads for each
forecast, no all-NaN sample, and finite IMD land rainfall on every day.
The IMD group has 26.6% finite cells on the full rectangular grid, reflecting
the India land mask.

For a real cross-season units check, the +24 h GraphCast daily rainfall was
converted from metres to millimetres and averaged over cells with IMD land
coverage. All 122 JJAS days were used in each year (the 2020 GraphCast chunks
were read from the public archive because the owner's 2020 Zarr store is not
in this checkout). In 2018, GraphCast averaged **7.324 mm/day** against IMD
**6.532 mm/day** (ratio 1.121); in 2020, GraphCast averaged **8.187 mm/day**
against IMD **7.844 mm/day** (ratio 1.044). Both are on the expected scale,
with no factor-of-1000 unit discrepancy. This is a magnitude sanity check,
not a skill score: the 00 UTC forecast and IMD's 03 UTC observation-day
windows are not exactly aligned.

The daily 00 UTC design uses 122 initializations (June 1 to September 30)
and leads 24, 48, 72, 96, and 120 h. GraphCast's earlier archive begins in
November 2017, so the +/-48 h lag window around JJAS 2018 lies inside it.
The lagged store includes GraphCast precipitation and temperature and Pangu
temperature, matching the 2020 daily manifest. The first measured GraphCast
chunk took 2.53 s; a serial extrapolation for its 610 daily
initialization/lead chunks is about 26 minutes per variable. The builders
batch and overlap requests. Each 2018 lagged source deduplicated 5,002
nominal lag/lead combinations to 1,735 unique pairs across 35 batches.
GraphCast fetched 3,470 variable chunks in 18.3 minutes; Pangu fetched
1,735 temperature chunks in 10.1 minutes. The completed lagged store is
749 MiB. The validator confirmed 122 nominal initializations with no gaps,
no all-NaN samples, and temperature member counts of 6/8/9/9/9 for
24/48/72/96/120 h. GraphCast precipitation has 5/7/9/9/9 members at those
leads, matching the documented accumulation constraint. Both stores'
provenance and counts are saved in `docs/data-manifests/`.

The [step 06 data handoff](step-06-data-handoff.md) confirms that the
2020 full-member daily IFS-ENS store was deliberately skipped and applies
the same decision to 2018. The IFS-ENS **mean** remains present daily in the
baseline store. The 50-member daily builder can be run with `--year 2018`
if a suitable data location is provided later. Tier 2's member-based
comparison therefore retains its documented weekly cadence.
