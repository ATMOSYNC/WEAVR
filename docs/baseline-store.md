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
rather than assuming recency. Only the second window overlaps a full
monsoon season. **2020 JJAS (June-September) is therefore the only year**
where GraphCast, Pangu, HRES, and IFS ENS all have data simultaneously.

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
  only at the two shortest lead times.** Both archives' `prediction_timedelta`
  axis starts at 6h (no 0h or negative lead exists). A lagged member
  initialized *after* the nominal init time needs a *shorter* lead to reach
  the same valid time; at nominal lead 24h, offsets of +36h/+48h would need
  leads of -12h/-24h, which don't exist. Measured directly: **lead=24h gets
  6 of 9 members, lead=48h gets 8 of 9, and lead=72h/96h/120h all get the
  full 9** — not padded with a substitute value, left as fewer real members.
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
