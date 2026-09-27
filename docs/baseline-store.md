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
