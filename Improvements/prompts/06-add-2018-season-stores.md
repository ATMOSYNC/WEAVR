# Step 06: Add the 2018 monsoon season

**Type**: data-engineering prompt (probe, then build; long fetches in the
background).
**Plan items**: A2 (data half). Step 07 does the evaluation half.
**Depends on**: step 05 (it reuses step 05's parametrised, deduplicated,
parallel builders).

## Goal

Build daily-cadence JJAS **2018** stores (GraphCast, HRES, IFS-ENS, the
lagged GraphCast ensemble and IMD) matching step 05's 2020 stores. That
gives WEAVR two independent seasons, which unlocks leave-one-year-out
cross-validation (the phase plan's original design) and the Kerala 2018
replay (step 19).

## Why this is possible, when the docs said it wasn't

The plan's F5 checked the live bucket:

- `graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours_derived.zarr`
  has `total_precipitation_24hr`, 884 inits (2017-11-16 → 2019-01-31) at
  0.25°, leads 6–240 h.
- It uses **`lat`/`lon`** dimension names, not `latitude`/`longitude`.
  That is probably why it was dismissed; the builder's slicing assumes the
  latter.
- HRES (2016-01-01 → 2023-01-10) and IFS-ENS (2016–2024) cover 2018 in the
  same archives the builders already use.
- Pangu's `2018-2022` store covers 2018 too. It has temperature only; keep
  it for step 24.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Probe before building. Open the GraphCast 2018 derived store lazily and
   record: dimension names, coordinate directions (the descending-latitude
   gotcha that _lat_slice_for() handles), longitude convention (0-360?),
   units (expect metres), the prediction_timedelta dtype, and the chunk
   shape. Time one chunk fetch.
   Compare against the 2020 store the builder already handles, and list
   every difference in the PR description. Check the store's attrs and
   WeatherBench 2's docs for which GraphCast checkpoint each window used.
   A model-version change between 2018 and 2020 is expected; document it.
   It also becomes a natural test for weavr.drift in step 07.

2. Generalise the builders so the archive path is chosen per (source, year)
   through one small, tested mapping, e.g. ARCHIVES[(source, year)] -> path
   plus a coordinate-rename dict. Apply {"lat": "latitude", "lon":
   "longitude"} before slicing and regrid_to_common. The existing
   regrid/slice guards (SourceTooCoarseError, the latitude-direction
   check) must still run. Add unit tests for the mapping and the rename on
   a synthetic dataset with lat/lon names.

3. Storage layout decision (decide it, don't ask; state the reasoning in
   docs):
   - Recommended: per-season stores (data/baseline_2018_jjas_daily.zarr,
     etc.), plus a pure loader src/weavr/stores.py:
     open_multi_season(paths, group) that concatenates along time and
     verifies identical lat/lon grids.
   - Reason: each season builds idempotently with its own manifest,
     avoiding Zarr append edge cases.
   - If you choose a single combined store instead, justify it.
   Test the loader on two synthetic stores, including a grid mismatch that
   must raise.

4. Build, in the background, with step 05's measured-cost discipline
   (measure, extrapolate, write it down, then launch):
   - data/baseline_2018_jjas_daily.zarr (graphcast, hres, ifs_ens_mean,
     pangu temperature, imd_observed)
   - data/lagged_ensemble_inputs_2018_jjas_daily.zarr
   - data/ifs_ens_2018_jjas_daily.zarr
   GraphCast's 2018 archive starts 2017-11-16, so the +/-48 h lag window
   around JJAS is available; verify this rather than assume it.
   IMD 2018 is in data/imd_seeps_climatology_jjas.zarr (group y2018), or
   fetch it with weavr.data.imd_gridded.fetch_year. Use the same source the
   2020 store used, for consistency.

5. Validate each store the same way step 05 did (init counts, NaNs, leads,
   members) and do one cross-season sanity check: domain-mean JJAS rainfall
   for 2018 vs 2020, forecasts vs IMD, is plausible (no factor-of-1000 unit
   slip).

6. Docs: in docs/baseline-store.md, add a "Second season (2018)" section
   with the probe findings, the archive map and the measured costs, and
   point back to the step 01 correction. README paragraph. One PR.
```

## Done when

- The 2018 daily stores exist and are validated.
- The per-(source, year) archive map and `weavr.stores.open_multi_season`
  are tested.
- The GraphCast checkpoint difference between years is documented.
- Step 07 can load 2018 + 2020 as one multi-season dataset.
