# Step 06 data hand-off (issue #68): reply to the data request

Answers to the data request for the 2018 monsoon stores. Nothing here is
data: the stores stay out of Git (`data/` is gitignored). Only the small
manifest files are committed, in [`data-manifests/`](data-manifests/).

## 1. The 2020 daily stores

The project owner's machine has them. They are **not** in the repository and
cannot be pushed to GitHub (they are large and gitignored).

| Store | Size on disk | Status | Manifest (committed) |
|---|---|---|---|
| `data/baseline_2020_jjas_daily.zarr` | 230 MB | built, validated, 122 inits | [`baseline_2020_jjas_daily.manifest.json`](data-manifests/baseline_2020_jjas_daily.manifest.json) |
| `data/lagged_ensemble_inputs_2020_jjas_daily.zarr` | 753 MB | built, validated, 122 nominal inits | [`lagged_ensemble_inputs_2020_jjas_daily.manifest.json`](data-manifests/lagged_ensemble_inputs_2020_jjas_daily.manifest.json) |
| `data/ifs_ens_2020_jjas_daily.zarr` | none | **not built**, deliberately (about 126 GB pull, skipped by decision) | none |
| `data/ifs_ens_2020_jjas.zarr` (weekly, 18 inits, 50 members) | 277 MB | built; the store Tier 2's EMOS and BMA use | [`ifs_ens_2020_jjas.manifest.json`](data-manifests/ifs_ens_2020_jjas.manifest.json) |

The two daily stores total about 1 GB. **To get them to a teammate, copy the
two `.zarr` folders over a shared drive or a direct transfer.** GitHub is not
suitable. Until then, the implementation and `open_multi_season` can be
tested with synthetic stores; the real cross-season check waits on the copy.

## 2. Location and capacity for the 2018 build

- The forecast archives are public WeatherBench 2 data and IMD rainfall goes
  through the repo's existing `imd_gridded` client. No credentials are needed.
- **The full-member daily IFS-ENS store is not requested by default.** The
  same decision made for 2020 applies: about 126 GB of source chunks, several
  hours, and staging space on top. The owner's machine has about 70 GB free,
  which is not enough. Recommendation: **skip it for 2018 too**, and build
  the two stores that step 07 needs (`baseline_2018_jjas_daily.zarr`, which
  already includes `ifs_ens_mean`, and `lagged_ensemble_inputs_2018_jjas_daily.zarr`).
  Tier 2's EMOS-on-IFS-ENS and BMA then stay on the weekly 18-init store, as
  in 2020 (the "mixed cadence" caveat in `Improvements/prompts/completed/DONE.md`).
  If the team does want it, it needs an external or shared disk with roughly
  200 GB free, and an explicit go-ahead from the owner.
- Whoever runs the build must check free space on their own machine
  immediately before starting any large pull.
- Nobody has built or downloaded any 2018 store yet.

## 3. Conventions the 2018 build must match

From the 2020 manifests:

- Window 2020-06-01 to 2020-09-30 becomes **2018-06-01 to 2018-09-30**.
  Daily 00 UTC inits, leads 24, 48, 72, 96 and 120 h, on the existing India
  0.25° grid.
- **Lagged store model set: GraphCast (precipitation and temperature) and
  Pangu (temperature only; the archive has no precipitation variable).** The
  2018 store should match both, not only GraphCast.
- Lag scheme: 4 lags spaced 12 h (9 offsets); members per lead 6, 8, 9, 9, 9
  for 24, 48, 72, 96, 120 h. Batch size 50, 4 workers.
- Archive paths in the 2020 manifests:
  - GraphCast: `gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr`.
    **The 2018 path is different and not yet recorded**: find it and
    probe it (see the issue) before any long build.
  - HRES: `gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr` (covers 2018)
  - IFS-ENS mean: `gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr` (covers 2018)
  - Pangu: `gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr` (covers 2018)

## Known wart

`_meta.init_cadence_days` in the lagged store's manifest records 7 (the
default flag) even though it was built from the daily baseline store
(`nominal_times_from` records that).
