# Step 05: Daily-cadence stores for JJAS 2020

**Type**: data-engineering prompt, with long-running fetches (resumable, in
the background).
**Plan items**: A1.
**Depends on**: nothing in code. Start its fetches as early as possible;
steps 02–04 can proceed while they run.

## Goal

Rebuild the 2020 stores at **daily** init cadence (00 UTC, June–September,
about 122 inits) instead of weekly (18 inits). Covers the baseline store,
the lagged GraphCast ensemble and the real IFS 50-member ensemble.

Build them as **new** stores next to the weekly ones, so every existing
result stays reproducible.

## Why this is cheaper than the docstring feared

- `scripts/build_baseline_store.py` chose weekly sampling because "every
  12-hourly init × every 6-hourly lead ≈ 4,880 chunk fetches" stalled. But
  WEAVR only uses **5 leads**. Daily 00 UTC is 122 × 5 = **610 chunk fetches
  per source per variable**, about 8–20 minutes at the measured 0.75–2 s per
  chunk. The `--init-cadence-days` flag already exists.
- `build_ifs_ensemble_store.py` measured about **47 s per chunk**, so 610
  chunks is roughly 8 h serially. Its staging/resume logic already exists;
  add a small worker pool.
- `build_lagged_ensemble_store.py` needs extra (init, lead) pairs around
  each nominal init. With daily nominal inits and 12 h-spaced lags, most
  pairs are **shared between neighbouring samples**. Deduplicate them before
  fetching, or it will download the same chunk many times.
- `build_ifs_ensemble_store.py` currently hard-codes "the exact 18 weekly
  JJAS-2020 timestamps". It must read the init times from a given baseline
  store instead.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Measure first; don't assume.
   - Time one real chunk fetch for each of: GraphCast derived rainfall,
     HRES, IFS-ENS (full 50 members), and one lagged-member (init, lead)
     pair.
   - Extrapolate the full daily cost for each store: chunk count x time
     per chunk / workers.
   - Write the numbers into the PR description before launching anything
     long.
   - If the IFS-ENS projection exceeds 12 h even with parallel workers,
     route the tradeoff via AskUserQuestion: full 50 members / a fixed
     member subset / 2-day cadence for IFS-ENS only. Include the measured
     numbers.

2. scripts/build_baseline_store.py
   - Build data/baseline_2020_jjas_daily.zarr with --init-cadence-days 1
     and a new --out.
   - The weekly store stays untouched.
   - Confirm 00 UTC-only inits (sample the 00 UTC time, not 12 UTC), and
     document which cycle is used.
   - Confirm the manifest records the cadence.

3. scripts/build_lagged_ensemble_store.py
   - Add --init-cadence-days (or a --nominal-times-from STORE option).
   - Compute the UNIQUE set of (init_time, lead) pairs needed across all
     nominal inits, fetch each once, then assemble members per nominal
     forecast. Keep weavr.ensemble.build_lagged_ensemble as the pure
     assembly step.
   - Test the deduplication logic on synthetic times: overlapping lags
     between consecutive days must produce one fetch each.
   - Output: data/lagged_ensemble_inputs_2020_jjas_daily.zarr.

4. scripts/build_ifs_ensemble_store.py
   - Replace the hard-coded 18 timestamps with --init-times-from STORE
     (read the baseline store's time coordinate). Keep the old behaviour as
     the default, so the weekly build stays reproducible.
   - Add --workers N: a thread pool over chunk fetches, keeping the
     existing staging/resume guarantees (each chunk written atomically
     under the .staging dir, and skipped on re-run).
   - Output: data/ifs_ens_2020_jjas_daily.zarr.
   - Wrap each fetch with a timeout and a bounded retry, so one stalled GCS
     request can't hang the run.

5. Run the builds in the background: baseline first (fast), then lagged,
   then IFS-ENS. Monitor them without polling tightly. While they run, do
   the code/test work above.

6. Validate each finished store:
   - init count (expect about 122)
   - no all-NaN samples
   - leads present
   - lagged member counts per lead (compare with the weekly store's
     documented 6/8/9/9/9 for temperature and its stricter rainfall limits)
   - IFS-ENS member count
   Record everything in the manifest and in a new "Daily cadence addition
   (2020)" section of docs/baseline-store.md, with the real measured costs.

7. Do NOT re-run the tier scripts here. Step 07 does that, once 2018
   exists too. Open one PR for the code, tests and docs (the data is
   gitignored).
```

## Done when

- `data/baseline_2020_jjas_daily.zarr`,
  `data/lagged_ensemble_inputs_2020_jjas_daily.zarr` and
  `data/ifs_ens_2020_jjas_daily.zarr` exist and are validated, with about
  122 daily inits each (or the user-approved IFS-ENS fallback).
- The builders are parametrised, deduplicated, parallel and tested.
- `docs/baseline-store.md` records the real measured costs.
