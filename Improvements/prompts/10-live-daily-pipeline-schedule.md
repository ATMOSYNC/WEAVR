# Step 10: Make the daily pipeline actually run every day

**Type**: operations prompt. Outward-facing: it enables a scheduled job, so
it needs explicit user confirmation.
**Plan items**: D2 part 1.
**Depends on**: step 07 (the v2 operational Tier 1 weights).

## Goal

Get `scripts/run_daily_pipeline.py` running **every day** on a schedule,
so that:

- a real "today" forecast exists for the demo
- the forecast history accumulates, so drift detection can actually fire
  (Phase 6's README says it cannot with today's history)
- step 21's self-healing weights have real days to learn from

Also resolve the two operational gaps Phase 6 left open: a possible
multi-hour hang in the ECMWF client, and "no live IMD observations".

## What is already true, checked in the plan

- `.github/workflows/` contains **only `ci.yml`**. There is no daily
  workflow. The README's "one-line `schedule:` addition" is really a new
  workflow file.
- `run_daily_pipeline.py`'s own docstring flags that ECMWF's client can
  retry a rate-limited request up to 500 times, which can hang a run.
- Phase 6 found `imdlib` fails on a current-year request **through the
  yearly-archive call it used**. `imdlib` also has a separate real-time API
  (`get_real_data` / `open_real_data` in its docs). Check whether that
  works before accepting "no live obs".

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Live observations.
   - Check the installed imdlib for a real-time API (inspect the package:
     get_real_data / open_real_data or equivalents; read their
     docstrings).
   - Live-test fetching IMD rainfall for the last ~7 days.
   - If it works: extend weavr.data.imd_gridded with fetch_recent(days)
     (cached, returning the same DataArray shape as fetch_year), wire
     run_daily_pipeline.fetch_today_obs to it (keep None as the expected
     fallback when a day isn't published yet), and add a skip-on-network-
     failure smoke test like the existing tests/data/ ones.
   - If it doesn't work: document exactly what was tried and the error,
     and keep the retrospective design.

2. Hang protection.
   - Check whether the installed ecmwf-opendata client exposes a
     retry/timeout setting. Prefer configuring it over wrapping it.
   - Otherwise run each fetch under a hard timeout (e.g. a worker process
     with a deadline), so one source's hang becomes a "missing source" for
     weavr.renormalize, which is exactly its designed path.
   - Add a unit test with a fake fetcher that hangs.

3. Point the pipeline at step 07's v2 operational weights file (both
   seasons), and check that it loads.

4. Draft .github/workflows/daily-pipeline.yml:
   - triggers: schedule (cron chosen from ECMWF's published open-data
     release times for the 00 UTC cycle; look them up and cite them) and
     workflow_dispatch
   - install only what the pipeline needs
   - run the pipeline with a job-level timeout-minutes
   - persist outputs
   Route two decisions via AskUserQuestion before enabling anything:
   a. Where the daily outputs go:
      - commit to main via the Actions bot (simple, but noisy history)
      - commit to a dedicated `pipeline-data` branch (clean main; the
        dashboard must read from there)
      - upload as workflow artifacts / release assets (no repo writes;
        they expire or need an extra fetch step)
      State the tradeoffs, including the `permissions: contents: write`
      each option needs.
   b. Whether to enable the schedule now, or merge with only
      workflow_dispatch and enable the cron later. It fetches real data
      daily and uses Actions minutes.

5. After merge, and only with the user's go-ahead in chat, trigger one
   workflow_dispatch run. Confirm it produced the expected outputs, and
   report its runtime and which sources arrived. A partial run (e.g. AIFS
   missing) is a valid, reported outcome.

6. Docs: in docs/phase6-operational-scope.md, add a "Live operation"
   section covering the workflow, the output location, the live-obs
   finding, the hang protection, and what the first real run produced.
   Correct the README's "one-line addition" wording. README paragraph.
```

## Done when

- A reviewed `daily-pipeline.yml` is merged, with the schedule
  enabled or dispatch-only as the user decided.
- ECMWF hangs are bounded and tested.
- The live-IMD question is answered by a real test.
- At least one real run has completed, with its outputs where the user
  chose.
