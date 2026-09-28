# Phase 6 operational scope: run-ourselves-vs-published, storage & runner

Issue #8's two non-code checkboxes, decided here before any renormalization
or drift-detection code gets written (steps 2-4 of
[`solving issues/07-phase-6-robustness-and-operations/`](../../solving%20issues/07-phase-6-robustness-and-operations/)).

## Decision 1: run-ourselves vs. consume-published-forecasts

This decision is not uniform across sources — checking it against what this
project can actually reach today splits it in two, differently for the NWP
side and the AI-model side.

### NWP (IFS/HRES): settled, published already works

`src/weavr/data/ecmwf_open_data.py` already fetches IFS from ECMWF Open
Data's `source="aws"` mirror successfully, with no rate-limiting
(`docs/data-sources.md`'s own confirmed test). Running an operational NWP
model like IFS ourselves would mean reproducing ECMWF's own forecasting
system — out of scope for this project by any measure. **Decision: use
published IFS/HRES via the existing client. No further tradeoff to track
here.**

### AI models: no published live route for GraphCast/Pangu — checked, not assumed

Phase 2 named GraphCast and Pangu as this project's "AI models." Checking
whether either has a published, near-real-time forecast this project could
consume daily (rather than assuming WeatherBench 2 could serve that role):

- `src/weavr/data/weatherbench2.py`'s own `GRAPHCAST_2020` path is a fixed
  historical eval window, `2019-11-16..2021-02-01` — a research archive, not
  a continuously-updated feed. The module's own comment says "Known dataset
  paths as of the 2020 GraphCast eval period." Pangu's WeatherBench 2 store
  is the same shape. **Neither is a live source.**
- No other published, near-real-time GraphCast or Pangu forecast feed is
  reachable from this project's existing code or dependencies.
- **ECMWF's own AI model, AIFS, is a real, checkable exception** — it is
  served through the exact same `ecmwf_open_data.py` client already used for
  IFS, since that client's own docstring already scopes it as "IFS HRES +
  AIFS." Live-tested directly (not assumed from the docstring alone): the
  bare model string `"aifs"` fails immediately with `ValueError: Cannot
  establish latest date for ...` — a real, checked gotcha, not a network
  issue. Reading `ecmwf.opendata.client`'s own source
  (`.venv/lib/python3.14/site-packages/ecmwf/opendata/client.py`) found the
  correct model string is `"aifs-single"` (deterministic) or `"aifs-ens"`
  (ensemble). Retrying with `model="aifs-single"` got past that error and
  reached ECMWF's real AWS-mirrored endpoint (confirmed: a real
  `HTTP 503 Slow Down` rate-limit response came back mid-transfer, which
  only happens once a request has actually reached the live service — not a
  config or auth failure). This confirms AIFS is a genuine, reachable,
  published near-real-time AI-model forecast; the fetch wasn't run to full
  completion (not needed to establish reachability), following this
  project's own established practice of only running a fetch to the depth
  needed to answer the question at hand.

**Decision (user-confirmed via `AskUserQuestion`): substitute AIFS for
GraphCast/Pangu as the daily pipeline's AI-model input.** No new compute or
model-hosting cost — reuses the existing, already-working
`ecmwf_open_data.py` client with `model="aifs-single"`. Trade-off, stated
plainly: AIFS is a different model from the GraphCast/Pangu this project's
combiners (Phases 2-5) were actually fit and evaluated against, so the daily
pipeline's real skill may not track those phases' historical results docs
exactly — this is a genuine, accepted divergence between "the models this
project was built and evaluated against" and "the models the daily pipeline
can actually reach in real time," not an oversight.

**Follow-on for step 4**: `scripts/run_daily_pipeline.py` fetches AIFS
(`model="aifs-single"`) wherever the pipeline previously would have fetched
GraphCast/Pangu, using the corrected model string documented above.

## Decision 2: storage & pipeline-runner shape

Checked against what this repo already has, rather than assuming a new
stack is needed:

- `.github/workflows/ci.yml` already exists and runs lint/type-check/tests
  on `push`, `pull_request`, and `workflow_dispatch` — **no `schedule:`
  cron trigger exists yet.**
- Every existing store-building script (`build_baseline_store.py`,
  `build_lagged_ensemble_store.py`, `build_ifs_ensemble_store.py`) writes to
  a local, gitignored Zarr store on whatever machine runs it, as a one-shot,
  full-season build — not an incremental daily append. `build_ifs_ensemble_store.py`'s
  own docs (`docs/baseline-store.md`) record a real operational gotcha
  running one of these on a laptop: idle-sleep silently killed live GCS
  connections three times mid-run, recovered only because that script's own
  resumable-fetch/staging pattern absorbed it.
- This project has never had, and does not plan to stand up, an always-on
  machine — every real data pull so far has been a one-off, manually-invoked
  local run.

**Decision (user-confirmed via `AskUserQuestion`): run the daily job on a
GitHub Actions `schedule:` trigger**, reusing the CI infra this repo already
has rather than requiring a new always-on machine. Since GitHub Actions
runners are ephemeral (no persistent disk across runs), the daily job's
output is a small, committed results artifact — mirroring this project's own
established convention of committing `results/*.csv` files alongside every
`run_tierN_*.py` script — rather than a growing local Zarr store. **This
means today's one-shot, full-season store-building scripts do not need to
become incremental appenders for Phase 6**: the daily job pulls just that
day's AIFS/IFS/HRES forecast and IMD observation (once available), scores
it, and commits/publishes the result — it does not need to touch or extend
`data/baseline_2020_jjas.zarr` or the other existing stores, which remain
the fixed 2020 JJAS research stores every prior phase's results were
computed against.

**Follow-on for step 3/4**: `scripts/run_daily_verification.py`'s trailing
window and `scripts/run_daily_pipeline.py`'s daily run both write small,
committed result files under `results/`, and step 4 adds the
`schedule:`-triggered workflow entry to `.github/workflows/` (a new
workflow file, or a new job in `ci.yml` — step 4 decides and states which
once it sees the real job shape).

## Summary

| Decision | Outcome |
|---|---|
| NWP (IFS/HRES) run-ourselves vs. published | Published (ECMWF Open Data) — settled, no tradeoff |
| AI-model run-ourselves vs. published | Published — **AIFS substituted for GraphCast/Pangu**, since neither has a live published route |
| Storage & pipeline-runner | GitHub Actions `schedule:` trigger, committed small results (not an incremental Zarr store) |
