# WEAVR improvement prompts

These prompts execute
[`../WEAVR-SIH-improvement-plan.md`](../WEAVR-SIH-improvement-plan.md), one
step at a time, in one flat folder.

- **Done:** steps 01–05 — see [`completed/DONE.md`](completed/DONE.md) for
  what was delivered, the measured findings, the state of the data stores
  and the caveats that carry forward. Their prompts are in
  [`completed/`](completed/).
- **Remaining:** the 21 prompts below. Order and gates:
  [`../EXECUTION-PLAN.md`](../EXECUTION-PLAN.md).

## Remaining prompts

| # | Prompt | Plan | Priority | Needs |
|---|---|---|---|---|
| 06 | [add-2018-season-stores](06-add-2018-season-stores.md) | A2 | P0 | 05 ✅ |
| 07 | [regenerate-results-v2-with-loyo](07-regenerate-results-v2-with-loyo.md) | A1, A2, A4 | P0 | 02 ✅, 04 ✅, 05 ✅, 06 — **the gate** |
| 08 | [integrate-neps-g](08-integrate-neps-g.md) | A3 (2) | P0 | 03 ✅, 06, 07 |
| 09 | [independence-weighting-and-contributions](09-independence-weighting-and-contributions.md) | B1 | P0 | 07 (08 recommended) |
| 10 | [live-daily-pipeline-schedule](10-live-daily-pipeline-schedule.md) | D2 (1) | P1 | 07 |
| 11 | [tail-repair-quantile-mapping](11-tail-repair-quantile-mapping.md) | B3 (1) | P0 | 04 ✅, 07 |
| 12 | [tail-repair-extreme-value-tail](12-tail-repair-extreme-value-tail.md) | B3 (2) | P0 | 11 |
| 13 | [combine-the-combiners](13-combine-the-combiners.md) | B5 | P1 | 07 |
| 14 | [district-warnings-science](14-district-warnings-science.md) | | P0 | 04 ✅, 12 |
| 15 | [trust-scale-and-economic-value](15-trust-scale-and-economic-value.md) | | P1 | 07, 12 |
| 16 | [dashboard-evidence-views](16-dashboard-evidence-views.md) | | P1 | 04 ✅, 09, 15 |
| 17 | [dashboard-district-warnings-and-explainer](17-dashboard-district-warnings-and-explainer.md) | | P0 | 09, 14 |
| 18 | [cap-alert-export](18-cap-alert-export.md) | | P1 | 14, 17 |
| 19 | [event-replay-kerala-2018](19-event-replay-kerala-2018.md) | | P1 | 07, 08, 14, 17 |
| 20 | [static-deploy-reproducibility-model-card](20-static-deploy-reproducibility-model-card.md) | | P1 | 16–19 |
| 21 | [self-healing-online-weights](21-self-healing-online-weights.md) | D2 (2) | P1 | 10, 07 |
| 22 | [optional-gencast-and-fuxi](22-optional-gencast-and-fuxi.md) | B2 | optional | 05 ✅, 07 |
| 23 | [optional-tier4-distributional-network](23-optional-tier4-distributional-network.md) | B4 | optional | 04 ✅, 07 |
| 24 | [optional-temperature-and-heatwave](24-optional-temperature-and-heatwave.md) | C6 | optional | 05 ✅, 06 |
| 25 | [optional-northeast-monsoon-extension](25-optional-northeast-monsoon-extension.md) | C7 | P2 optional | 06, 07 |
| 26 | [pitch-rebuild-and-finale-prep](26-pitch-rebuild-and-finale-prep.md) | §8–10 | P0 | every step chosen |

Dependencies come from each prompt's own header, which is authoritative.

## Local companion folders (not in this repo)

Some prompts refer to three folders that sit **next to** the repository
checkout, not inside it:

| Folder | Holds | Why it is not committed |
|---|---|---|
| `../ppt/` | pitch deck sources (`content/slide*.md`), QR codes | presentation material, not part of the package |
| `../research/` | reference papers | third-party publications |
| `../HEPPI/` | HEPPI reference dataset and MATLAB/R code | licence unconfirmed — never commit it or grids derived from it |

A prompt that says "outside the repo" means one of these. If you don't have
them locally, skip the parts that need them and say so in the PR.

## Carry-forward from 01–05

Step 07 must state the **mixed cadence** wherever it reports a Tier 2
number: Tier 0, Tier 1 and single-source get 122 days, while Tier 2's
EMOS-on-IFS-ENS and BMA stay on the weekly 18-init IFS-ENS store (the daily
IFS-ENS pull was skipped). Details and the other measured findings are in
[`completed/DONE.md`](completed/DONE.md).

## Carry-forward from the OpenStreetMap basemap

The blended-map and extreme-probability views now draw over an offline
OpenStreetMap basemap (details: `docs/basemap-scope.md`). Later steps build
on it:

- **Map views reuse `BasemapMap`** (`dashboard-web/js/charts/basemapMap.js`:
  `create`, `shared`, `setCells`, boundary layer, fallback). Don't add a
  second projection or a second map library. Affects steps 16, 17, 19.
- **The basemap draws no national or disputed boundaries.** The national
  outline is a user-supplied official-depiction file at
  `data/basemap/india-boundary.geojson` (gitignored, may not exist). Step
  14's district polygons must sit under it and agree with it.
- **Never hot-link a public tile server.** Tiles come from the local
  `data/basemap/india.pmtiles`, served with Range support. Every map view
  needs a working plain-grid or list fallback.
- **Step 20 must handle the tile file** (148 MB, over git's 100 MB limit,
  and it still holds an undrawn `boundaries` layer) and the OSM/Protomaps/
  Natural Earth attributions. Step 26 packs it into the offline bundle.
- **Front-end PRs run no CI** (the workflow ignores Markdown and
  `dashboard-web/`). Steps 16, 17 and 19 must run their own browser checks
  and the basemap tests (`tests/test_dashboard_basemap.py`,
  `tests/test_check_basemap_boundaries.py`) locally.

## Conventions

Apply to every step.

1. **Fresh branch.** Start from an up-to-date `main`
   (`git checkout main && git pull`) and create a new branch named
   `improve-NN-<slug>`. Open one PR per step, then wait for the user's
   explicit merge instruction. Rebase on `origin/main` immediately before opening.
2. **Measure before claiming.** Some prompts quote numbers from the plan's
   §2 as *expected* values. Reproduce them. If a number doesn't reproduce,
   report the real one and why. Never copy an expected number into a doc.
3. **Pre-registered claims only.** Once step 02 lands,
   `docs/preregistration.md` is the only source of headline claims.
   Anything not pre-registered is labelled "exploratory". A failed criterion
   is reported plainly, the way Tier 3's no-go was.
4. **Route genuine tradeoffs through the user,** with the tradeoff stated.
   Don't ask about choices that have a clear default; decide them, and write
   down the reasoning.
5. **Tests and checks.**
   - Every new pure function gets tests. CI has no data stores, so use
     synthetic fixtures and the existing skip patterns
     (`pytest.importorskip`, or skip when a store is missing).
   - Run all of these, and keep them clean: `ruff check .`, `mypy src`,
     `mypy` on any touched `dashboard/` or `scripts/` files, `pytest`.
6. **UI changes need a real browser.**
   - Run `uvicorn dashboard.api:app` on a **fresh, never-used port** (the
     browser pane caches by origin).
   - Exercise every lead and option.
   - Check for errors with direct JS checks, not just the console log.
   - Never report "works" without driving it.
   - Map views build on `BasemapMap`; see "Carry-forward from the
     OpenStreetMap basemap".
   - New chart modules use module-prefixed globals. This avoids the
     `el`/`SVG_NS` collision from frontend step 4.
7. **Data hygiene.**
   - Large data is never committed (`data/` is gitignored).
   - **HEPPI raw data, and grids derived from HEPPI, are never committed or
     published** unless the user confirms the licence.
   - Derived scores and metadata are fine, with a citation.
8. **Long fetches.** Measure one chunk first, extrapolate, and write the
   estimate down. Run the fetch resumable in the background, and never block
   on an unmeasured multi-hour job.
9. **Docs every step.** Each step adds or updates a `docs/` results
   doc and a README paragraph, in the existing style.
10. **Outside-repo edits** (`ppt/content/`, `Improvements/`) are not part of
    any PR. List them in the PR description.
11. **Outward-facing actions need explicit confirmation in chat.** Never
    send an email or message, publish a site, send an alert, or enable a
    scheduled job without it.

## A note on what step 01 already established

Anything built on top of these has to account for them — they are measured,
committed, and reproducible from `docs/single-source-and-independence-results.md`:

- **No blend currently beats raw GraphCast on RMSE at 24–96 h.** The only
  things that do are Tier 2's EMOS runs, which are single-source
  calibrations. Blending is not yet earning its place; steps 07–09 and 11–12
  are the attempts to make it.
- **The three sources are worth ~1.1 independent models** (error
  correlations 0.62–0.90). This is why averaging fails, and why step 08's
  NEPS-G and step 22's GenCast matter more than any re-weighting scheme.
- **GraphCast detects no 115.6 mm event at any lead; no source detects any
  204.5 mm event.** RMSE and warning skill point in opposite directions, so
  nobody should optimise RMSE alone.
- **The train-chosen and test-chosen best source disagree at 2 of 5 leads.**
  On 3–4 test days, nothing here is significant yet. That is what step 04's
  confidence intervals and step 07's extra season exist to fix.
