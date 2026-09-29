# Step 07: Regenerate every result on the v2 evidence base (daily, two seasons, LOYO)

**Type**: evaluation prompt. This is the step where the new numbers arrive.
**Plan items**: A1 + A2 (evaluation half) + A4 (the first real verdicts).
**Depends on**: step 02 (pre-registration merged *before* this runs),
step 04 (significance machinery), and steps 05–06 (the daily 2018 + 2020
stores).

## Goal

Re-run every existing tier on about 240 daily samples across two seasons,
with leave-one-year-out (LOYO) cross-validation over **both** folds. Then:

- compute the scorecard and the pre-registered verdicts with real CIs
- re-measure which rain bins are fittable
- update every doc, dashboard caption and example grid that quotes a v1
  number

Report the outcome exactly as it lands, including any claim that gets
*weaker*.

## Why this step needs care

- `run_tier1/2/3_*.py` call `seasonal_block_split` directly, so they
  never switch to LOYO.
- `run_tier0_baseline.py` does try LOYO, but it scores only the **first**
  fold (`next(iter(leave_one_year_out(...)))`). With two years, both folds
  must be scored and pooled.
- Several shipped texts quote v1 facts as if permanent, for example:
  - the skill-trends caption in `dashboard-web/js/charts/skillTrends.js`
    ("tier1 beats tier0 domain-wide at every lead … tier3 … 0 of 5")
  - `dashboard/data_loading.py`'s docstring
  - README.md
  - each `docs/*-results.md`
  - `ppt/content/`
  Each must be updated to v2 facts, or explicitly marked as a v1 historical
  record.
- BMA's Monte Carlo CRPS and EMOS fitting will take about 13× longer (7×
  more days, 2 seasons). Measure the runtime before assuming it is fine.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Make every scoring script multi-season and LOYO-aware:
   - Load stores via weavr.stores.open_multi_season (step 06) from new
     --baseline-stores / --lagged-stores / --ifs-stores arguments. Default
     to the 2018 + 2020 daily stores.
   - Replace direct seasonal_block_split calls (tier1/2/3, single-source,
     independence) with one shared helper, e.g.
     weavr.splits.iter_evaluation_folds(times): all
     leave_one_year_out folds when there are 2+ years, else
     seasonal_block_split, returning a split label.
   - Fix tier0 and phase2 to iterate ALL folds, not next(iter(...)).
   - Every aggregate CSV gains `fold` rows plus `pooled` rows (scores
     recomputed from the pooled per-day outputs, not averages of fold
     averages). Per-day outputs (step 04) carry the fold label.
   - Tests: the helper yields 2 folds for 2 synthetic years and 1 labelled
     block split for a single year.

2. Archive v1. Move today's results/*.csv (the weekly, single-season
   results) into results/archive_v1_weekly_2020/ with a README saying what
   they are and which commit produced them. Point dashboard/data_loading.py
   defaults at the new v2 files, and make sure the API tests still pass
   (update the tests' expected leads/shapes only if v2 genuinely changes
   them).

3. Measure runtime on one lead first, extrapolate, and write it down. If
   BMA's Monte Carlo sampling makes the full run impractical (> ~6 h),
   route it via AskUserQuestion: fewer MC samples (with the CRPS error that
   causes quantified on a subset) / subsample cells / run overnight
   as-is.

4. Re-run everything, in dependency order:
   - run_single_source_baseline
   - run_independence_diagnostic
   - run_tier0
   - run_phase2
   - run_tier1 (weights refit per fold; ALSO fit on both years for the
     operational weights file)
   - run_tier2
   - run_tier3 (re-run Phase 5's own go/no-go criterion unchanged; report
     whether more data changes the verdict)
   - run_daily_verification (the drift demo now has more than 14 samples,
     so check whether drift can fire across the 2018 -> 2020 GraphCast
     checkpoint change, and report what actually happens)
   - export_dashboard_example_grids (from which season? use 2020, the
     season the operational weights target, and state it)

5. Re-measure the per-bin fittability table from
   docs/phase4-data-and-combiner-scope.md on v2 train folds. Report which
   bins become fittable under MIN_TRAIN_DAYS_PER_BIN = 5 at which leads.
   Do not change that constant here.

6. Significance:
   - Compute lag-1 autocorrelation of daily domain-wide MSE per lead
     (weavr.significance.lag1_autocorrelation) and choose the block length
     per docs/preregistration.md's rule. If the rule requires changing it,
     commit that doc change FIRST, as its own commit, then compute.
   - Run run_scorecard.py and produce results/preregistration_verdicts.csv.
     Claims whose inputs don't exist yet (H4-H11) are NOT_YET_TESTED.

7. Write docs/v2-evidence-base-results.md:
   - a v1 vs v2 table per tier and lead
   - fold-level and pooled scores
   - the new bin fittability
   - the H1-H3 verdicts with CIs, reported whatever they are
   - what changed and why
   If a v1 claim no longer holds (e.g. Tier 1 beating Tier 0 at every
   lead), say so in the first paragraph, not buried.

8. Update every text that quotes v1 facts:
   - README.md: add a v2 section; keep v1 text as clearly-labelled history
   - each docs/*-results.md: a dated "v2 update" note pointing to the new
     doc
   - dashboard-web/js/charts/*.js captions and dashboard/data_loading.py
     docstrings, to v2 facts
   Then browser-verify all 4 views on a fresh port, across every lead and
   both metrics.
   List any ppt/content/ lines now contradicted by v2 in the PR
   description. Step 26 rewrites the slides; don't edit them here.

9. One PR (code + v2 results CSVs + docs). The data stores stay gitignored.
```

## Done when

- Every scoring script runs LOYO over both folds.
- v1 results are archived and v2 results committed.
- Per-bin fittability is re-measured.
- The scorecard and H1–H3 verdicts exist with real CIs.
- `docs/v2-evidence-base-results.md` reports the outcome plainly.
- No shipped text quotes a v1 number as current.
