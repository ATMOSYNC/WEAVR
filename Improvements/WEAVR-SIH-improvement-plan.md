# WEAVR: improvement plan to win SIH (PS 26081)

*Written 29 Sep 2026, against the repo at `b4b9257` (after PR #51).*

Everything in **§2 "What was verified today"** was measured against the real
repo, the local data stores and the live WeatherBench 2 bucket before it was
written down. The snippets to re-run each check are in Appendix A.
Everything else is a plan: each item says how to check it before anyone
claims it on a slide.

---

## 0. TL;DR

The engineering is strong and unusually honest. The things stopping WEAVR
from winning are not missing features. They are four weaknesses a sharp jury
(MoES / IMD / NCMRWF scientists) will find within five minutes of Q&A:

1. **The evidence is thin.** Every result comes from 18 *weekly* samples of
   a single season (JJAS 2020). Test sets are **3–4 days**, and no result
   has a confidence interval.
2. **The blend does not yet beat the best single model.** On the same test
   days, raw GraphCast has lower RMSE than both Tier 0 (equal mean) and
   Tier 1 (regional blend) at 24–96 h, and ties at 120 h. Slide 1 says WEAVR
   is "provably better than any single source".
3. **India's own model is missing.** Slides 1–2 say NCUM/NEPS-G are
   ingested, but no code touches them.
4. **The pitch is about extremes, but the system can't detect them.** POD at
   204.5 mm is 0 at every lead. POD at 115.6 mm is 0 beyond 24 h at most
   leads. The very-heavy and extremely-heavy bins have 0 test cells. The
   lagged "AI ensemble" is 4–7× under-dispersive.

**The good news, verified today: the data to fix all four is already within reach.**

- **Daily sampling instead of weekly.** The `--init-cadence-days` flag
  already exists. It gives about 122 samples per season instead of 18 (≈7×),
  and costs about 610 chunk fetches per source, not the 4,880 the docstring
  feared.
- **A second season is available.** GraphCast's 2018 WeatherBench 2 archive
  does contain 24 h rainfall for all of JJAS 2018. `docs/baseline-store.md`
  says it doesn't. HRES and IFS-ENS cover 2016–2022+. Two seasons unlock
  leave-one-year-out cross-validation, the phase plan's original design.
- **HEPPI's missing calendar is recoverable.** For 241 of 242 monsoon days,
  the recovered date is the single best match among 1,830 candidate IMD
  days. That makes HEPPI **NCMRWF's NEPS-G 23-member ensemble for JJAS 2018
  + JJAS 2019**, already on WEAVR's exact 129×135 grid.
- **Two more AI models are available for 2020.** GenCast (a real 56-member
  AI ensemble) and FuXi are in WeatherBench 2 at 0.25° with 24 h rainfall.

**The five moves that change the outcome most, in order:**

| # | Move | Items |
|---|---|---|
| 1 | Evidence base: daily samples, a 2018 season, leave-one-year-out CV, confidence intervals, and an honest best-single-model baseline | A1, A2, A4, A5 |
| 2 | India's own model in the blend (NEPS-G via HEPPI date recovery), compared head-to-head against the *published* HEPPI EMOS post-processing | A3 |
| 3 | The "3 models ≈ 1.1 independent opinions" finding, plus independence-aware weights and per-model contribution maps (unique, and a gap your own research brief flagged) | B1 |
| 4 | Decision products: district-level IMD colour warnings with a *verified* probability→colour rule, CAP alert export, and economic-value curves | C1, C3, C4 |
| 5 | The **Kerala August 2018 replay**, trained only on 2020, so it is truly out-of-sample | D1 |

**Suggested one-line positioning:**

> *"Blends assume they have N models. WEAVR measured that GraphCast, HRES and
> IFS share so much error they're worth about 1.1 independent forecasts. So it
> adds India's own NEPS-G, weights by independence and not just skill, repairs
> the heavy-rain tail AI models smooth away, and delivers district-level IMD
> colour warnings a CAP alert system can ingest. Every number is scored
> against IMD's own gauges, with confidence intervals."*

---

## 1. How the jury will read WEAVR

SIH juries typically mix ministry / nodal-agency scientists (here likely
MoES, IMD and NCMRWF), academic domain experts and industry evaluators.
Check the official SIH 2026 rubric. Juries generally reward:

- completeness against the problem statement
- technical depth and correctness
- novelty
- adoptability by the ministry
- a working demo
- a clear pitch

WEAVR's rigor is a real asset. Right now it is presented as *process*
("we checked everything"). It needs to become *results* with error bars, and
*products* a forecaster or disaster manager would actually use.

### 1.1 Problem-statement coverage today

This checklist was reconstructed from the research brief's framing prompt
and `docs/phase-plan.md`. **Re-check it against the official PS 26081 text**
and add any line that is missing.

| The PS asks for | WEAVR today | Gap |
|---|---|---|
| Blend physics NWP, ensemble systems and AI models | GraphCast, HRES and IFS-ENS for rain. Pangu for temperature only. AIFS only in the live runner | NCUM/NEPS-G absent. FuXi and GenCast unused |
| Rainfall, temperature and wind | Rainfall only | Temperature and wind not blended (C6) |
| Adaptive weights by lead, region, season and regime | Lead × 6 zones. Regime tested honestly (no-go) | One season only, so no "season" dimension |
| Skill on extremes, at IMD thresholds | Bins exist, but extreme bins are never fittable. POD is 0 at 204.5 mm | **The headline gap** (A1, A2, B3) |
| Blended map, weight map, skill trends, extreme-probability map in IMD colours | All 4 built; the two grid maps now draw over an offline OpenStreetMap basemap (`docs/basemap-scope.md`) | These are research views, not decision products (C1–C5) |
| Operational robustness | Renormalization, drift detection and a daily runner exist | Not scheduled, AIFS mismatch, no live obs (D2) |

---

## 2. What was verified today

### F1. The evidence base is 18 weekly samples, with 3–4 test days

- Every store (`graphcast`, `hres`, `ifs_ens_mean`, `pangu`) has exactly 18
  init times (2020-06-01 → 2020-09-28, weekly), because of
  `DEFAULT_INIT_CADENCE_DAYS = 7` in `scripts/build_baseline_store.py`.
- Every tier's CSV reports `n_train=14` and `n_test=4` (24–72 h) or
  `n_test=3` (96–120 h).
- The weekly choice was justified in that script's docstring as
  "every 12-hourly init × every 6-hourly lead ≈ 4,880 chunk fetches". But
  the project only uses **5 leads**. Daily 00 UTC over JJAS is
  122 inits × 5 leads = **610 chunk fetches per source**. At the measured
  0.75–2 s per chunk, that is roughly 8–20 minutes.

### F2. The blend loses to raw GraphCast on RMSE

All numbers are domain-wide RMSE in mm, on the **same**
`seasonal_block_split` test days, scored with the project's own
`_align_to_imd_day` and `weavr.verify.rmse`:

| Lead | GraphCast alone | IFS-ENS mean alone | HRES alone | Tier 0 (equal mean) | Tier 1 (regional) | Tier 2 EMOS on GraphCast | Test days |
|---|---|---|---|---|---|---|---|
| 24 h | **11.49** | 11.76 | 13.50 | 11.66 | 11.55 | 10.86 | 4 |
| 48 h | **12.45** | 13.78 | 16.42 | 13.54 | 12.62 | 12.08 | 4 |
| 72 h | **11.80** | 13.85 | 17.97 | 13.38 | 12.96 | 11.27 | 4 |
| 96 h | **12.71** | 14.48 | 19.29 | 14.27 | 13.65 | 12.95 | 3 |
| 120 h | 14.83 | 15.38 | 17.65 | 15.30 | **14.82** | 14.88 | 3 |

The single-source columns were recomputed today. Tier 0 and Tier 1 come from
their own results CSVs, on the same split. The Tier 2 column comes from
`tier2_hierarchical_baseline.csv`, which scores a slightly different cell
mask: its own Tier 0 at 24 h is 11.55, not 11.66. So compare Tier 2 to the
other columns loosely.

What this means:

- Tier 1 beats Tier 0 at every lead, which is true and already reported. But
  **neither blend beats the best single source**, except for a 0.01 mm tie
  at 120 h.
- The best RMSE comes from *post-processing GraphCast alone* (Tier 2
  EMOS-GraphCast).
- Tier 2's multi-source BMA beats raw GraphCast at 24–48 h, ties at 72 h and
  loses at 96–120 h.
- Mitigating context: RMSE rewards smooth, MSE-trained AI output (research
  brief §1.3). Blends should also win on CRPS, heavy-rain POD/ETS and
  reliability, so they must be judged there too. **The slide claim still has
  to change now** (A5), and the result has to be re-earned (A1, A2, B1).
- With 3–4 test days, none of these differences is statistically meaningful
  either way (A4).

### F3. The three rainfall sources are ≈1.1 independent opinions

Pairwise correlation of forecast errors (forecast − IMD), pooled over all
grid cells and all 18 samples:

| Lead | GraphCast ~ HRES | GraphCast ~ IFS-ENS mean | HRES ~ IFS-ENS mean | Effective number of models\* |
|---|---|---|---|---|
| 24 h | 0.84 | 0.93 | 0.92 | 1.07 |
| 48 h | 0.81 | 0.92 | 0.88 | 1.10 |
| 72 h | 0.75 | 0.90 | 0.84 | 1.13 |
| 96 h | 0.71 | 0.88 | 0.82 | 1.15 |
| 120 h | 0.76 | 0.92 | 0.83 | 1.12 |

\* `N_eff = N / (1 + (N−1)·ρ̄)`, with N = 3 and ρ̄ the mean pairwise
correlation. This is a quick diagnostic. Re-measure it per region on the
train split only (B1).

- GraphCast's errors track IFS-ENS most closely (0.88–0.93), consistent with
  GraphCast being trained on ERA5, which comes from IFS.
- The Tier 1 weights (`results/tier1_regional_weights.csv`) give **HRES a
  weight of exactly 0 in 25 of 30 region×lead cells**. That is collinearity
  with IFS-ENS (0.82–0.92), not "HRES is useless". It is also the question a
  judge will ask first when they see the weight map.

### F4. HEPPI's missing dates are recoverable

HEPPI's `IMD_observed.nc` has 334 undated samples. Each one was matched
against every day in the local IMD archive
(`data/imd_seeps_climatology_jjas.zarr`: JJAS 2006–2020, 1,830 days).

- **Mapping:**
  - HEPPI indices 0–119 → 2018-06-02 … 2018-09-30, skipping 2018-06-24.
    HEPPI's own `verif_hitmiss.m` notes a missing 24 June.
  - Indices 181–302 → 2019-06-01 … 2019-09-30.
- **Strength of the match:**
  - With the mapping as given, the hypothesised date is the single best of
    1,830 candidates for **241 of 242** days.
  - Shifted by −1 day: best match for 1 of 241 days.
  - Shifted by +1 day: best match for 0 of 240 days.
- **Indices 120–180 and 303–333 (92 samples)** fall outside JJAS. They are
  probably Oct–Nov 2018 and Oct 2019 (181 + 153 = 334 fits exactly). This is
  unverified, because the local archive is JJAS-only.
- The values are close but **not bit-identical**. IMD revises its gridded
  product over time. So use IMD from `imdlib` as the ground truth, and HEPPI
  only for the NEPS-G *forecasts*.
- **Consequence:** `docs/phase4-data-and-combiner-scope.md` rejected HEPPI as
  training data only because the calendar was missing. That reason no longer
  holds.

### F5. GraphCast has a usable 2018 monsoon

- `gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours_derived.zarr`
  has `total_precipitation_24hr`:
  - 884 init times, 2017-11-16 → 2019-01-31
  - 0.25°
  - leads 6–240 h
- It uses `lat`/`lon` dimension names, not `latitude`/`longitude`. That is
  probably why it looked unusable.
- `docs/baseline-store.md`'s line "Only the second window overlaps a full
  monsoon season" is wrong and should be corrected.
- HRES (2016-01-01 → 2023-01-10) and IFS-ENS (2016–2024) both cover 2018,
  2019 and 2020.
- `scripts/run_tier0_baseline.py` and `run_phase2_ensemble_baseline.py`
  already switch to `leave_one_year_out` automatically once a second year
  exists. **`run_tier1/2/3_*.py` call `seasonal_block_split` directly**, so
  they need that same one-block change.
- Tier 0's version also scores only the *first* fold
  (`next(iter(leave_one_year_out(...)))`). With two years, every script must
  loop over all folds.

### F6. More AI models exist for 2020 at 0.25°

| Archive | Rain variable | Coverage | Notes |
|---|---|---|---|
| `gencast/2020-1440x721.zarr` | `total_precipitation_24hr` | 2020-01-01 → 2020-12-31, leads 12–360 h | **56-member AI ensemble** (`sample` dimension) |
| `fuxi/2020-1440x721.zarr` | `total_precipitation_24hr_from_6hr` | 2020-01-01 → 2020-12-16, leads 6–360 h | Deterministic AI model |
| `neuralgcm_*` | — | 2020 | Only 0.7–1.4°: `SourceTooCoarseError`. **Skip** |
| `aurora` | — | 2022 only | No overlap. **Skip** |

### F7. The extreme-rain numbers

- **POD by threshold (Tier 1)** comes from `results/tier1_regional_baseline.csv`:
  - ≥64.5 mm: 0.16 / 0.02 / 0.09 / 0.04 / 0.04 across the 5 leads
  - ≥115.6 mm: 0.12 at 24 h, then **0** at every later lead
  - ≥204.5 mm: **0 at every lead**
  - Tier 0 is the same or similar.
- **Test cells per bin**, from `tier2_hierarchical_baseline_by_bin.csv`:
  - heavy (64.5–115.6 mm): 23–75 cells
  - very heavy: 0–1
  - extremely heavy: 0 at every lead
- **Spread-skill ratio of the lagged GraphCast ensemble** is 4.8 / 5.4 / 4.0 /
  5.5 / 6.9 across the leads, against a calibrated target of about 1.05–1.10
  (`results/phase2_ensemble_baseline.csv`). It is severely under-dispersive.

### F8. Slide claims versus reality

Fix these before anything else. Each one is a Q&A trap.

| Slide | Claim | Reality | Fix |
|---|---|---|---|
| 1, 2 | "aligns AI (GraphCast, Pangu) and NWP (NCUM/NEPS-G, IFS)" | NEPS-G/NCUM never ingested. Pangu has no rainfall | Make it true (A3), or reword |
| 1 | "provably better than any single source" | Loses to raw GraphCast on RMSE at 24–96 h (F2) | Reword now, re-earn later (A1, A2, B1, B5) |
| 2 | "using the strongest combiner per bin" | No per-bin selector exists in `src/` or `scripts/` (grepped) | Build it (B5) or reword |
| 3 | "biggest gains exactly in the heavy-rainfall bin" | No Tier 0 per-bin comparison exists. The heavy bin has 23 test cells at 24 h | Measure after A1/A2, then state it with a CI |
| 3 | "70% chance of >115.6mm in this district" | Very-heavy and extreme bins are unfittable, and there is no district product | Mark it "illustrative", or build C1 and B3 |
| 1, 3 | Chennai floods as motivation | Chennai floods are north-east-monsoon (Oct–Dec) events. WEAVR is JJAS-only | Drop it, or extend to October–December (C7) |
| 2 | "Tier 2 … beat Tier 0 at every single lead" | True, but with 3–4 test days and no CI | Keep it, and add CIs (A4) |

---

## 3. The winning narrative

Many teams will say "we blend models with ML". WEAVR can stand on three words:

- **Independent.** WEAVR measures how much *independent* information each
  model adds (N_eff, contribution maps). It adds the one source with
  genuinely different lineage: India's own NEPS-G (Unified Model–based, with
  NCMRWF's own analysis). Hypothesis to test: it lowers error correlation and
  raises N_eff.
- **Extreme-aware.** WEAVR repairs the heavy-rain tail AI models smooth away.
  It issues district-level IMD colour warnings through an explicit
  probability→colour rule, and verifies those warnings, not just pixels.
- **Accountable.** Every claim comes with a confidence interval, an
  ECMWF-style scorecard, a pre-registered go/no-go and an out-of-sample
  replay of a real disaster. Keep the honesty brand, but show it as numbers
  with error bars rather than prose.

---

## 4. The improvements

Each card follows the same shape:

- **Why:** the problem it fixes and its link to the PS
- **Unique because:** why it stands out
- **How:** concrete steps, reusing existing modules
- **Effort / risk:** in person-days (pd)
- **Done when:** a measurable exit
- **Demo moment:** how it shows up in the pitch

Priorities: **P0** must-do, **P1** strongly recommended, **P2** if capacity
allows.

### Pillar A: evidence (P0; nothing else is credible without it)

#### A1. Daily sampling: about 7× more evidence from data already reachable (P0)

- **Why:** fixes F1. Heavy-rain bins starve because the store sees one day a
  week. With 3–4 test days, no comparison means anything.
- **How:**
  1. Run `python scripts/build_baseline_store.py --init-cadence-days 1 --out data/baseline_2020_jjas_daily.zarr`
     (00 UTC inits only; the flag already exists).
  2. `build_lagged_ensemble_store.py` and `build_ifs_ensemble_store.py`
     hard-code "the exact 18 weekly timestamps". Parametrise them to read
     init times from the baseline store.
  3. IFS-ENS was measured at about 47 s per chunk. 610 chunks is about 8 h
     serially, so add a small thread pool (4–8 workers) on top of the
     existing staging/resume logic. Run it overnight.
  4. Re-run every `run_*` script against the daily stores, and regenerate
     `results/*.csv` and the dashboard example grids.
  5. Re-measure the per-bin train-day table in
     `docs/phase4-data-and-combiner-scope.md`.
  - Daily samples are autocorrelated, so pair this with block bootstrap
    (A4). Effective sample size will be well under 122, but the test window
    grows from 3–4 days to about 25.
- **Effort / risk:** 2–3 pd plus overnight fetches. Low risk, since the
  mechanisms are already built and tested.
- **Done when:** every results CSV reports `n_samples ≈ 120`, and the
  per-bin fittability table is re-measured. Projection to confirm: at 24 h,
  heavy-bin train days go from 9 to about 60, and the extremely-heavy bin
  may become fittable at short leads.
- **Demo moment:** "Every number on this slide comes from about 120 real
  forecast days, not 18."

#### A2. A second season (2018) and leave-one-year-out CV (P0)

- **Why:** a single season cannot show that weights generalise across years,
  which is the core claim of any skill-weighting scheme (Wanders & Wood 2016
  fit per year and needed several years). It also unlocks the Kerala 2018
  replay (D1).
- **How:**
  1. Extend `build_baseline_store.py` with a per-source archive map by year.
     GraphCast 2018 uses the `.../2018/..._derived.zarr` path and needs a
     `lat/lon → latitude/longitude` rename before `regrid_to_common`.
     HRES and IFS-ENS use the same paths with different dates.
  2. Pull IMD 2018 (already in the local climatology store, or fetch it via
     `imdlib`).
  3. Replace the direct `seasonal_block_split` call in `run_tier1/2/3_*.py`
     with tier0's pattern: try `leave_one_year_out`, fall back otherwise.
  4. Report both folds (train 2018 → test 2020, and train 2020 → test 2018),
     plus **weight stability across years**: the same region×lead weights
     fit on each year separately.
  - Note which GraphCast version each WeatherBench 2 window uses. A version
    change between folds is itself a real drift test for `weavr.drift`.
- **Effort / risk:** 3–4 pd plus fetches. Medium risk: the 2018 archive may
  differ in small ways beyond the coordinate names, so measure one chunk
  first, as Phase 1 did.
- **Done when:** every tier reports `split=leave_one_year_out` with 2 folds,
  and `docs/baseline-store.md`'s 2020-only claim is corrected.
- **Demo moment:** "Weights learned only on 2020, tested on 2018."

#### A3. India's own model: NEPS-G via HEPPI date recovery (P0, the most unique data move)

- **Why:** the PS is Indian, and the jury may include NCMRWF itself. A blend
  without NCUM/NEPS-G invites "why should we adopt this?" F4 turns HEPPI from
  a methodology reference into **two seasons of NEPS-G 23-member forecasts**
  on the exact grid. No other team is likely to have this.
- **Unique because:**
  1. It is an Indian operational ensemble blended with AI models.
  2. It allows a head-to-head against a **published** Indian
     post-processing benchmark: Angus et al. 2024 (QJRMS), NEPS-G + EMOS
     over India in the monsoon, which is exactly HEPPI's
     `NCMRWF_EMOS_forecast.nc`.
  3. It tests whether a different model lineage genuinely adds
     independence (B1).
- **How:**
  1. Commit a small `scripts/recover_heppi_dates.py`, the matching method in
     Appendix A, and a committed `docs/heppi-date-map.csv` (index → date).
     Commit derived metadata only; the raw data stays out of git, as now,
     because its licence is unclear.
  2. Extend `weavr.data.heppi_reference` to attach a real `time` coordinate
     from that map, and drop non-JJAS samples by default.
  3. **Confirm the lead first.** Angus et al. evaluate day-1, but check it
     by scanning the forecast–obs correlation at IMD-day offsets of 0, 1 and
     2 (the same technique as F4).
  4. Add NEPS-G as a 4th source at the 24 h lead for 2018 and 2019.
     Available sources: GraphCast (2018 only), HRES, IFS-ENS and NEPS-G.
     Refit Tier 1 and Tier 2 there.
  5. **Head-to-head on identical days:** raw NEPS-G, HEPPI-UQM, HEPPI-EMOS,
     IFS-ENS EMOS, and the WEAVR blend. Use CRPS, Brier score at IMD
     thresholds, reliability and SEEPS.
  6. Recompute N_eff with NEPS-G included.
  - Consider emailing the HEPPI authors (University of Birmingham / WCSSP
    India) to confirm the dates and the licence. It is a small effort and a
    strong credibility line.
- **Effort / risk:** 3–5 pd. Risks: the lead/init time needs confirming, and
  NEPS-G covers day 1 only.
- **Done when:** NEPS-G appears in the 24 h weight map and in the scorecard
  as "WEAVR vs published NEPS-G EMOS (Angus et al. 2024)".
- **Demo moment:** "We added India's own ensemble. Here is what it
  contributes, and how WEAVR compares to the published NCMRWF-partner
  post-processing on the same days."

#### A4. Significance, honest baselines and an ECMWF-style scorecard (P0)

- **Why:** fixes F2's "is this real?" problem. IMD/NCMRWF scientists read
  verification scorecards daily; a grid of ▲/▼ with significance is instantly
  credible to them.
- **How** (a new `weavr/significance.py` plus one dashboard view):
  1. **Block bootstrap.** Resample 7-day blocks of test days 1,000 times, and
     report 95% CIs on every score *and on every difference* (e.g. blend −
     best single model).
  2. **Diebold–Mariano test** on daily score differences for each headline
     claim.
  3. **Baselines in every table:**
     - IMD climatology, from the 15-year local archive, giving CRPSS and BSS
       against climatology
     - each raw source
     - the equal mean
     - the best single member, **chosen on train data, not test**
  4. **Missing standard diagnostics:**
     - reliability diagrams and rank/PIT histograms (HEPPI's `ver_rd.m` and
       `rank_histograms.m` show the exact plots this community expects)
     - SEDI for rare thresholds (Ferro & Stephenson 2011)
     - threshold-weighted CRPS for the tail (Gneiting & Ranjan 2011), which
       research brief §3.6 already recommends
  5. **Scorecard:** rows are metric × threshold, columns are lead, and a cell
     is coloured only when its CI excludes zero.
  6. **Pre-register every headline claim** (metric, lead, threshold,
     criterion) in a doc before re-running, extending the Tier 3 go/no-go
     habit to everything.
- **Effort / risk:** about 3 pd. Low risk.
- **Done when:** every number on the slides has a CI, and the scorecard view
  is live.
- **Demo moment:** one scorecard slide that replaces three slides of prose.

#### A5. Claims audit (P0, half a day, do it first)

- Rewrite every row of F8 so the slides only say what the repo proves today.
  Then upgrade them as A1–B5 land.
- Add F2's single-model table to the internal results docs now, before
  someone else computes it in Q&A.
- **Suggested honest wording in the meantime:** "Tier 1 beats the
  equal-weight blend at every lead. On RMSE, a post-processed GraphCast is
  currently the strongest single component; the multi-model advantage is
  being re-tested on a 2-season, daily dataset."

### Pillar B: blending science (the novelty)

#### B1. Independence-aware weighting, N_eff and contribution maps (P0, the signature idea)

- **Why:** F3 shows the three sources share 71–93% of their error. The
  research brief lists correlation-aware weighting for AI+NWP blends as an
  **open question with no study over India** (§1.5, "Where the team can
  stand out" #3). The PS's "weight map" deliverable only becomes
  *explanatory* once it shows *why* a model gets zero weight.
- **Unique because:** most teams will show weights. WEAVR would show:
  - **(a)** how many independent opinions the ensemble really contains, per
    region and lead
  - **(b)** each model's **marginal contribution** to skill (Shapley
    attribution), which answers "what does adding GraphCast / NEPS-G / GenCast
    buy India?" directly
  - **(c)** a weighting scheme that is stable with short training data
- **How:**
  1. **Diagnostics:**
     - an error-correlation matrix per region × lead, computed on train data
     - `N_eff = N / (1 + (N−1)ρ̄)`, or the eigenvalue-based participation
       ratio for unequal correlations
  2. **Shapley skill attribution:**
     - fit the blend on every subset of sources (2^N subsets: 8 for 3
       sources, 32 for 5, which is cheap)
     - score each subset on held-out data
     - use Shapley values to split the total skill gain over climatology
       among models
     - this becomes a new **contribution map**, a stronger version of the
       weight map
  3. **Shrinkage-covariance weights:**
     - use the minimum-variance combination `w ∝ Σ⁻¹1` (Bates & Granger 1969)
     - estimate the error covariance `Σ` with Ledoit–Wolf shrinkage (stable
       with few samples)
     - clip negatives and renormalise with the existing
       `weavr.weighting.clip_and_renormalize` (Wang et al. 2025 found
       correlation-weighted averaging is the BLUE, and that clipping helps)
     - compare against the current OLS (`fit_region_weights`) on
       leave-one-year-out CV
  4. **Pre-registered go/no-go:** adopt shrinkage weights only if they beat
     OLS on LOYO RMSE at ≥3 of 5 leads, with the CI excluding 0.
- **Effort / risk:** 2–3 pd. Low risk: even a "no-go" on the weights leaves
  the N_eff and Shapley diagnostics as unique deliverables.
- **Done when:** the dashboard shows an "Independence" view (correlation
  heat-map, N_eff by lead, contribution map), and README explains HRES's zero
  weights with numbers.
- **Demo moment:** "Three models, 1.1 opinions." Then, after A3/B2, "adding
  NEPS-G raises that to X", but only once measured.

#### B2. Real AI ensembles: GenCast and FuXi (P1)

- **Why:** the only "AI ensemble" today is a lagged pseudo-ensemble that is
  4–7× under-dispersive (F7). GenCast is a real 56-member probabilistic AI
  ensemble (Price et al. 2024, Nature). FuXi (Chen et al. 2023) is a third
  deterministic AI model. Together they cover the PS's AI-model list much
  better, and they test the brief's open question: does an AI ensemble beat
  IFS-ENS over India, and for extremes?
- **How:**
  1. Add both to `build_baseline_store.py`'s source list for 2020, daily
     00 UTC, leads 24–120 h, rainfall only.
  2. GenCast needs its `sample` dimension renamed to `member`, so that
     `weavr.verify.crps` and `weavr.emos` accept it as-is.
  3. Measure one chunk first. If a full 56-member pull is too slow, take a
     fixed subset of about 20 members and say so.
  4. Evaluate within 2020 only (no 2018 data), and label it as such.
  5. Re-run B1 with 5 sources. Hypothesis: the ERA5-trained AI models are
     highly correlated with each other, which would itself be a finding.
- **Effort / risk:** 4–6 pd plus fetch time. Medium risk (fetch cost).
- **Done when:** GenCast appears as a Tier 2 ensemble source, with a
  spread-skill ratio and CRPS next to IFS-ENS and lagged GraphCast.
- **Demo moment:** a spread-skill chart titled "Real AI ensemble vs lagged
  vs physics ensemble over India".

#### B3. Tail repair: quantile mapping plus an extreme-value tail (P0 for the extremes story)

- **Why:** the pitch is about extremes, but POD at ≥115.6 mm is about 0
  beyond day 1, and the extremely-heavy bin is never fittable (F7). AI models
  are known to under-predict heavy rain over India (brief §3.4). The fix is a
  tail, not only more data.
- **How:**
  1. **Per-source, per-region quantile mapping (QM)** before blending: map
     each source's train-period forecast CDF onto IMD's climatological CDF.
     The 15-year local archive gives 1,830 JJAS days, plenty for the upper
     tail.
     - validate the implementation against HEPPI's `Generalized_QM.m` and
       its `NCMRWF_UQM_forecast.nc` output
     - HEPPI's authors found QM better at distribution shape, and EMOS better
       at reliability and heavy rain (brief, Update 3), so test QM *feeding*
       EMOS, not replacing it
  2. **Extreme-value tail for the unfittable bins:**
     - fit a Generalized Pareto distribution above the heavy threshold,
       pooled across regions and leads (shared shape parameter,
       region-specific scale)
     - anchor it on the 15-year IMD extremes
     - splice it onto the CSGD so that
       `P(Y>y) = P_CSGD(Y>u) · (1 + ξ(y−u)/σ)^(−1/ξ)` for `y > u`
     - this replaces today's point-mass fallback with a real, flagged
       `P(≥204.5 mm)`
  3. **Verify with:**
     - tw-CRPS
     - SEDI
     - POD/FAR/CSI at 64.5/115.6/204.5 mm, with a neighbourhood tolerance
       (C2)
     - reliability of `P(≥115.6)`
  4. **Pre-registered go/no-go:** keep it only if it improves tw-CRPS or SEDI
     at ≥3 of 5 leads without making Brier score worse at 7.5 mm.
- **Effort / risk:** 4–5 pd. Medium risk: the tail may stay poorly
  calibrated, in which case keep the grey "not verifiable" overlay and say
  so.
- **Done when:** the extreme-probability map shows real, verified
  probabilities in more cells than today, with the method stated in the view.
- **Demo moment:** a before/after of the same Kerala 2018 day. Before: grey
  "unfittable". After: a verified tail probability.

#### B4. Tier 4, a learned distributional blend: the "AI" in AI–NWP blending (P1)

- **Why:** a judge will ask "where is the AI in *your* system?" The Tier 3
  gating was declined because it had 2 "active" training days. After A1 and
  A2 there are about 240 days × about 18.5k land cells per lead. That is
  enough for a *small* model, if cross-validation is honest.
- **How:**
  1. **Build a distributional regression network (DRN)** after Rasp & Lerch
     (2018, MWR):
     - a small MLP maps per-cell features to CSGD parameters
     - trained by minimising your **own closed-form `csgd_crps`**, ported to
       PyTorch so the loss is the metric
     - features: each source's forecast, IFS-ENS and GenCast mean and spread,
       inter-model spread, lat/lon, elevation, region embedding, day of
       season, lead, and MJO phase (already in the repo)
  2. **Protocol:**
     - train on one year, early-stop on part of it, test on the other year
     - use no random splits, following the project's own rule
  3. **Pre-registered go/no-go**, in the same style as Tier 3: beat the best
     of EMOS/BMA on LOYO CRPS at ≥3 of 5 leads, with the CI excluding 0.
  4. **Explainability:** permutation importance / SHAP per region, giving
     maps of "what the model relies on, where". This extends the weight-map
     deliverable to ML.
- **Effort / risk:** 5–7 pd. CPU is enough. Medium-high risk (overfitting),
  but a well-run "no-go" is still a strong slide.
- **Done when:** a Tier 4 results doc with a pre-registered criterion and an
  honest verdict.
- **Demo moment:** "Our ML is judged by the same scorecard as everything
  else, and here is the verdict."

#### B5. Combine the combiners (P1; also makes slide 2 true)

- **Why:** EMOS-CSG and BMA genuinely trade off (Phase 4's own result, and
  Javanshiri 2021). Slide 2 already claims the strongest combiner is used
  per bin.
- **How:** compare two options on LOYO CRPS:
  - **(a) Per-bin × lead selection.** Pick the combiner with the best train
    CRPS; this is what the slide already claims.
  - **(b) Quantile averaging** (Vincentization) of the EMOS and BMA
    predictive distributions. Lichtendahl et al. (2013) show quantile
    averaging keeps sharpness, whereas the linear pool over-disperses.
- **Effort / risk:** about 2 pd. Low risk.
- **Done when:** a `weavr.stacking` module exists and a results row reports
  the verdict.

### Pillar C: decision products (what the ministry would adopt)

#### C1. District-level IMD colour warnings, with a verified rule (P0)

- **Why:** IMD issues **district-wise, colour-coded, impact-based**
  warnings. The current maps colour 0.25° pixels by a deterministic bin. A
  forecaster needs "which districts go Orange or Red, and how reliable is
  that call".
- **How:**
  1. **Aggregate to districts:** turn `P(≥64.5)`, `P(≥115.6)` and
     `P(≥204.5)` into district values, using the area-fraction or max over
     the cells in each district.
  2. **Probability→colour rule:** make it explicit, with thresholds chosen on
     *train* data. Either maximise ETS or economic value (C4), or match IMD's
     historical warning frequency.
  3. **Verify the colours out-of-sample:** district-level POD, FAR and CSI
     for Orange and Red.
  4. **Frontend:** a district choropleth drawn as a GeoJSON layer on the
     existing OpenStreetMap basemap (`BasemapMap`), with a district list for
     keyboard access and as the fallback. Clicking a district opens a card:
     probabilities, colour, and why (C5).
  - **Boundaries:** use a map consistent with the Survey of India's official
    depiction of India's borders, and check the GeoJSON's licence before
    committing it. Ministry judges notice non-compliant maps. The basemap
    itself draws no national boundaries; the official outline is a
    user-supplied file drawn on top, and district polygons must agree with
    it.
- **Effort / risk:** 3–4 pd. Low-medium risk.
- **Done when:** a new "District warnings" view, plus a verification table
  for the colour calls.
- **Demo moment:** the Kerala 2018 replay rendered as district colours,
  day by day.

#### C2. "Trust scale" and neighbourhood probabilities (P1)

- **Why:** pixel-exact heavy-rain verification double-penalises
  near-misses. Forecasters work at neighbourhood and district scale.
- **How:**
  - FSS is already implemented. For each lead × threshold, find the smallest
    neighbourhood where `FSS ≥ 0.5 + f₀/2`, the Roberts & Lean (2008)
    "useful" criterion. Show it as a caption on every map, e.g.
    "day-3 heavy-rain forecasts are useful at ≥ X km".
  - Offer `P(≥ threshold within 25 km)` as a layer.
- **Effort / risk:** 1–2 pd. Low risk.
- **Done when:** a "useful scale by lead" chart, plus the caption on the
  map views.

#### C3. CAP alert export (P1)

- **Why:** India's national Integrated Alert System (NDMA's SACHET) is built
  on the **Common Alerting Protocol (CAP 1.2)**; confirm against NDMA's
  public documentation. If WEAVR exports CAP, adoption needs no new
  integration project.
- **How:**
  - Map each Orange or Red district warning to a CAP 1.2 `<alert>`:
    - `<info>` with event, urgency, severity, and certainty mapped from the
      probability
    - `<area>` with the district polygon
  - Add a test that validates the output against the OASIS CAP 1.2 XSD.
  - It is **export only**: a downloadable file and an API endpoint. Never
    send alerts.
- **Effort / risk:** 1–2 pd. Low risk.
- **Done when:** a `/api/cap?lead=…` endpoint and an XSD-validated test.
- **Demo moment:** click a Red district and download the CAP XML.

#### C4. Economic value curves by user type (P1)

- **Why:** this turns skill scores into "who benefits, and how much".
  Ministries and impact-minded judges respond to it, and almost no hackathon
  team will show it.
- **How:**
  - Compute Richardson (2000) relative economic value `V(C/L)` from
    contingency tables at many probability thresholds.
  - Plot it for raw GraphCast, IFS-ENS, NEPS-G and WEAVR.
  - Annotate cost/loss bands with example users: a farmer delaying pesticide
    spraying (low C/L), a reservoir operator, a district evacuation decision
    (high C/L).
- **Effort / risk:** about 2 pd, reusing `weavr.verify.contingency_scores`.
  Low risk.
- **Done when:** a "Value" view per threshold and lead.

#### C5. "Why this forecast?" explainer and confidence rating (P1)

- **Why:** trust. The PS asks for weight maps; forecasters need
  *per-location* reasoning.
- **How:** clicking a cell or district shows:
  - each source's forecast, weight and contribution (weight × forecast)
  - ensemble spread
  - the rain bin, and whether the fit was a fallback
  - each source's historical skill in that region × lead
  - a **confidence rating** (High/Medium/Low) from spread and inter-model
    disagreement, calibrated on the historical spread-error relationship
  - also add a model-disagreement map as its own layer
- **Effort / risk:** 2–3 pd. Low risk.

#### C6. Temperature and heat-wave blending (P1 if the PS lists temperature; verify)

- **Why:** the brief's framing names rainfall, temperature *and* wind.
  Temperature is the easiest to add, and Pangu finally contributes (it has
  temperature).
- **How:**
  1. `2m_temperature` is already in the baseline store for all four sources.
     Ground truth is IMD's 1° gridded Tmax/Tmin via `imdlib`; coarsen the
     forecasts to 1°.
  2. **Caveat:** WeatherBench 2 gives *instantaneous* 2 m temperature at
     synoptic hours. Daily Tmax is approximately the max over 6-hourly steps,
     but the peak near 09 UTC isn't sampled, so expect a cold bias. EMOS/NGR
     learns that offset; also check whether HRES has a max-temperature
     variable.
  3. **Method:** Gaussian EMOS (NGR), which has a closed-form CRPS and is far
     easier than rain.
  4. **Heat-wave probabilities** from IMD's criteria: Tmax ≥ 40 °C in the
     plains and departure ≥ 4.5 °C, or Tmax ≥ 45 °C. Early June is inside
     JJAS, and both GraphCast windows include the April–June heat season.
  - **Wind** is a stretch goal: there is no IMD gridded wind, so ground truth
    would be ERA5, which the project avoids for rain. Only do it if time is
    left.
- **Effort / risk:** 4–5 pd. Low-medium risk.

#### C7. North-east monsoon extension (October–December) (P2)

- Both GraphCast windows cover October–December (2018 and 2020), and IMD
  observations exist year-round.
- This makes the Chennai/Tamil Nadu motivation legitimate, and adds a real
  "season" dimension to the weights, which the PS asks for.
- Mostly configuration once A1 and A2 exist. About 3 pd.

### Pillar D: operations and demo

#### D1. Event replay: Kerala, August 2018 (P0 for the pitch)

- **Why:** judges remember a story. The Kerala 2018 floods are within
  GraphCast 2018, HRES, IFS-ENS, NEPS-G (day 1) and IMD. **Train on 2020
  only**, so the replay is truly out-of-sample.
- **How:**
  1. Build a "Replay" view: pick a date, and see the day-5 → day-1 evolution
     of `P(≥115.6)` and `P(≥204.5)` and the district colours for Kerala.
     Show them against the raw models and against the IMD observed field.
  2. Report the **first lead time at which WEAVR would have issued Orange
     and Red**, next to each raw model.
  3. Second cases:
     - early July 2019 Mumbai (NEPS-G, IFS and HRES; no GraphCast)
     - 2020 events within the season
  4. **Pre-register** the replay criteria, and commit to showing it whatever
     it shows. A miss, explained through the tail analysis (B3), is still a
     strong and honest slide.
  5. **"Judge's choice":** a date picker over every JJAS day in
     2018/2019/2020 that replays any day live during the finale.
- **Effort / risk:** 3–4 pd after A2 and A3. The risk is a disappointing
  result, which is handled by honesty and B3.
- **Demo moment:** the centrepiece of the pitch.

#### D2. A live pipeline and self-healing weights (P1)

- **Why:** "operational" is only credible if something runs every day. Drift
  detection cannot fire until history accumulates (README, Phase 6).
- **How:**
  1. Add a scheduled workflow now, so history starts accumulating. Only
     `ci.yml` exists today; the README's "one-line addition" is really a new
     `daily-pipeline.yml`. Wrap the ECMWF client in an explicit timeout;
     your own docstring flags its 500-retry hang.
  2. **Close the loop from drift to adaptation.** Use online weight updating
     (exponentially weighted / Hedge-style updates from daily losses, e.g.
     Thorey, Mallet & Baudin 2017, online learning with the CRPS for
     ensemble forecasting), starting from the Tier 1 weights as a prior.
     This gradually fixes the AIFS-for-GraphCast mismatch the README admits,
     and handles model upgrades without refitting from scratch.
  3. **Live observations:** before accepting "no current-year IMD data",
     check `imdlib`'s separate **real-time** API (`get_real_data`), which is
     distinct from the yearly archive call the pipeline tried. If it works,
     live verification becomes real.
- **Effort / risk:** 3–4 pd. Medium risk (live dependencies). Keep a cached
  fallback for the demo.
- **Done when:** a "Today" page with the latest blended forecast, plus a
  weights-over-time chart that visibly adapts.

#### D3. A public URL with zero server (P1)

- The API is read-only over committed files. Export every API response to
  static JSON with a small `scripts/export_static_site.py`, and point
  `api.js` at those files in a "static mode". Host `dashboard-web/` on
  GitHub Pages or Netlify.
- Put the URL and a QR code on the slides so judges can open it on their
  phones.
- This also permanently fixes the VS Code "Show Preview" 404s, because the
  static JSON is served from any static server.
- **Effort:** 1–2 pd. Keep FastAPI for development.

#### D4. Reproducibility and a model card (P1)

- One command (`make reproduce` or `scripts/reproduce_all.py`) rebuilds the
  stores, runs all tiers and exports the dashboard data.
- Add a `Dockerfile`.
- Add a one-page **model card**: inputs, training period, known limits,
  fallbacks, intended use, and what not to use it for.
- **Effort:** 1–2 pd. It answers "can we run this at IMD?" with a command.

---

## 5. Uniqueness matrix

| Topic | What most teams will do | What WEAVR can uniquely show |
|---|---|---|
| Models | 2–3 WeatherBench 2 models | + **NEPS-G** (India's own ensemble, via recovered HEPPI dates) + GenCast / FuXi |
| Weighting | XGBoost / LSTM weights | **Independence-aware weights, N_eff and Shapley contribution maps** |
| AI | "our model is AI" | Tier 4 DRN with a **pre-registered go/no-go**, judged by the same scorecard |
| Extremes | colour a deterministic map | **Tail repair** (QM + EVT) and verified `P(≥204.5 mm)` |
| Verification | RMSE / accuracy | **ECMWF-style scorecard with CIs**, reliability, SEDI, tw-CRPS, and a published benchmark (Angus et al. 2024) |
| Products | pixel map | **District IMD colour warnings**, CAP 1.2 export, economic-value curves |
| Demo | today's map | **Out-of-sample Kerala 2018 replay**, plus "judge's choice" replay |
| Ops | none | live scheduled pipeline, renormalization and self-healing weights |

---

## 6. What NOT to do

- **Don't redo regime gating** on the same data. Re-run the existing Tier 3
  go/no-go once A1/A2 land (it's cheap), and keep the honest result either
  way.
- **Don't add an LLM chatbot as a headline feature.** Many teams will, and
  generated warning text can hallucinate. If you want bulletins, use
  deterministic templates filled from C1, optionally translated.
- **Don't use NeuralGCM or Aurora** (too coarse, or no 2018/2020 overlap),
  and don't chase Pangu rainfall (it does not exist in the archive).
- **Don't fit per-grid-cell weights.** Wanders & Wood needed ≥6 years for
  stable precipitation weights; keep regional pooling.
- **Don't claim "operational"** until D2 has run for real for at least 2
  weeks.
- **Don't rebuild the frontend** in React or add a *second* map stack. The
  vanilla JS stays. The one map library is the already-vendored MapLibre
  basemap (no build step, fully offline); new map views reuse it. Spend the
  time on C1–C5.
- **Don't start large data fetches at the finale venue.** Pre-build every
  store, keep a backup drive, and make the demo work offline.

---

## 7. Roadmap

Adjust to the official SIH 2026 calendar (idea-submission deadline,
shortlisting, grand finale).

### Wave 0: this week (≤3 days)

1. **A5:** fix the slide claims, and add F2 and F3 to the docs.
2. **A1:** start the daily fetches overnight (baseline store first, then
   IFS-ENS with a thread pool).
3. **A3, part 1:** commit the HEPPI date-map script and CSV; confirm the lead.
4. **A2, part 1:** measure one GraphCast-2018 chunk, and correct
   `docs/baseline-store.md`.

### Wave 1: next 2–3 weeks (the evidence and the signature idea)

- A1 re-run → A2 (LOYO in tiers 1–3) → A3 (NEPS-G in the blend, and the
  HEPPI head-to-head) → A4 (scorecard) → B1 (N_eff, Shapley, shrinkage
  weights).
- **Exit:** a new results summary with CIs, and a slide deck rewritten
  around those numbers.

### Wave 2: before the finale (products, extremes, demo)

- B3 → C1 → D1 (the replay depends on A2, A3, B3 and C1) → C2 → C3 → C4 →
  C5 → D3 → D4.
- Then **one** of {B2, B4, C6}, depending on team size. Rule of thumb: a
  large team does B4 (the "AI" answer); a small team does C6 (PS coverage).
- B5 and D2 fit in wherever there is slack. D2's scheduled workflow should
  start as early as possible (prompt 10) so history accumulates.

Step-by-step prompts that execute this plan are in [`prompts/`](prompts/README.md).

### Wave 3: the finale itself (36 h)

- Have all stores and results pre-built. Rehearse the demo offline.
- **Plan things to build live** so mentors and judges see progress: polish
  the CAP export, add districts to "judge's choice" replay, add a bilingual
  template bulletin, and handle mentor-requested views.
- Keep a written log of what was built at the venue versus before, in case
  the jury asks.

### Effort summary

| Item | Priority | Effort (pd) | Depends on |
|---|---|---|---|
| A5 Claims audit | P0 | 0.5 | none |
| A1 Daily sampling | P0 | 2–3 (+fetch) | none |
| A2 2018 season + LOYO | P0 | 3–4 (+fetch) | A1 (same builder changes) |
| A3 NEPS-G via HEPPI | P0 | 3–5 | A2 (2018 obs/sources) |
| A4 Scorecard + significance | P0 | 3 | A1 |
| B1 Independence / N_eff / Shapley | P0 | 2–3 | A1, A2 (A3 to include NEPS-G) |
| B3 Tail repair (QM + EVT) | P0 | 4–5 | A1, A4 |
| C1 District warnings | P0 | 3–4 | B3 (for real probabilities) |
| D1 Kerala 2018 replay | P0 | 3–4 | A2, A3, C1 |
| B2 GenCast + FuXi | P1 | 4–6 (+fetch) | A1 |
| B4 Tier 4 DRN | P1 | 5–7 | A1, A2, A4 |
| B5 Combine combiners | P1 | 2 | A2 |
| C2 Trust scale | P1 | 1–2 | A1 |
| C3 CAP export | P1 | 1–2 | C1 |
| C4 Economic value | P1 | 2 | A4 |
| C5 Explainer + confidence | P1 | 2–3 | B1 |
| C6 Temperature / heat | P1 | 4–5 | A1 |
| D2 Live + self-healing | P1 | 3–4 | none (start early) |
| D3 Static public URL | P1 | 1–2 | none |
| D4 Repro + model card | P1 | 1–2 | none |
| C7 OND extension | P2 | 3 | A1, A2 |

P0 total ≈ 24–33 person-days. With P1 it is ≈ 50–70.

### Team split

The merge-sequence doc assumes 2 people; SIH teams are up to 6.

- **6 people:**
  - Data (A1, A2, A3, B2): 1–2
  - Science (A4, B1, B3, B4, B5): 2
  - Products and frontend (C1–C5, D3): 1–2
  - Ops and pitch (D1 storyline, D2, D4, slides, Q&A drills): 1
- **2 people:**
  - Person 1: A1 → A2 → A3 → A4 → B1 → B3
  - Person 2: A5 → D3 → C2 → C1 → C3 → C4 → the D1 UI → slides
  - Keep the existing "one PR per step, merge in order" discipline.

---

## 8. The demo storyline (about 6 minutes)

1. **Hook (30 s):** August 2018, Kerala. The forecast models existed; the
   warning still wasn't sharp enough.
2. **Problem (30 s):** single models; AI models that are fast but smooth and
   deterministic; no Indian blend verified on IMD gauges.
3. **Insight (60 s):** "Three global models, about 1.1 independent opinions"
   (F3 chart), then what India's own NEPS-G adds (B1 + A3, as measured).
4. **Replay (90 s):** Kerala 2018, day 5 → day 1, district colours from
   WEAVR vs the raw models vs the IMD observed field. The model was trained
   on 2020 only.
5. **Products (60 s):** click a district → why this forecast → the CAP XML
   download → the economic-value curve for a reservoir operator.
6. **Proof (60 s):** the scorecard with CIs, the honest limits, and the
   pre-registered go/no-go results (including the ones that failed).
7. **Live (30 s):** today's blended forecast from the scheduled pipeline.
   Toggle "GraphCast feed missing" and show the weights renormalize.

---

## 9. Q&A preparation: the hard questions

| Likely question | Prepared answer (after the relevant items land) |
|---|---|
| Does the blend beat the best single model? | Show A4's "blend − best single member (chosen on train)" row, with CIs, on RMSE *and* CRPS / heavy-rain scores. Today's honest answer is F2. |
| Your test set was 4 days? | That was true; it is now about 120 days × 2 seasons with leave-one-year-out CV and block-bootstrap CIs (A1, A2, A4). |
| Where are NCUM / NEPS-G? | NEPS-G is included via the HEPPI dataset, with dates recovered by fingerprint matching against IMD (F4, A3). NCUM needs NCMRWF archive access; ask the organisers, as originally planned. |
| Where is the AI in your blending? | Tier 4 DRN, trained on your own closed-form CRPS, with a pre-registered verdict (B4). The AI *models* are inputs; the blend itself can be learned. |
| GraphCast is trained on ERA5 from IFS. Isn't it redundant? | Yes, measured: error correlation 0.88–0.93. That is exactly why WEAVR weights by independence and reports N_eff (B1). |
| Why is HRES's weight zero? | Collinear with IFS-ENS (0.82–0.92), which already carries its information; the Shapley view shows its marginal contribution (B1). |
| How do you forecast extremes you've never seen in training? | A pooled extreme-value tail anchored on 15 years of IMD extremes, verified with SEDI and tw-CRPS, with a visible flag where it can't be verified (B3). |
| How would IMD use this tomorrow? | District colour warnings, CAP 1.2 export, a daily scheduled pipeline, `make reproduce`, and a model card (C1, C3, D2, D4). |
| What if a model feed is late? | Live toggle: renormalization (existing `weavr.renormalize`), then online weight adaptation (D2). |
| Model upgrades? | Drift detection feeding online reweighting (D2). The 2018 vs 2020 GraphCast versions act as a natural test (A2). |
| Why not per-grid-cell weights? | Wanders & Wood needed ≥6 years for stable precipitation weights; regional pooling is the evidence-based choice. |
| Why only June–September? | Scope: JJAS is ~70–75% of annual rainfall. The October–December extension is possible with the same archives (C7). |
| Compute cost? | Statistical post-processing on a laptop CPU. No GPU; the AI models' published runs are consumed, not re-run. |
| What failed? | Tier 3 regime gating (no-go, with the reasons), plus whatever else fails its pre-registered criterion. Show them. |

---

## 10. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Fetches are slow or stall (IFS-ENS, GenCast) | Delays A1, A2, B2 | Measure one chunk first; thread pool plus the existing staging/resume; member subsets; run overnight early |
| More data makes the results *weaker* | Pitch numbers change | Pre-register, report honestly, lean on the products and the replay; the honesty brand turns this into credibility |
| HEPPI licence or lead-time ambiguity | A3 challenged | Commit only derived metadata and scores, cite Angus et al. 2024, confirm the lead empirically, email the authors |
| The GraphCast 2018 archive has other quirks | Delays A2 | One-chunk probe; per-source adapters already exist in the builder pattern |
| The Kerala replay misses the event | A weak demo moment | Show it anyway, with the tail analysis (B3); add the Mumbai 2019 case; "judge's choice" shows it wasn't cherry-picked |
| The India boundary GeoJSON is non-compliant | Embarrassment with the ministry jury | Use a Survey of India–compliant depiction, and check the licence. The basemap draws no borders; `check_basemap_boundaries.py` guards that |
| The 148 MB basemap tile file can't be pushed or hosted as-is | Public site loses its map | Lower-zoom public extract or separate hosting; plain-grid fallback already works; full file stays in the offline bundle |
| Live dependencies fail during the finale | A broken demo | Cached "today" snapshot and an offline static build (D3) |
| Scope creep | Nothing finishes | Finish P0 before any P1; each item has a "done when" |

---

## Appendix A: reproduce the verified facts

Run each snippet from `WEAVR/` with the venv active.

**F2: single sources vs blends on the same test days**

```python
import sys, pandas as pd, xarray as xr
sys.path.insert(0, "scripts")
from run_tier0_baseline import _align_to_imd_day, PRECIP_M_TO_MM
from weavr.splits import seasonal_block_split
from weavr import verify as V
st = "data/baseline_2020_jjas.zarr"
obs = xr.open_zarr(st, group="imd_observed").load()
for lead in [24, 48, 72, 96, 120]:
    for s in ("graphcast", "hres", "ifs_ens_mean"):
        da = _align_to_imd_day(xr.open_zarr(st, group=s)["total_precipitation_24hr"]
                               .sel(prediction_timedelta=lead).load() * PRECIP_M_TO_MM, lead)
        o = obs["rain"].reindex(time=da["sample"].values).rename(time="sample")
        keep = ~o.isnull().all(dim=["latitude", "longitude"])
        da, o = da.isel(sample=keep.values), o.isel(sample=keep.values)
        _, test = seasonal_block_split(pd.DatetimeIndex(da["sample"].values), test_fraction=0.2)
        print(lead, s, float(V.rmse(da.isel(sample=test), o.isel(sample=test))))
```

**F3: error correlation**

Same alignment as F2. Take `err = forecast_mm − imd` for each source,
flatten it over (sample, lat, lon), keep cells finite in all three sources,
then run `np.corrcoef`. Note that WeatherBench 2 rainfall is in **metres**;
multiply by 1000 first. Forgetting this gives a spurious correlation of 1.0.

**F4: HEPPI date recovery**

```python
import xarray as xr, numpy as np, pandas as pd
arrs, dates = [], []
for y in range(2006, 2021):
    r = xr.open_zarr("data/imd_seeps_climatology_jjas.zarr", group=f"y{y}")["rain"].transpose("time", ...)
    arrs.append(r.values); dates.extend(pd.to_datetime(r["time"].values))
clim = np.nan_to_num(np.concatenate(arrs).reshape(len(dates), -1), nan=-1)
obs = xr.open_dataset("../HEPPI/IMD_observed.nc")["IMD_rainfall_observed"] \
        .transpose("forecast", "lat", "lon").values
for i in range(obs.shape[0]):
    o = np.nan_to_num(obs[i].reshape(-1), nan=-1); m = o >= 0
    mae = np.abs(clim[:, m] - o[m]).mean(axis=1)
    best = dates[int(np.argmin(mae))]          # best-matching IMD day for HEPPI index i
```

- The hypothesised mapping:
  - `i ∈ [0, 119]` → `2018-06-02 + i days`, plus 1 day from 24 June onward
    (the gap)
  - `i ∈ [181, 302]` → `2019-06-01 + (i − 181) days`
- Result: the mapped date is the best match for 241 of 242 days. Shifted by
  −1 day it is the best for 1 of 241; shifted by +1 day, for 0 of 240.

**F5 and F6: archive contents**

```python
import xarray as xr
p = "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours_derived.zarr"
ds = xr.open_zarr(p, storage_options={"token": "anon"})
print([v for v in ds.data_vars if "precip" in v], ds.time.values[[0, -1]], dict(ds.sizes))
# likewise: gencast/2020-1440x721.zarr, fuxi/2020-1440x721.zarr
```

---

## Appendix B: references this plan adds

These are not already in `ppt/content/slide4.md`.

- Bates, J. M. & Granger, C. W. J. (1969). The combination of forecasts. *OR Quarterly.*
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivariate Analysis.*
- Wang et al. (2025). Correlation-weighted averaging as the best linear unbiased estimator; clipping negative weights. *GRL* (already in the research brief).
- Knutti et al. (2017); Brunner et al. (2019). Performance-plus-independence weighting (already in the brief).
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *J. Business & Economic Statistics.*
- Gneiting, T. & Ranjan, R. (2011). Threshold- and quantile-weighted scoring rules. *J. Business & Economic Statistics.*
- Ferro, C. A. T. & Stephenson, D. B. (2011). Extremal dependence indices (SEDI). *Weather and Forecasting.*
- Richardson, D. S. (2000). Skill and relative economic value of the ECMWF EPS. *QJRMS.*
- Rasp, S. & Lerch, S. (2018). Neural networks for postprocessing ensemble weather forecasts. *Monthly Weather Review.*
- Lichtendahl, K. C., Grushka-Cockayne, Y. & Winkler, R. L. (2013). Is it better to average probabilities or quantiles? *Management Science.*
- Thorey, J., Mallet, V. & Baudin, P. (2017). Online learning with the CRPS for ensemble forecasting. *QJRMS.*
- Price, I. et al. (2024). Probabilistic weather forecasting with machine learning (GenCast). *Nature.*
- Chen, L. et al. (2023). FuXi: a cascade ML forecasting system for 15-day global weather forecast. *npj Climate and Atmospheric Science.*
- Angus, M. et al. (2024). NEPS-G post-processing over India (EMOS vs QM). *QJRMS* (the HEPPI benchmark).
- Roberts, N. M. & Lean, H. W. (2008). FSS "useful scale" criterion (already cited).
- OASIS (2010). Common Alerting Protocol v1.2. NDMA SACHET / Integrated Alert System (verify the CAP profile from NDMA's public docs).
