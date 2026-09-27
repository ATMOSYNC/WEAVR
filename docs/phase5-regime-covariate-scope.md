# Phase 5 scoping: regime-covariate feasibility

Answers, checked against real published definitions and real data-source
coverage (not assumed from issue #7's phrasing), to the open questions in
[`solving issues/06-phase-5-regime-conditioned-weighting/01-check-regime-covariate-feasibility.md`](../../solving%20issues/06-phase-5-regime-conditioned-weighting/01-check-regime-covariate-feasibility.md)
before any regime-covariate code is written.

## Monsoon active/break: real, precisely defined, free (no new fetch)

Confirmed via **Rajeevan, Gadgil & Bhate (2010)**, "Active and break spells
of the Indian summer monsoon," *Journal of Earth System Science* 119,
229-247 — IMD's own real operational definition: during the peak monsoon
months (July-August), a day is **active** if the normalized rainfall
anomaly over the **monsoon core zone** (18-28°N, 65-88°E) exceeds +1, or
**break** if it is below -1, each sustained for **at least 3 consecutive
days**. The core zone sits entirely within weavr's own common grid (lat
6.5-38.5, lon 66.5-100.0, confirmed against `src/weavr/grid.py`), and the
normalized anomaly needs only a climatological mean/std of core-zone-
averaged rainfall — computable directly from `imd_observed.rain`, already
in `data/baseline_2020_jjas.zarr`, using the same multi-year IMD-only
climatology machinery `scripts/build_seeps_climatology.py` already builds
for SEEPS. **Decision: in scope, no new external data needed.**

One real, checked scope limitation: 2020's JJAS window is June-September,
but Rajeevan et al.'s own definition is specifically for **July-August**
(the peak monsoon months) — June and September days will need either an
honestly-labeled extension of the same anomaly/duration rule outside its
originally-validated window, or explicit exclusion from active/break
labeling. This is step 2's call to make and document, not a reason to
avoid building the covariate at all.

## MJO phase: real, real-time, but the two standard indices split on 2020 accessibility

- **Wheeler & Hendon (2004)'s RMM index**, hosted by the Australian Bureau
  of Meteorology (`bom.gov.au/climate/mjo/`), is the standard, most-cited
  MJO index and is continuously updated in real time since 1974 — it
  certainly covers 2020. However, **this environment's own fetch attempt
  against BoM's site returned HTTP 403** (likely bot/scraping protection),
  so direct accessibility from wherever step 2 actually runs needs
  re-confirming, not assumed from this finding alone.
- **NOAA PSL's own "RMM\*" mirror** was directly checked and ruled out:
  it only populates **November-April** (winter-season) values, with
  summer months explicitly set to missing — useless for a JJAS-only
  project.
- **Kiladis, Dias, Straub, Wheeler, Tulich, Kikuchi, Weickmann & Ventrice
  (2014)**, "A Comparison of OLR and Circulation-Based Indices for
  Tracking the MJO," *Monthly Weather Review* 142(5), 1697-1715 — the
  **OMI (OLR-based MJO Index)**, a real, peer-reviewed, full-year,
  continuously-updated (through at least mid-2026, confirmed via NOAA
  PSL's own MJO page) daily index hosted directly on a NOAA government
  domain (`psl.noaa.gov/mjo/`), a materially different access risk than
  BoM's site. **Decision: use OMI as the concrete MJO covariate**,
  re-attempting BoM's RMM index only as a fallback if step 2's own fetch
  environment can reach it (RMM is the more commonly cited index in the
  literature, but OMI measures the same underlying phenomenon and is the
  one this step could actually confirm end-to-end).
- **BSISO (APCC/Kikuchi)**: real and peer-reviewed (Kikuchi 2021 review,
  *Journal of the Meteorological Society of Japan*), but this
  environment's direct fetch of the index-hosting domain
  (`iprc.soest.hawaii.edu`) failed at DNS resolution, not merely a 403 —
  a harder access failure than either MJO source. **Decision: not
  pursued** — MJO (via OMI) already covers issue #7's named "BSISO/MJO
  phase" option, and chasing a second, harder-to-reach intraseasonal index
  for the same broad phenomenon isn't warranted for a timeboxed phase.

## Monsoon-depression presence: real and downloadable; western-disturbance presence real but not meaningful for JJAS-only data

- **Vishnu, Boos, Ullrich & O'Brien (2020)**'s "optimized tracking
  algorithm" for South Asian monsoon lows/depressions (published in
  *JGR: Atmospheres*) is the real, peer-reviewed methodological basis for
  low-pressure-system tracking over India. Their own original static
  archive (Zenodo record 3890646) only covers ERA5 through **2019** — does
  **not** include 2020, checked directly against that record's own
  metadata.
- A maintained, extended catalogue built on the same broad approach,
  **kieranmrhunt/monsoon-low-atlas** ("v5.6", ERA5-derived, 1940-2025,
  Parquet/CSV via Zenodo), **does cover 2020** — confirmed directly. Its
  own README does not state a clear citation/authorship chain back to a
  specific published paper for this extended version, which step 2 must
  re-verify (and cite honestly, including this exact provenance gap, if it
  can't be fully resolved) before using it, rather than presenting it as
  more rigorously validated than confirmed here.
- **Decision: monsoon-depression presence is in scope**, sourced from the
  extended catalogue above, with step 2 re-confirming its provenance.
- **Western-disturbance presence specifically is real but ruled out for
  this project's actual data**: checked directly, western disturbances are
  reported as roughly **50x rarer in August than January**, and remain
  "relatively rare" in JJAS even accounting for a documented recent summer
  increase. A JJAS-only, single-season (2020) covariate for this
  phenomenon would carry almost no realized variance to condition a
  weighting model on — not a data-availability problem, a signal problem.
  **Decision: out of scope for this project's current single-season
  data**, revisit if a future phase extends beyond JJAS-only sampling.

## Neal et al.'s 30-pattern classification: real, but doesn't cover this project's actual year

`docs/phase3-cv-and-regional-scheme.md` already flagged **Neal, Robbins,
Dankers et al.**'s "Weather pattern definitions for India and their daily
historical classifications" (PANGAEA, DOI 10.1594/PANGAEA.902030, from
Neal et al. 2020, *International Journal of Climatology*) as a real,
day-level synoptic regime classification matching this phase's need.
Checked directly against the dataset's own PANGAEA record: it covers
**1979 to 2016 only** — it does **not** include 2020, this project's only
data year. Reproducing it for 2020 would mean re-implementing Neal et
al.'s own weather-pattern clustering methodology against real ERA5/
ERA-Interim wind fields from scratch, not downloading an existing product
— disproportionate new data-science engineering for a phase issue #7
itself says to timebox. **Decision: out of scope.**

## Genuine tradeoff surfaced to the user, decided

Given all three real, confirmed-feasible covariates above (monsoon
active/break, MJO phase via OMI, monsoon-depression presence) are each
individually a reasonable amount of work for a timeboxed phase — none
prohibitively expensive on its own — whether to build all three now
(broader real-signal coverage, more upfront engineering: 2 new small
external fetches) or start minimal with just active/break (zero new
fetches, fastest path to testing whether regime-conditioning shows any
promise at all before investing further) was a genuine cost/benefit
choice, not a forced answer. Put to the user via `AskUserQuestion`.
**Decision: build all three now** — step 2 implements monsoon active/
break, MJO phase (OMI), and monsoon-depression presence.

## Summary of decisions for step 2

| Covariate | Decision | Basis |
|---|---|---|
| Monsoon active/break | In scope, no new fetch | Rajeevan, Gadgil & Bhate (2010); derivable from `imd_observed.rain` directly |
| MJO phase | In scope, new fetch (OMI) | Kiladis et al. (2014), hosted on NOAA PSL; BoM's RMM as a fallback if reachable from step 2's own environment |
| Monsoon-depression presence | In scope, new fetch | Vishnu et al. (2020) methodology; kieranmrhunt/monsoon-low-atlas's extended catalogue, provenance to be re-verified in step 2 |
| Western-disturbance presence | Out of scope | Real data likely exists, but ~50x rarer in JJAS than winter — near-zero signal in a JJAS-only single season |
| Neal et al. (2022) 30-pattern classification | Out of scope | Real dataset, but covers only 1979-2016, not this project's 2020 season; reproducing it needs new clustering methodology, not a data fetch |

## Step 2 results: all three built, real per-category counts

All three in-scope covariates were built for real against 2020 JJAS
(`src/weavr/regimes.py`, `scripts/fetch_omi_mjo_index.py`,
`scripts/build_monsoon_depression_index.py`):

- **Monsoon active/break**: of 122 real JJAS days, **14 active** (2020-08-13
  to 08-17 and 08-20 to 08-24 and 08-27 to 08-30, three separate ≥3-day
  spells) and **3 break** (2020-09-28 to 09-30, at monsoon withdrawal);
  105 neutral. Spot-checked against a real, independently reported source
  (not just internal consistency): contemporary coverage of the 2020
  monsoon season reports "a very active monsoon from August 11 to 14"
  and renewed heavy rain "from August 18" — this classifier's own computed
  active spells (Aug 13-17, Aug 20-24) land within 1-2 days of both real
  reported windows, a real, checked correspondence (not an exact match,
  expected given the pooled-climatology simplification documented above).
- **MJO phase**: real OMI PC1/PC2 fetched for all 122 JJAS days; every one
  of the 8 phases occurs at least 7 times across the season (phase counts:
  1:10, 2:7, 3:23, 4:17, 5:12, 6:13, 7:17, 8:23) — a real, usable spread
  for conditioning, not concentrated in one or two phases.
- **Monsoon-depression presence**: 2,059 real hourly low-pressure-system
  positions found in the India domain over 2020 JJAS (1,886 "low", 109
  "depression", 44 "deep_depression", 20 "cyclonic_storm"); after
  requiring depression-or-stronger (`imd_category >= 2`) and collapsing to
  IMD days, **14 of 122 days** have a real depression (or stronger)
  present.

Provenance note, resolved: the v5.6 catalogue flagged as unverified in
step 1 is confirmed still unpublished (`"zenodo_status": "not_published"`
in its own release manifest, checked again here). **v5.5.1 is the version
actually used** — confirmed published (Zenodo DOI 10.5281/zenodo.22142640,
`"zenodo_status": "published"` in its own release manifest) and
successfully downloaded (61MB parquet, 287,773 real rows, 58 columns
including a direct IMD-equivalent `imd_category`/`imd_label` classification
this project uses as-is rather than re-deriving one from raw intensity
fields).
