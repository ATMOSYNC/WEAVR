# Phase 3 scoping: CV strategy and regional pooling scheme

Answers, checked against the real data and real cited literature, to the
two open questions in
[`solving issues/04-phase-3-static-regional-skill-weighting/01-check-cv-and-regional-scheme-feasibility.md`](../../solving%20issues/04-phase-3-static-regional-skill-weighting/01-check-cv-and-regional-scheme-feasibility.md)
before any weight-fitting code is written.

## CV strategy: `seasonal_block_split` for every source

Issue #5 names leave-one-year-out CV (Wanders & Wood 2016) explicitly.
Checked directly against the real WeatherBench 2 archives' calendar
coverage (not re-quoting `src/weavr/splits.py`'s docstring without
re-verifying it):

| Source | Archive range | Full JJAS seasons available |
|---|---|---|
| GraphCast | 2019-11-16 .. 2021-01-31 | **2020 only — one season, ever** |
| Pangu | 2018-01-01 .. 2022-12-31 | 2018, 2019, 2020, 2021, 2022 (5) |
| HRES | 2016-01-01 .. 2023-01-10 | 2016-2022 (7) |
| `ifs_ens_mean` | 2018-01-01 .. 2022-12-31 | 2018-2022 (5) |

**GraphCast's public archive contains exactly one full JJAS monsoon
season, permanently** — its date range simply doesn't span a second one;
no additional fetch, however costly, can produce one. Since GraphCast is
the only AI-model source that carries `total_precipitation_24hr` (Pangu
has no precipitation variable at all, per `docs/baseline-store.md`), true
leave-one-year-out CV is impossible for the one variable/source
combination this project actually scores.

Pangu, HRES, and `ifs_ens_mean` do have multiple JJAS seasons available,
so a real leave-one-year-out fit is technically reachable for them if the
baseline store were extended to pull those extra years (~45 min estimated
fetch for HRES + `ifs_ens_mean` at Phase 0/1's measured per-chunk latency
and weekly sampling density — tractable, not a Phase 1/2-style
intractable pull).

This was a genuine cost/benefit tradeoff, not a single obviously-correct
answer, so it was put to the user directly rather than decided silently.
**Decision: `seasonal_block_split` (within the single 2020 JJAS season) is
used for every source's weight fit, including HRES and `ifs_ens_mean` even
though they could support a more rigorous split.** Reasoning: keeping one
CV methodology across every source in the same blend keeps their fitted
weights comparable and avoids a real methodological inconsistency (some
sources' weights coming from a fundamentally different, more rigorous data
regime than others' within the same weighted average) — and avoids the
added engineering complexity of a mixed-CV fitting path for a two-source
partial gain. This is a deliberate, documented deviation from Wanders &
Wood's literal recipe, forced by GraphCast's real data ceiling, not an
oversight.

If a future phase adds a genuinely multi-year paired store (e.g. once
NEPS-G access or a wider GraphCast archive appears), revisit this — the
constraint is about *today's* data, not a permanent methodological choice.

## Regional pooling scheme: Sreekala & Babu's 6-zone scheme

Issue #5 frames this as a choice between Neal et al.'s 30-pattern k-means
regimes and Sreekala & Babu's simpler 6-zone scheme. Checked directly
(not assumed from the issue's phrasing) what each actually is:

- **Sreekala & Babu (Sreekala, P.P. & Babu, C.A., "Identification and
  Variability Analysis of New Homogeneous Summer Monsoon Rainfall Regions
  Over India by Using K-Means Clustering Technique," *International
  Journal of Climatology*, Royal Meteorological Society/Wiley, 26 Jan
  2025) is a **spatial** classification: India's summer monsoon rainfall
  is k-means clustered into 6 homogeneous geographic zones — **West Coast
  India (WC), Southeast India (SI), West India (WI), Central India (CI),
  Northeast India 1 (NE1), Northeast India 2 (NE2)**. This is exactly the
  kind of static, geography-based pooling Phase 3's "static/regional skill
  weighting" needs.
- **Neal et al. (2022)'s "30 weather patterns for India"** are, checked
  directly, a **temporal** classification — each *day* is assigned to one
  of 30 large-scale synoptic weather-pattern regimes (used for medium-range
  predictability studies), not a spatial partition of gridpoints into 30
  geographic regions at all. Issue #5's framing of these two as
  interchangeable spatial-pooling alternatives doesn't hold up: Neal et
  al.'s regime classification is a *day-level* covariate, matching almost
  exactly Phase 5's own phase-plan description ("regime covariates: monsoon
  active/break, BSISO/MJO phase") rather than Phase 3's spatial-region
  need. **This makes the choice between them not a real tradeoff to weigh
  for Phase 3** — Sreekala & Babu's scheme is the one that actually fits
  what "regional" means here; Neal et al.'s belongs to Phase 5's
  regime-conditioned weighting, not this phase. Worth flagging back for
  the phase-plan's own wording, which conflates the two.

**Data-density check** (so the decision isn't just "the other option
doesn't apply," but also confirmed to be adequate on its own terms): IMD's
real gridded rainfall has 4625 valid (land) gridpoints on the common 0.25°
grid, and Phase 1/2's 18 weekly JJAS samples give, for a uniform 6-region
split, roughly 771 gridpoints × 18 samples ≈ **13,875 space×time data
points per region per lead** to fit one weight from — even a hypothetical
30-region split would still give ≈2,775 per region per lead. Neither
scheme is data-starved at this pooling granularity; the "limited AI
hindcast years" problem issue #5 names is specifically about *per-gridpoint*
fitting (which needs ≥6-11 years for stability), and pooling into even a
handful of regions is exactly what makes one season workable.

**Exact zone boundaries**: Sreekala & Babu's published k-means cluster
assignments (the precise gridpoint-level result of their clustering) are
not reproducible from this environment without their original input
dataset and methodology — only the paper's 6 named zones and their broad
geographic identity are available. Step 2 will approximate each of the 6
named zones (WC, SI, WI, CI, NE1, NE2) with reasonable, documented lat/lon
boundaries matching their commonly-understood geographic identity (India's
west coast, southeast, west, central, and two northeastern zones), not
invented from scratch and not claimed to be pixel-identical to the
published cluster result.

## "Season" as a fitting dimension

Every real store built so far (`data/baseline_2020_jjas.zarr`,
`data/lagged_ensemble_inputs_2020_jjas.zarr`) is JJAS-only. **"Per lead ×
region × season" collapses to "per lead × region" in practice right now**
— season always takes the single value "JJAS 2020." The weight-fitting
function (step 3) should carry a season parameter/dimension for future
extensibility (e.g. once a non-JJAS window or multiple monsoon years'
paired data exists), but should not be built to look more general than the
data currently supports, and this doc states that plainly rather than
silently implying multi-season fitting already works.

## Summary of decisions for steps 2-3

| Decision | Choice | Basis |
|---|---|---|
| CV strategy | `seasonal_block_split`, all sources | User-confirmed tradeoff; keeps methodology consistent given GraphCast's permanent one-season ceiling |
| Regional scheme | Sreekala & Babu 6-zone (WC/SI/WI/CI/NE1/NE2) | Single sound answer — Neal et al.'s scheme is temporal, not spatial, and doesn't fit Phase 3's need |
| Zone boundaries | Documented lat/lon approximation of the 6 named zones | Exact k-means cluster result not reproducible here |
| Season dimension | Present in the API, currently always "JJAS 2020" | Matches what the real store actually contains |
