# WEAVR

Hybrid AI–NWP multi-model forecast blending (SIH PS 26081).

See [docs/phase-plan.md](docs/phase-plan.md) for the build plan, tracked as
GitHub issues per phase. See [docs/data-sources.md](docs/data-sources.md),
[docs/grid-and-time-convention.md](docs/grid-and-time-convention.md), and
[docs/baseline-store.md](docs/baseline-store.md) for the Phase 0 data
pipeline: `scripts/build_baseline_store.py` builds the baseline JJAS
forecast + IMD observation store. See
[docs/heppi-reference-dataset.md](docs/heppi-reference-dataset.md) for a
third-party NCMRWF-ensemble + IMD reference dataset (not redistributed here)
useful for Phase 1 bias-correction work. See
[docs/phase-1-data-requirements.md](docs/phase-1-data-requirements.md) for
what each Phase 1 verification metric needs and what was extended:
`scripts/build_seeps_climatology.py` builds the multi-year IMD-only
climatology archive SEEPS/ACC-for-precipitation need.
[src/weavr/verify.py](src/weavr/verify.py) implements the Phase 1
verification protocol itself: RMSE/bias/ACC, CRPS/Brier, SEEPS, FSS, and
POD/FAR/CSI/ETS at IMD's own rain thresholds.
[src/weavr/splits.py](src/weavr/splits.py) is the only sanctioned way to
split time series data in this codebase: `leave_one_year_out` (needs 2+
years, not yet true for the current single-season baseline store) and
`seasonal_block_split` (usable now) — never a random split. See
[docs/tier0-baseline-results.md](docs/tier0-baseline-results.md) for Phase
1's exit criterion: `scripts/run_tier0_baseline.py` scores the equal-weight
mean forecast against IMD gauges — the benchmark every later phase must beat.
See [docs/baseline-store.md](docs/baseline-store.md)'s Phase 2 addition for
`scripts/build_lagged_ensemble_store.py`, which pulls the extra GraphCast/
Pangu init times a ±4-starts/12h-spaced lagged ensemble needs, turning these
deterministic AI models into probabilistic ensembles comparable to NWP
ensemble members.
[src/weavr/ensemble.py](src/weavr/ensemble.py) is the pure-function
counterpart: `build_lagged_ensemble` assembles one nominal forecast's
lagged ensemble from an already-open source dataset, producing the
`member`-dimensioned shape `verify.py`'s `crps`/`brier_score` expect. See
[docs/phase2-ensemble-baseline-results.md](docs/phase2-ensemble-baseline-results.md)
for Phase 2's exit criterion: `scripts/run_phase2_ensemble_baseline.py`
scores GraphCast's lagged precipitation ensemble with CRPS/Brier against
IMD gauges — the probabilistic counterpart to Phase 1's Tier 0 baseline —
and documents its dispersion characteristics (`weavr.verify.spread_skill_ratio`,
`calibrated_spread_skill_ratio`): the lagged ensemble is measured to be
strongly under-dispersive at every lead, a finding later phases' weighting
decisions need to account for. See
[docs/phase3-cv-and-regional-scheme.md](docs/phase3-cv-and-regional-scheme.md)
for Phase 3's scoping decisions, and
[src/weavr/regions.py](src/weavr/regions.py) for `assign_regions`, which
pools weavr's common India grid into Sreekala & Babu's six homogeneous
monsoon rainfall zones (WC/SI/WI/CI/NE1/NE2) for per-region skill weighting.
[src/weavr/weighting.py](src/weavr/weighting.py) fits those per-region
weights (`fit_region_weights`): unconstrained regression per Wanders &
Wood (2016), negative weights clipped to zero (Wang et al. 2025) and
renormalized to sum to 1, with a documented equal-weight fallback for
regions with too few train samples or a rank-deficient fit. See
[docs/tier1-regional-weights-results.md](docs/tier1-regional-weights-results.md)
for Phase 3's exit criterion: `scripts/run_tier1_regional_baseline.py`
fits and scores the real regional blend against IMD gauges and compares
it directly, region by region and lead by lead, against Tier 0's
equal-weight mean — the regional blend wins domain-wide at every lead and
in 22 of 30 region×lead cells, not universally, reported plainly either way.
See [docs/phase4-data-and-combiner-scope.md](docs/phase4-data-and-combiner-scope.md)
for Phase 4's scoping decisions: `seasonal_block_split` substitutes for an
unreachable rolling training window, GraphCast's lagged pseudo-ensemble
plus a newly-decided real IFS 50-member ensemble pull back the EMOS-CSG/BMA
combiners (HEPPI is a validation reference only, not training data), and
the real per-bin sample counts found the extreme-rain bin (204.5mm+) is
never fittable at any lead. See `docs/baseline-store.md`'s Phase 4 addition
for `scripts/build_ifs_ensemble_store.py`, which pulls that real IFS
50-member ensemble into `data/ifs_ens_2020_jjas.zarr` — ~55 minutes of real
GCS transfer time, resumable across the idle-sleep-induced connection
stalls that interrupted the live run.
[src/weavr/rain_bins.py](src/weavr/rain_bins.py) implements the first
Phase 4 checkbox itself: `classify_rain_bin` labels each forecast value
into IMD's rain-intensity bins (dry/light/heavy/very_heavy/extremely_heavy
— the upper three are IMD's own published category names, at its own
64.5/115.6/204.5mm boundaries), distinct from `verify.py`'s obs-vs-forecast
contingency scoring at the same thresholds, with missing (NaN) forecast
values explicitly excluded rather than silently landing in the last bin
(`numpy.digitize`'s own default behavior).
[src/weavr/emos.py](src/weavr/emos.py) implements issue #6's named default
combiner, EMOS-CSG: `fit_emos_csg` fits a censored-shifted-gamma
distribution (Scheuerer & Hamill 2015; Baran & Nemoda 2016) per rain-
intensity bin, per ensemble source, minimizing CRPS via a from-scratch,
numerically-verified closed form (`csgd_crps`) rather than the paper's own
published equation, which did not reproduce the correct value when checked
against brute-force numerical integration. Falls back to a point-mass-at-
zero prediction, flagged via `CensoredShiftedGammaResult.is_fallback`, for
a bin with too few train days or a degenerate (all-dry) climatology.
[src/weavr/bma.py](src/weavr/bma.py) implements issue #6's named
comparison combiner, hierarchical BMA: `fit_hierarchical_bma` fits a
mixture, per rain-intensity bin and per `weavr.regions` zone, of each
source's own point-mass-plus-power-transformed-gamma predictive
distribution (Sloughter et al. 2007 -- not Raftery et al. 2005's Gaussian
form), with mixture weights fit by EM. Ensemble sources (GraphCast, IFS)
regress their component's variance on real per-cell ensemble spread;
HRES, with no ensemble, uses Sloughter et al.'s own fixed-variance
"kernel dressing" route instead. `score_bma` estimates CRPS via Monte
Carlo sampling (no closed form exists for a general mixture), reusing
`weavr.verify.crps`'s ensemble machinery -- the same predict/score
interface shape as `weavr.emos.score_csgd`, so Phase 4's final comparison
can call both combiners identically.
See [docs/tier2-hierarchical-baseline-results.md](docs/tier2-hierarchical-baseline-results.md)
for Phase 4's exit criterion:
`scripts/run_tier2_hierarchical_baseline.py` fits and scores both real
combiners against the real stores on one shared train/test split, and
reports an honest, mixed comparison -- EMOS-CSG and BMA really do trade
off which they win (EMOS-CSG dominates the "heavy" rain-intensity bin at
most leads, BMA wins domain-wide at 2 of 5 leads), and Tier 2 does not
beat Tier 0/Tier 1 everywhere either. Running it against the real data
also caught and fixed a real numerical bug in `weavr.emos.csgd_crps`
(tiny negative-precipitation numerical-noise artifacts in GraphCast's own
real forecast data could produce a physically nonsensical CRPS), documented
in that results doc.
See [docs/single-source-and-independence-results.md](docs/single-source-and-independence-results.md)
for the baseline every tier should have been measured against from the
start, and for why the tiers look the way they do.
`scripts/run_single_source_baseline.py` scores each raw source **alone** on
exactly the tier scripts' own split (reusing their alignment, scoring and
split code, not a second path), and finds that **raw GraphCast is not beaten
by Tier 0 or Tier 1 at any lead from 24 h to 96 h** -- the only blends that
beat it, Tier 2's EMOS runs, are themselves single-source calibrations, so
what is helping is calibration rather than combination. It also records
`best_single_member_on_train`, the defensible baseline picked on train
rather than on test, whose ranking disagrees with the test ranking at 2 of
5 leads on these 3-4 day test sets. `scripts/run_independence_diagnostic.py`
and `weavr.independence` explain the mechanism: the three sources' errors
correlate at 0.62-0.90, giving an effective number of models of **about 1.1
out of 3**, so averaging them cancels almost nothing -- and HRES's zero
Tier 1 weight in 25 of 30 cells is its 0.77-0.93 correlation with IFS-ENS
(both are ECMWF's own system), not evidence that HRES is useless. The
opposite, in fact: HRES is the only source in the store that detects **any**
115.6 mm event, while GraphCast, the RMSE winner, detects none at any lead.
That is the research brief's section 1.3 caveat -- RMSE rewards smooth,
See [docs/preregistration.md](docs/preregistration.md) for the eleven
headline claims WEAVR hopes to make (H1-H11), each with its exact data,
split, metric, comparison and pass rule -- **registered before the v2
evidence base produced a single number**, so the git history of that file
proves the criteria came first. Phase 5's Tier 3 go/no-go was the precedent
(stated before the run, and published as a clean NO-GO); this makes it the
rule for every claim that reaches a slide. The verdicts land in
`results/preregistration_verdicts.csv`, only PASS claims may appear as
findings, and a failed claim is reported as prominently as a passed one. Two
decisions are locked in there: **H1 is judged on RMSE and CRPS separately**,
never merged into one answer, so the metric WEAVR currently loses on stays
on the record; and every "best of" comparator is chosen on **training** data
with the headline configuration declared before verdicts are computed, since
reporting five tiers and claiming the best is five chances at a
one-in-twenty error rather than one.

[scripts/recover_heppi_dates.py](scripts/recover_heppi_dates.py) resolves the
missing calendar in the third-party HEPPI reference dataset
([docs/heppi-reference-dataset.md](docs/heppi-reference-dataset.md)): by matching
HEPPI's IMD rainfall observation fields against the 15-year (2006-2020) IMD JJAS
climatology (`data/imd_seeps_climatology_jjas.zarr`), **all 242 of 242 JJAS days
across 2018 and 2019 match their hypothesized date as the unique rank-1 best match**
(documented in [docs/heppi-date-map.csv](docs/heppi-date-map.csv)). Shifting by
+/-1 day drops matches to 0, ruling out calendar ambiguity. Forecast validity is
confirmed: NCMRWF NEPS-G ensemble-mean correlation with IMD observations peaks
at offset 0 (r = 0.5946), confirming index i validates against observation i.
`weavr.data.heppi_reference` now attaches a real `time` coordinate and drops
unconfirmed non-monsoon days by default, unlocking India's operational 23-member
NEPS-G ensemble for multi-model blending.

See [docs/scorecard-and-significance.md](docs/scorecard-and-significance.md)
for the machinery that turns those numbers into claims with error bars, and
for what it says about the evidence base that existed before it.
`weavr.significance` adds a moving-block bootstrap, paired-difference
confidence intervals and a Diebold-Mariano test; `weavr.verify` gains SEDI,
threshold-weighted CRPS, PIT, reliability tables and skill scores;
`weavr.climatology` builds the climatological reference every skill score is
measured against, **excluding the test year**; and every scoring script now
also writes per-day domain-wide scores to `results/per_day/` so differences
can be bootstrapped over days (existing aggregated CSVs are byte-for-byte
unchanged -- this is purely additive). `scripts/run_scorecard.py` turns
those into `results/scorecard.csv` and
`results/preregistration_verdicts.csv`. The finding: **not one of the 1,691
comparisons on today's weekly store has a computable confidence interval.**
With 3-4 test days per lead and a 7-day block there is exactly one possible
resample, so no interval exists -- and a first run that ignored this
reported 57% of comparisons as "significant", which is the precise failure
this step exists to prevent. Every published WEAVR comparison to date rests
on a sample too small to put an interval around; step 07's daily,
two-season data is what fixes that. Scoring the 434-member climatological
reference also hit a real limit: `properscoring`'s CRPS is O(m^2) in memory
and needs ~83 GB there, so `weavr.verify.crps_large_ensemble` computes the
identical number from the sorted-ensemble identity in O(m) memory, checked
against `crps` to ~1e-15.
See [docs/phase5-regime-covariate-scope.md](docs/phase5-regime-covariate-scope.md)
for Phase 5's scoping: which regime covariates issue #7 names are real and
actually obtainable for this project's 2020 JJAS-only data -- monsoon
active/break (Rajeevan, Gadgil & Bhate 2010, derivable from `imd_observed`
directly) and MJO phase (Kiladis et al. 2014's OMI) and monsoon-depression
presence (a real ERA5-derived low-pressure-system track catalogue) are in
scope; Neal et al.'s 30-weather-pattern classification (covers only
1979-2016, not this project's 2020 season) and western-disturbance
presence (real, but ~50x rarer in JJAS than winter -- near-zero signal in
a JJAS-only season) are not.
[src/weavr/regimes.py](src/weavr/regimes.py) implements the first
covariate directly (`classify_monsoon_active_break`, spot-checked against
real, independently reported 2020 monsoon activity); the other two are
built by [scripts/fetch_omi_mjo_index.py](scripts/fetch_omi_mjo_index.py)
and
[scripts/build_monsoon_depression_index.py](scripts/build_monsoon_depression_index.py)
-- real per-category counts for all three (14 active / 3 break days; an
8-phase MJO spread; 14 real depression-or-stronger days) are in
`docs/phase5-regime-covariate-scope.md`'s own step 2 update.
See [docs/phase5-regime-conditioned-results.md](docs/phase5-regime-conditioned-results.md)
for step 3's real assessment of whether a regime covariate plausibly
explains any of Phase 4's own error pattern (checked directly against the
real Tier 2 train/test data, not assumed): the held-out test window used
to measure Phase 4's heavy-bin EMOS-vs-BMA split turns out to have almost
no regime diversity at any lead, ruling out a regime explanation for that
specific alternation, while training data shows only a weak, noisy
correlation between monsoon-active days and elevated heavy-rain cell
counts. [src/weavr/regime_weighting.py](src/weavr/regime_weighting.py)
implements the simplest defensible regime-stratified fit anyway (reusing
`weavr.weighting.fit_region_weights`'s own OLS/fallback machinery,
conditioned on monsoon active/break instead of spatial region); given this
weak evidence and a training set with only 2 "active" days per lead, the
user decided via `AskUserQuestion` not to build the optional heavier
GBM/ViT-style gating model issue #7 names.
[scripts/run_tier3_regime_conditioned_baseline.py](scripts/run_tier3_regime_conditioned_baseline.py)
applies issue #7's own go/no-go criterion (decided before running it: beat
Phase 4's real EMOS-CSG/BMA on domain-wide CRPS at >= 3 of 5 leads),
recomputing Phase 4's own combiners on the identical split. **Result: NO
GO** -- the regime-conditioned model wins 0 of 5 leads domain-wide, and 0
of 15 (lead, bin) cells, reported plainly per
`docs/phase5-regime-conditioned-results.md`'s own step 4 (which also names
why: weak underlying regime signal, plus a structural CRPS disadvantage
from being a deterministic point-forecast blend rather than a fitted
predictive distribution like EMOS-CSG/BMA). Phase 5's regime-conditioned
model is not adopted.

[src/weavr/renormalize.py](src/weavr/renormalize.py) fills issue #8's real
operational gap: every combiner above fits/blends assuming all of its
sources are present, but a real GraphCast/IFS/HRES pull can fail or arrive
late. `renormalize_weights` redistributes an already-fit flat `{source:
weight}` dict (`weavr.weighting`/`weavr.regime_weighting`'s own shape)
proportionally over whichever sources are actually present for a given day
-- no re-fitting -- falling back to passing a single surviving source
through with weight 1.0 (flagged `is_fallback`/`reason`, mirroring every
other combiner's own convention) when too few sources survive.
`weavr.bma.renormalize_bma_for_present_sources` adapts the same idea to
BMA's mixture shape (dropping a missing source's fitted predictive
component, not just its weight, before renormalizing); `weavr.emos` needs
no adaptation at all, checked rather than assumed -- it fits one source at
a time and has no cross-source weight for a missing source to redistribute
onto. Detecting "missing" itself is two distinct, real shapes in this
project's own stores (`renormalize.py`'s own docstring): a source that
never got pulled at all is simply absent as a dict key (`build_baseline_
store.py`/`build_lagged_ensemble_store.py`/`build_ifs_ensemble_store.py`
each write one independent, independently-failable Zarr group per source),
while a source present overall but missing one particular day is NaN-filled
by the existing `xr.align(..., join="outer")` step
(`run_tier1_regional_baseline.py`/`run_tier2_hierarchical_baseline.py`
already use this to detect a missing day for `obs`; `missing_for_sample`
applies the identical check to a forecast source).

See [docs/phase6-operational-scope.md](docs/phase6-operational-scope.md) for
Phase 6's two operational decisions, checked against what this project can
actually reach rather than assumed: NWP (IFS/HRES) already has a working
published route via `src/weavr/data/ecmwf_open_data.py`, settled with no
tradeoff; the AI-model side does not — WeatherBench 2's GraphCast/Pangu
archives are fixed historical eval windows, not a live feed, so **AIFS
(ECMWF's own AI model, reachable via the same client with
`model="aifs-single"`) is substituted for GraphCast/Pangu** as the daily
pipeline's AI-model input, a real, checked, user-confirmed divergence from
the models Phases 2-5 were actually evaluated against. The daily job itself
runs on a GitHub Actions `schedule:` trigger (reusing this repo's existing
CI infra rather than a new always-on machine), publishing small committed
result files the same way every `run_tierN_*.py` script already does,
rather than extending the existing one-shot, full-season Zarr stores.

[src/weavr/drift.py](src/weavr/drift.py) implements issue #8's last
checkbox: rolling verification and drift detection, reusing
`weavr.verify`'s existing metrics on a trailing window rather than a new
scoring path. Checked, not assumed, before writing any detection logic:
no model-version identifier exists anywhere this project can reach — not
in the three historical store-building scripts' own manifests (fixed, see
`docs/baseline-store.md`'s Phase 6 addition: they now record
`source_archive_path`), and not in the live `ecmwf_open_data.py` fetch
path either (live-tested: a real fetched IFS dataset's own GRIB metadata
carries no model-cycle field at all). So the practical "model upgraded"
proxy is a real statistical drift check instead — `TRAILING_WINDOW_SAMPLES
= 14` reuses Phase 5's own already-established real monsoon-active-spell
length (`docs/phase5-regime-covariate-scope.md`'s real 14-day 2020 JJAS
spell), and `DRIFT_THRESHOLD_STD_MULTIPLIER = 2.0` is grounded in this
project's own real historical per-sample RMSE spread (std/mean ~28-33%
consistently across leads, measured directly against
`data/baseline_2020_jjas.zarr`). [scripts/run_daily_verification.py](scripts/run_daily_verification.py)
demonstrates the mechanism against the only real accumulated history that
exists today (the baseline store's 18 real weekly JJAS-2020 samples,
standing in for daily ones) — and states plainly, checked directly with a
synthetic 500mm late-sample error injection, that the demo's own baseline
overlaps its trailing window at today's real sample count (`n <= 14`), so
drift genuinely cannot fire yet no matter how large a real shift is; this
resolves once step 4's daily pipeline has accumulated more real days than
`TRAILING_WINDOW_SAMPLES`, with no code change needed here.

[scripts/run_daily_pipeline.py](scripts/run_daily_pipeline.py) is issue #8's
actual operational deliverable, wiring steps 1-3 into one real job: fetch
each source per `docs/phase6-operational-scope.md`'s decision (AIFS/IFS/
HRES), renormalize over whatever sources actually arrived
(`weavr.renormalize`), blend, and retrospectively verify/drift-check once a
forecast's valid day's IMD obs becomes available (`weavr.drift`). **Which
combiner, checked and routed to the user rather than assumed**: not
EMOS-CSG/BMA — their fitted parameters are per-source statistical fits
(CSGD regression coefficients, spread-variance regressions) tied to
GraphCast/HRES/`ifs_ens_mean`'s own real historical error characteristics,
which the live AIFS/deterministic-IFS substitutions don't match; reusing
them would apply one model's bias-correction to a different model's output.
**Uses Phase 3's `weavr.weighting.fit_region_weights` instead**
(`results/tier1_regional_weights.csv`, already fit and committed) — its
flat linear OLS weights are a much lighter, explicitly flagged
approximation under the same substitution. Two more real substitutions
stated plainly: AIFS (`model="aifs-single"`) fills the `graphcast` weight
slot, and a single deterministic IFS pull fills the `ifs_ens_mean` slot (the
real ensemble mean it was fit against isn't published near-real-time).
**Ground truth checked live, not assumed**: IMD's real gridded product has
no valid current-year data through this project's client (live-tested,
`imdlib` fails to parse a current-year request) — `fetch_today_obs` returns
`None` as an expected case, and verification is necessarily retrospective
(a forecast issued `lead_hours` ago validates today), scored against a
small per-day `.npy` archive (`results/daily_pipeline_forecasts/`) kept
specifically to make that retrospective scoring possible. A known,
unfixed operational risk is flagged in the module's own docstring: ECMWF's
client retries a rate-limited response up to 500 times internally, which
can make a real run hang far longer than expected.

Issue #8's checkboxes, against what's actually been built: **renormalize
weights for a missing/late source** — done (`weavr.renormalize`, step 2).
**Run-ourselves vs. consume-published-forecasts** — decided
(`docs/phase6-operational-scope.md`, step 1). **Storage/pipeline-runner** —
decided (step 1: GitHub Actions `schedule:` trigger, small committed
artifacts); the `schedule:` workflow entry itself is not added in this
step — wiring the trigger is a one-line addition once whoever runs this
operationally is ready for it to fetch real data on a real cadence, left
for a follow-up rather than turned on silently here. **Daily-refreshed
rolling verification and drift detection** — done
(`weavr.drift`/`scripts/run_daily_verification.py`, step 3, and this
script's own retrospective scoring).

## Phase 7: dashboard scope

Before building issue #9's 4 dashboard views, `docs/phase7-dashboard-scope.md`
checks each view's real data source directly: the weight-map and skill-trend
views are fully supported by already-committed CSVs
(`results/tier1_regional_weights.csv`, the 4 tiers' own baseline CSVs), while
the blended-map and extreme-probability views need real `(latitude,
longitude)` arrays that only exist in a local, gitignored Zarr store today —
the doc decides those two views will read a small, committed example export
(`dashboard/data/*.npz`, built in later steps) rather than requiring that
store locally. It also cites IMD's real published rainfall-warning colour
code (green/yellow/orange/red at 64.5/115.6/204.5mm — which turn out to
already match this project's own `weavr.verify.IMD_RAIN_THRESHOLDS_MM`
exactly) for both spatial views' colour scales, and locks the dashboard
stack as Streamlit + Plotly (`pip install -e ".[dashboard]"`).

Two of the 4 views are built:
`dashboard/views/weight_map.py` renders
`results/tier1_regional_weights.csv`'s real per-region, per-lead source
weights (stating plainly that this project's real scheme is Phase 3's OLS
fit, not a softmax/GBM gate, with fallback regions shown hatched), and
`dashboard/views/skill_trends.py` renders
real per-lead RMSE/CRPS across all 4 tiers' own results CSVs, checked
column-by-column (`dashboard/data_loading.py`'s own docstring) rather than
assumed identical -- including the real, checked finding that tier3's
`emos_graphcast`/`emos_ifs_ens`/`bma` columns are bit-for-bit duplicates of
tier2's own numbers, so only tier3's own new `regime_conditioned` method is
plotted from that file. Both views were run locally with
`streamlit run` and confirmed to render without error before this was
written down.

A third view is built:
`dashboard/views/blended_map.py` renders
Tier 1's real regional blend over India for a selectable lead time, using
`scripts/run_tier1_regional_baseline.py`'s own `build_region_weight_grid`
/ `blend_with_region_weights` (the same functions
`scripts/run_daily_pipeline.py` already reuses) and IMD's real
rain-intensity colour breakpoints (`dashboard/colors.py`). The map's data
comes from `dashboard/data/example_blend_grid.npz`, a small (~700KB),
committed, one-time export built by
[`scripts/export_dashboard_example_grids.py`](scripts/export_dashboard_example_grids.py)
from the real local `data/baseline_2020_jjas.zarr` store, reusing
`results/tier1_regional_weights.csv`'s already-fitted weights rather than
re-fitting them -- per `docs/phase7-dashboard-scope.md`'s decision, so this
view renders for anyone who opens the dashboard, not only someone with the
local Zarr store built. Run locally with `streamlit run` and confirmed to
actually render (via a browser, not just an HTTP check) for every one of
the 5 lead times before this was written down; this also caught and fixed
a real bug -- `streamlit run` inserts the script's own directory into
`sys.path`, not its parent, so `from dashboard...` imports fail unless the
repo root is added to `sys.path` first (step 5's assembled app needs the
same fix).

All 4 of issue #9's views are now built. The 4th,
`dashboard/views/extreme_probability.py`,
renders P(rain > 204.5mm) -- IMD's own real "extremely heavy rain"
boundary -- from EMOS-CSG's real fitted `ifs_ens` combiner (the real
50-member IFS ensemble), a choice routed to the user via `AskUserQuestion`:
EMOS-CSG's censored-shifted-gamma has a real closed-form survival function
(`weavr.emos.exceedance_probability_csgd`, verified in `tests/test_emos.py`
against a Monte Carlo simulation of the real censored-shifted-gamma
sampling process), while BMA's mixture has no closed form
(`weavr.bma`'s own docstring) and would need a slower, approximate
per-gridpoint Monte Carlo estimate. This diverges from the blended map's
own Tier 1 combiner (stated plainly in the view), since Tier 1's
deterministic blend has no predictive distribution to compute an
exceedance probability from at all. Cells whose own forecast fell in a
rain-intensity bin EMOS-CSG could not fit for real (checked directly,
again, while building this view: the `extremely_heavy` bin has 0 real
train days at every one of the 5 leads, confirming
`docs/phase4-data-and-combiner-scope.md`'s own finding) render with a
visible grey overlay, not a confident-looking probability number. The
grid is `dashboard/data/example_probability_grid.npz`, built by the same
`scripts/export_dashboard_example_grids.py` (extended in this step),
reusing `run_tier2_hierarchical_baseline.py`'s own real data-loading
functions rather than reimplementing EMOS-CSG fitting a second time. Run
locally with `streamlit run` and confirmed to actually render for every
lead, including one (96h) with real fallback cells rendering the grey
overlay correctly.

### Original Streamlit prototype (retired)

Issue #9's dashboard was originally built as a Streamlit app
(`dashboard/app.py`, `dashboard/views/*.py`) -- all 4 real views verified
by driving a real browser through it, no HTTP-only health check. Once the
HTML/CSS/vanilla-JS frontend migration (below) reached full, real,
side-by-side parity with it, the Streamlit app was **retired** (removed,
not just left unused) per a real, routed decision -- see "Streamlit's
status" below. It is no longer present in this repo; this section is kept
as the historical record of what issue #9's checkboxes were actually
verified against before retirement.

Issue #9's checkboxes, against what was actually built and verified:
**Blended map view** -- done (Tier 1's real regional blend, IMD's real
colour breakpoints). **Weight-map per model x lead time** -- done (this
project's real weighting scheme is Phase 3's per-region OLS fit, stated
plainly in the view -- not the softmax/GBM gate this checkbox's own
wording references, since that was never this project's real method).
**Skill trend charts** -- done (the real, checked, mixed cross-tier
outcome, not a smoothed trend). **Extreme-probability maps in IMD colour
codes** -- done (EMOS-CSG's real fitted `ifs_ens` combiner, IMD's real
colour identities, fallback cells shown distinctly). **Build with
Streamlit/Plotly/Leaflet** -- done at the time, as Streamlit + Plotly
(`docs/phase7-dashboard-scope.md`'s own decision: Leaflet's Python
binding is for tile-served imagery, which this project's fixed-domain
grid data doesn't need); superseded by the HTML/CSS/vanilla-JS frontend
below once that reached parity.

A real, stated gap that still applies to the current frontend, not
silently glossed over: the blended-map and extreme-probability views
render from small, committed *example* grids (`dashboard/data/*.npz`),
not a live re-blend of whatever is in a local `data/*.zarr` store -- per
`docs/phase7-dashboard-scope.md`'s own data-source decision (the real
spatial data these two views need only exists in a local, gitignored
Zarr store, not guaranteed to exist on whoever opens this dashboard).
Anyone with that store built locally can refresh both example grids for
real with `python scripts/export_dashboard_example_grids.py`.

## Frontend migration

A separate effort (planned and driven from outside this repo, under
`workspace/frontendplan.md` and `workspace/frontend-prompts/`) migrated
the Streamlit dashboard above to a plain HTML/CSS/vanilla-JS frontend
behind a small JSON API, so it can be hosted without a Python process
rendering every page. `docs/frontend-migration-scope.md` locks the real
decisions this migration needed: `dashboard/data_loading.py` and
`dashboard/colors.py` are confirmed (by grepping their own import lines)
to have zero Streamlit/Plotly coupling, so the new API reuses both
unchanged; charting is hand-rolled `<canvas>`/SVG (no JS charting
dependency, chosen over Plotly.js); the API layer is FastAPI + uvicorn (a
`dashboard-api` optional-dependency group, matching this project's
existing `mypy`-gated typing convention); and grid responses stay plain
nested JSON arrays (both committed example grids are a fixed 129x135, well
within normal response sizes).

### Running the new frontend

```bash
pip install -e ".[dashboard-api]"
uvicorn dashboard.api:app --reload
```

Then open `http://127.0.0.1:8000/` — [`dashboard/api.py`](dashboard/api.py)
serves both the JSON API (`/api/*`) and the static
[`dashboard-web/`](dashboard-web/) frontend from this one process, so
there's nothing else to run and no CORS configuration needed.

### Parity with the Streamlit app

All 4 real views ([`dashboard-web/js/charts/`](dashboard-web/js/charts/):
`weightMap.js`, `skillTrends.js`, `blendedMap.js`, `extremeProbability.js`)
are built and were checked side by side against the live Streamlit app
(`dashboard/app.py`) across all 5 real lead times, both skill-trends
metrics, and every real caption/subheader/chart-title string -- not just
that the numbers matched. This real side-by-side check caught two genuine
text bugs that a read-through alone would have missed: the weight-map
view's caption had drifted to say "hover **a bar** for the real reason"
(the extra words aren't in `dashboard/views/weight_map.py`'s real text),
and the extreme-probability view's caption kept a literal `` `ifs_ens` ``
backtick pair as plain-text characters instead of the plain word
`ifs_ens` Streamlit's own markdown renderer actually shows. Both are
fixed. Each view's `st.subheader()` title and Plotly chart-title text are
now also reproduced in the new frontend, which an earlier step had missed
(captions and warnings were reproduced, but not these two).

One real, accepted difference remains, not a defect: Streamlit's Plotly
legend for the weight map combines colour and hatch pattern into one
`source, fit_status` legend entry per bar; the new frontend instead shows
plain colour-swatch legend entries plus a `*` marker on hatched regions'
axis labels and a separate warning banner -- different presentation, same
real information (which regions are OLS-fitted vs. fallback), not
reproduced pixel-for-pixel because a hand-rolled SVG legend has no
equivalent to Plotly's combined dual-encoding legend without meaningfully
more code for a cosmetic difference.

### Streamlit's status

Once the parity check above confirmed the new frontend was a genuine
replacement, the Streamlit-vs-retire tradeoff was routed to the user via
`AskUserQuestion` rather than assumed (frontendplan.md §6's deliberately
deferred 4th tradeoff): a second app to keep in sync indefinitely vs.
losing a fast, pure-Python reference for prototyping future view changes.
**The decision: retire it.** `dashboard/app.py` and `dashboard/views/*.py`
are removed from this repo -- `dashboard/data_loading.py` and
`dashboard/colors.py` stay, unchanged, as the single source of truth the
API (`dashboard/api.py`) still imports directly. The `dashboard` optional
dependency group (`streamlit`, `plotly`) is removed from `pyproject.toml`;
`dashboard-api` (`fastapi`, `uvicorn`) is the only one needed to run the
dashboard now.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest --cov=weavr
```

CI (`.github/workflows/ci.yml`) runs lint, type-check and tests (3.11/3.12)
only when it matters:

- **Pull requests into `main`**, except drafts and PRs that touch only
  Markdown, `docs/`, `dashboard-web/` or `.gitignore`, since no CI job reads
  those. Changes to `results/` or `dashboard/data/` still trigger it,
  because tests read those committed files.
- **Release tags (`v*`)** and **manual runs** (Actions → CI → Run
  workflow). These also build the package and upload it as an artifact.

Merging a PR does not re-run CI on `main`: the squash-merged code is exactly
what the PR's checks already passed. Add `[skip ci]` to a commit message to
skip a run on a PR by hand.
## Improvement plan

The step-by-step plan for strengthening the evidence, the tail forecasts and
the demo lives in [`Improvements/`](Improvements/):
[`WEAVR-SIH-improvement-plan.md`](Improvements/WEAVR-SIH-improvement-plan.md)
(what and why), [`EXECUTION-PLAN.md`](Improvements/EXECUTION-PLAN.md) (order),
[`prompts/`](Improvements/prompts/) (one prompt per remaining step) and
[`prompts/completed/`](Improvements/prompts/completed/) (steps 01–05, with
`DONE.md` summarising what they delivered).

## OpenStreetMap basemap (in progress)

An optional "Show geography" layer for the two map views is being added,
built from OpenStreetMap data served from a local file so the demo works
offline. The scope, decisions and measured tile sizes are in
[`docs/basemap-scope.md`](docs/basemap-scope.md).

The dashboard server now also serves the local basemap file at
`/basemap/india.pmtiles` (with a fallback status at `/api/basemap/status`),
and the map libraries are vendored under `dashboard-web/vendor/`. Nothing in
the dashboard uses them yet; the "Show geography" toggle arrives in a later
step.
