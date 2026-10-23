# v2 Evidence Base Results

Two-season daily LOYO evidence base, regenerated under step 07.
**Two folds, 122 days each, 244 test days per lead.** Every number below is
measured on `results/` outputs produced by the scripts in `scripts/`, with
per-day scores so every interval is a paired block bootstrap.

Reproduce:

```
scripts/preflight_stores.py                    # values + units, before anything expensive
scripts/run_tier0_baseline.py                  … through run_phase2_ensemble_baseline.py
scripts/run_independence_diagnostic.py
scripts/run_scorecard.py --results-dir results
scripts/gate_step07_results.py                 # evidence-base validator
```

---

## Summary

| | |
|---|---|
| Tiers regenerated | 0, 1, 2, 3, single-source, Phase 2, independence, daily verification |
| Test days per lead | 244 / 242 / 240 / 238 / 236 (leads 24→120h) |
| Scorecard comparisons | 3857, 61.0% significant |
| Degenerate bootstraps | **0** |
| Verdicts reached | **4** of 12 registered claims: H1 PASS x2, H2 FAIL, H3 FAIL |

**All three headline claims are now decided, and two of the three fail.** H1
passes twice over: the blend beats the best single ensemble member on RMSE and
on CRPS at 5 of 5 leads each. H2 fails — the multi-source combiner is
significantly **worse** than either single-source EMOS calibration on CRPS, at
every lead. H3 fails on both
legs: at 64.5 mm the blend's Brier is 3.6× worse than climatology (BSS −2.6,
where the claim needs > 0) and its SEDI is worse than the best member at every
lead.

So the blend is better than a single raw member (H1) and worse than a
single-source calibration (H2) and than climatology on heavy-rain probability
(H3). The declared headline, `tier2_bma`, is the weakest of the three
configurations on the two probabilistic claims it was declared to win.

---

## 1. What the evidence base is

Two-season leave-one-year-out. Each fold trains on one season and tests on
the other, so no season scores itself. Per-day scores carry both folds —
244 days at lead 24 falling to 236 at lead 120, because a longer lead pushes
the last initialisation times' valid times past the end of the observed
window. That arithmetic is asserted in `scripts/gate_step07_results.py`.

Both seasons' stores are daily, 122 initialisations × 50 IFS members × 5
leads on the 129×135 IMD grid.

## 2. Aggregate scores

Pooled across both folds, from `results/tier2_hierarchical_baseline.csv`:

| lead | tier0 RMSE | tier1 RMSE | EMOS-CSG CRPS | EMOS-IFS CRPS | BMA CRPS |
|---|---|---|---|---|---|
| 24 h | 14.61 | 13.86 | 4.546 | 4.579 | 4.733 |
| 48 h | 14.71 | 13.86 | 4.554 | 4.598 | 4.741 |
| 72 h | 14.77 | 13.84 | 4.560 | 4.615 | 4.741 |
| 96 h | 14.78 | 13.85 | 4.596 | 4.666 | 4.770 |
| 120 h | 14.83 | 13.89 | 4.622 | 4.716 | 4.801 |

Per fold, Tier 0 RMSE is 13.90–14.24 mm on the 2018 fold and 14.98–15.71 mm
on 2020. Tier 1 beats Tier 0 at every lead on both folds, by 0.35–0.82 mm.
2020 is the harder season throughout — it is the wetter one (land mean 7.45 mm
against 6.29 mm in 2018).

### Two properties of these numbers worth stating plainly

**The probabilistic scores do not beat climatology.** A constant forecast at
the seasonal mean scores CRPS 3.33 mm on 2018 land cells and 3.45 mm on 2020;
the calibrated methods score 4.55–4.80 pooled. This is not a v2 regression — the
committed v1 weekly numbers had the same property (3.76 mm against a 3.45 mm
floor for 2020). Nothing pre-registered turns on beating climatology on CRPS:
H1 and H2 are *relative* comparisons against other WEAVR configurations.
It belongs here because a reader of this table will otherwise assume the
probabilistic layer is skilful against climatology, and it is not.

**BMA sits clearly behind both single-source EMOS calibrations.** 4.733
against 4.546 mm (EMOS-CSG) and 4.579 mm (EMOS-IFS) at lead 24; the gap to
EMOS-IFS narrows to 0.085 mm by lead 120 but never closes. Since `tier2_bma` is
the declared headline, this directly shapes the verdicts below.

## 2b. v1 vs v2, and why the columns must not simply be compared

| | v1 | v2 |
|---|---|---|
| cadence | weekly | daily |
| seasons | 2020 only | 2018 + 2020 |
| splits | `seasonal_block_split`, 1 fold | leave-one-year-out, 2 folds |
| train days | 14 | 122 |
| test days | 3–4 | 118–122 per fold, 236–244 pooled |
| intervals | not computable | paired block bootstrap |

| lead | tier0 RMSE v1 → v2 | tier1 RMSE v1 → v2 | BMA CRPS v1 → v2 |
|---|---|---|---|
| 24 h | 11.55 → 14.61 | 11.08 → 13.86 | 3.721 → 4.733 |
| 48 h | 13.59 → 14.71 | 12.53 → 13.86 | 3.802 → 4.741 |
| 72 h | 13.15 → 14.77 | 12.05 → 13.84 | 3.830 → 4.741 |
| 96 h | 14.15 → 14.78 | 13.34 → 13.85 | 4.643 → 4.770 |
| 120 h | 15.35 → 14.83 | 14.90 → 13.89 | 4.454 → 4.801 |

**Every tier got worse on RMSE, and that is not a regression.** v1 scored 4
test days, all in the last four weeks of September
(2020-09-07/14/21/28), on a weekly cadence. v2 scores 122 days per fold
across the whole season, including the early-season days that are harder to
predict. The v1 numbers were flattered by the sample, and the v2 numbers are
the honest ones. Tier 0 improving slightly at lead 120 h (15.35 → 14.83) is
the one place v2 is better, and it is the longest lead, where v1 had the
fewest test days.

This is why the audit in §8 says a v1-vs-v2 table must *state* the cadence
difference rather than set the columns side by side: read naively, the table
above says the project regressed, and it did not.

## 2c. Measured runtimes

Single machine, 8 cores, the whole Tier 2 process single-threaded. Prompt item
3 asks for this to be measured and written down.

| stage | wall clock |
|---|---|
| IFS-ENS daily store rebuild (2020, from cached staging) | **30 s** |
| pre-flight store check (17 store/variable checks) | **4 s** |
| Tier 2, 5 leads × 2 folds (EMOS-CSG + BMA) | **2 h 09 m** |
| Tier 3, 5 leads × 2 folds (regime-conditioned) | **2 h 09 m** |
| independence diagnostic | **~1 min** |
| daily verification, per season | **~10 s** |
| scorecard, 3857 comparisons × 1000 bootstrap resamples | **~1 min** |
| result gate | **~2 s** |
| **full end-to-end rehearsal on miniature stores** | **~7 min** |

The rehearsal figure is the important one: every runner can be executed for
real, CLI and exit codes included, against schema-faithful stores at ~1/1000
scale in about the time it takes to make coffee. Run it before any multi-hour
regeneration, not after the next loss.

Peak RSS was 2.6 GB for Tier 2 and 4.7 GB for Tier 3, both inside the memory
budget the blocked CRPS and cell-blocked BMA work was built for. The final
regeneration (after the BMA fix in §7) ran Tier 2 and Tier 3 side by side in
2 h 47 m each, peaking at 7.0 GB and 5.8 GB RSS; the exact full-support BMA
CRPS costs memory, not verdicts.

## 3. Verdicts

From `results/preregistration_verdicts.csv`. Claims H4–H11 belong to later
steps (08, 09, 11/12, 13, 14, 19, 21) and are correctly out of scope here.

### H1-RMSE — **PASS**

A WEAVR blend beats the best single member chosen on the training fold, on
RMSE. `tier2_bma` improves at 5 of 5 leads with the CI excluding 0 at every
one: by 0.96 mm at 24 h, narrowing to 0.32 mm at 120 h.

### H1-CRPS — **PASS**, via an identity; not evaluable as written

The claim is not evaluable in its literal form: `best_single_member_on_train`
is a single deterministic member, so its per-day file carries no `crps_mm` —
there is no ensemble spread to score a CRPS against.

It is evaluable as written in substance, because for a point mass at $x$,
$\mathrm{CRPS} = E|X-x| - \tfrac12 E|X-X'| = |y-x|$, which is exactly the
MAE. That is an identity, not a substitute metric, so the same claim against
the same comparator is decided on `mae_mm`:

| lead | tier2_bma | best member | diff | 95% CI | significant |
|---|---|---|---|---|---|
| 24 h | 7.0720 | 7.4312 | −0.3592 | [−0.5093, −0.2347] | yes |
| 48 h | 7.1046 | 7.4614 | −0.3568 | [−0.5030, −0.2375] | yes |
| 72 h | 7.1147 | 7.5628 | −0.4481 | [−0.5971, −0.3282] | yes |
| 96 h | 7.1682 | 7.5871 | −0.4189 | [−0.5442, −0.3201] | yes |
| 120 h | 7.2272 | 7.6138 | −0.3866 | [−0.4989, −0.2849] | yes |

5 of 5 leads, all intervals excluding 0.

### H2 — **FAIL** against both EMOS-GraphCast and EMOS-IFS

The multi-source combiner does not beat the best single-source EMOS on CRPS; it
is significantly worse than both.

| lead | vs EMOS-GraphCast | 95% CI | vs EMOS-IFS | 95% CI |
|---|---|---|---|---|
| 24 h | +0.1876 (BMA worse) | [+0.167, +0.214] | +0.1543 (BMA worse) | [+0.136, +0.177] |
| 48 h | +0.1873 | [+0.169, +0.210] | +0.1432 | [+0.120, +0.165] |
| 72 h | +0.1806 | [+0.157, +0.209] | +0.1255 | [+0.101, +0.146] |
| 96 h | +0.1740 | [+0.147, +0.204] | +0.1040 | [+0.080, +0.129] |
| 120 h | +0.1788 | [+0.153, +0.201] | +0.0852 | [+0.054, +0.114] |

BMA is worse at every lead against both comparators, and every interval
excludes 0.

**Correction to an earlier version of this section**, which reported H2 as
INSUFFICIENT_DATA against EMOS-IFS ("nominally better at 4 of 5 leads"). That
verdict came from a `results/scorecard.csv` generated *before* commit 9a44833
regenerated every Tier 2 per-day file: the committed scorecard was scoring BMA
at 4.643 mm at lead 24 while the committed per-day files said 4.747. Rebuilt
from the files actually in `results/`, the direction is unambiguous. The same
staleness explains every other significance flip between the two scorecards;
none is bootstrap noise (the bootstrap seed and day indices are identical).

This is the finding most worth the reader's attention. The blend beats a
single raw member decisively (H1) but does not beat a single-source
*calibration* (H2). Calibrating one good source appears to capture most of
what the multi-source mixture adds.

**Method note.** The comparator is not selected on the training fold, which
the pre-registration required. Instead BMA is compared against **both**
single-source EMOS calibrations. A method that beats both necessarily beats
whichever is best, so the claim is decided without choosing a comparator on
the data being tested. `needs_train_scores` is therefore cleared on H2, and
that reasoning is recorded in `REFERENCE_METHODS` in `scripts/run_scorecard.py`.

### H3 — **FAIL**

`P(>=64.5mm)` has BSS > 0 vs climatology and higher SEDI than the best member.
Both legs fail.

**Brier/BSS leg — fails at every threshold and every lead.** `tier2_bma` is
*worse* than IMD climatology on Brier throughout, all significant:

| threshold | tier2_bma | climatology | BSS (needs > 0) |
|---|---|---|---|
| 7.5 mm | 0.1435 | 0.0429 | −2.35 |
| 64.5 mm | 0.01236 | 0.00343 | **−2.60** |
| 115.6 mm | 0.00256 | 0.00069 | −2.70 |
| 204.5 mm | 0.00027 | 0.00007 | −2.82 |

(lead 24 h.) BSS at 64.5 mm is −2.60 to −2.62 across the five leads, where the claim needs
it above 0. Climatology is a strong forecast for a rare event precisely
because it is close to the base rate, and a 24–120 h probabilistic blend does
not beat simply knowing how often 64.5 mm falls.

**SEDI leg — fails at the claim's own threshold.** Against the best member:

| threshold | tier2_bma better at | significant at |
|---|---|---|
| 7.5 mm | 5/5 leads | 5/5 |
| 64.5 mm | **0/5** | 5/5 (worse) |
| 115.6 mm | **0/5** | 3/5 (worse) |
| 204.5 mm | 0/5 | 0/5 |

The blend's SEDI advantage exists only at 7.5 mm — light rain. At the 64.5 mm
threshold the claim names, it is worse at every lead and significantly so at
all five. Against climatology it is better at 7.5 mm; above that climatology's
SEDI is undefined, because a constant seasonal-mean forecast never reaches the
threshold and so has no hits or false alarms to score.

**Correction to an earlier draft of this file**, which reported this leg as
"better at 115.6 mm and worse at 204.5 mm". That was wrong on both counts: the
SEDI comparisons are computed at all four IMD thresholds, and the ordering is
*better at 7.5 mm, worse at everything above it*. A favourable reading at a
threshold the claim does not name was reported instead of the one it does.

This leg was previously `NOT_YET_TESTED` because Tier 2's per-day files carried
only hits/misses/false alarms/correct negatives, and a squared-probability
error is not reconstructible from counts. Tier 2 now records exceedance
probabilities per threshold, so the comparison exists and the claim is decided.

## 4. Tier 3: the regime-conditioned go/no-go failed

Re-running Phase 5's own criterion unchanged, on both daily seasons:

| lead | fold 2018 | fold 2020 | EMOS-CSG (2018 / 2020) |
|---|---|---|---|
| 24 h | 7.02 | 7.87 | 4.20 / 4.89 |
| 48 h | 6.90 | 7.90 | 4.21 / 4.90 |
| 72 h | 6.85 | 7.94 | 4.21 / 4.91 |
| 96 h | 6.87 | 8.02 | 4.24 / 4.95 |
| 120 h | 6.91 | 8.06 | 4.27 / 4.98 |

The regime-conditioned OLS blend loses to EMOS-CSG by 60–65% at every lead
and fold, and is essentially flat across lead times (7.02 → 6.91 for 2018 as
lead goes 24 → 120 h) where EMOS degrades slightly. A point-forecast OLS
blend conditioned on a binary active/break regime is decisively worse than a
calibrated censored-gamma. More data does not change the verdict.

## 5. Independence: three models, about 1.1 independent opinions

From `results/independence_diagnostic.csv`, per region and lead: $N_{eff}$
1.12–1.21, mean pairwise error correlation 0.74–0.84. This reproduces the
plan's F3 finding on the daily two-season base. Practical reading: adding a
fourth model should be expected to add little, which is directly relevant to
H4's NEPS-G go/no-go.

## 6. Drift: no drift flagged

`run_daily_verification.py` reports **0 of 15 (source, lead) cells flagged**
in 2018 and **0 of 15** in 2020. Drift does not fire across the 2018 → 2020
GraphCast checkpoint change at daily cadence.

Note: this script takes a single `--store` and predates the multi-season
refactor, so it was run once per season into
`results/daily_verification_2018.csv` and `_2020.csv` rather than as one
two-fold table. The comparison the prompt asks for is drawn from the two.

## 7. Open defects found while regenerating

### Resolved: the "degenerate fits" -- two causes, found in two passes

`scripts/gate_step07_results.py` used to flag two cells as degenerate, in both
the by-bin and by-region tables. Pooled CRPS, lead 24 h:

| bin | region | as first flagged | after 9a44833 | now |
|---|---|---|---|---|
| heavy | NE1 | 119.5 mm | 48.5 mm | **33.2 mm** |
| heavy | SI | 101.3 mm | 38.3 mm | **26.4 mm** |

For scale, the other regions in the same bin sit at 26.7 mm (CI). **The gate now
raises no advisories at all.**

**First cause: the Monte Carlo estimator (fixed in 9a44833).** The mixture draws
`X ~ Gamma(kappa, theta)` on the cube-root scale and then cubes it, so the
predictive variable is `Y = X**3`. `E[Y]` is `theta**3 kappa (kappa+1)
(kappa+2)`, not `(E[X])**3`, and `Var(Y)` is large enough that 500 draws cannot
represent it. Every summary taken from those draws was biased upward: the
predictive mean of heavy x SI read 327.6 mm, true 62.2 mm. `weavr.bma` now
computes the mean from `E[X**3]`, the exceedance probability from the gamma
survival function (exact, because cubing is monotone), and CRPS by integrating
the mixture CDF.

**Second cause: two clips that compounded (fixed in 636ac86).** 9a44833 called the
remaining excess "an estimator artifact", and that was wrong -- its integration
grid stopped at 1000 mm, so it could not see where the mass had gone. Over the
full support the affected fold-cells scored 145.4 and 155.0 mm. The mechanism:

1. The variance regression has a negative slope in these cells and
   extrapolated below zero on the test forecasts, so the variance was floored
   at `_TINY` (1e-6).
2. `bma_mixture_cdf` and `bma_exceedance_probability` then clipped
   `theta = var / mean` at `_TINY` *independently* of `kappa`, which moved the
   component mean from a cube-root 3.5 (~43 mm) to 12.3 (~1850 mm) and put
   58-71% of the mass above 1000 mm.

Fixes: no independent clip of `kappa` or `theta`; the predicted variance is
bounded to the range seen in training (`BmaComponentFit.min_variance_ct` /
`max_variance_ct`); `BMA_CRPS_GRID_MM` extends geometrically to 1e5 mm;
`bma_predictive_quantiles` solves levels beyond the grid by bisection instead of
extrapolating (it had returned 4,054,806 mm). Each fix has a test that fails on
the old code.

**H3's SEDI leg was affected, its verdict was not.** SEDI is built from
contingency counts of the predictive *mean*, which the estimator bias inflated:
at 64.5 mm, lead 24, BMA's SEDI moved 0.397 → 0.389, against 0.464 for the best
member. The Brier leg uses the exceedance probabilities, which were already
accurate. Both legs fail before and after.

### Three further defects fixed in 9a44833

- **CRPS was reduced to a scalar** before `score_bma_cells` assigned it into a
  per-cell grid, so every cell of a `(bin, region)` group received the *group*
  average, and the per-day BMA files had no within-group variation.
- **CRPS is now per-cell**, matching `weavr.emos.score_csgd` and what
  `score_bma_and_mean`'s docstring always claimed.
- **The trapezoid mishandled the step at the observation.** Integrating
  `(F - 1{x >= y})**2` directly biases the one interval straddling `y` by about
  half a grid step -- enough to score a point mass at zero against `obs = 3` as
  2.5 rather than 3.0. Splitting the integral at `y`, where the CDF is available
  in closed form, removes it exactly.

### Two earlier root-cause claims were wrong, and are retracted

1. The first attempt blamed `gamma_mean_intercept`/`gamma_mean_slope` as the only
   unbounded parameters and capped the predicted cube-root mean at the training
   maximum. That cap **provably does not bind**: a `heavy`-classified cell holds
   observations from **0 to 355 mm**, because bins are classified from the
   *forecast* rather than the observation. The slopes were never the problem
   either, measuring 0.04-1.73.
2. 9a44833 concluded that after the exact estimator the residual was not a fit
   problem. It was: the variance floor and the `theta` clip above.

Recorded because it cost several full regenerations: a well-formed number that
means the wrong thing passes every structural check, and a validation grid that
stops short of the support validates nothing beyond it.

### Verdicts after the fix

H1-RMSE PASS, H1-CRPS PASS, H2 FAIL (now against both comparators), H3 FAIL.
H1-RMSE strengthened from 3 to 5 significant leads and H2-vs-EMOS-IFS moved from
INSUFFICIENT_DATA to FAIL; both moves come from rebuilding the scorecard on the
current per-day files (see the H2 correction above), not from the BMA fix,
which moved domain BMA CRPS by at most 0.015 mm.

### Still open

Two cells remain mildly high rather than absurd: `heavy` x CI at 120 h
(+28.9 mm bias) and `heavy` x NE1 at 96 h (+28.4 mm). Neither is flagged by the
gate and neither moves a verdict, but both are the residue of the same
forecast-defined-bin problem and belong with any future revisit of the bins.

### Bin-level CRPS scales with the bin

`heavy` (target 64.5–115.6 mm) sits at 22–30 mm CRPS and `very_heavy` at
30–41 mm, while the median row is 8.3 mm. This is expected, and it is why the
gate compares a region against its own bin's group rather than against a
fixed threshold.

### Bin fittability under `MIN_TRAIN_DAYS_PER_BIN = 5`

**Correction.** An earlier draft of this file said all five bins now fit in
every fold and lead with no fallback remaining. That was wrong, and it
conflated two different granularities. `scripts/measure_bin_fittability.py`
measures the actual fit outcomes on the v2 train folds:

| lead | dry | light | heavy | very_heavy | extremely_heavy |
|---|---|---|---|---|---|
| 24 h | 4/4 | 4/4 | 4/4 | 4/4 | **0/4** |
| 48 h | 4/4 | 4/4 | 4/4 | 4/4 | **0/4** |
| 72 h | 4/4 | 4/4 | 4/4 | 4/4 | **0/4** |
| 96 h | 4/4 | 4/4 | 4/4 | 2/4 | **0/4** |
| 120 h | 4/4 | 4/4 | 4/4 | 2/4 | **0/4** |

Per bin, EMOS-CSG pooled over regions:

| lead | dry | light | heavy | very_heavy | extremely_heavy |
|---|---|---|---|---|---|
| 24 h | 12/12 | 12/12 | 12/12 | 3/12 | **0/12** |
| 48 h | 12/12 | 12/12 | 12/12 | 3/12 | **0/12** |
| 72 h | 12/12 | 12/12 | 12/12 | 1/12 | **0/12** |
| 96 h | 12/12 | 12/12 | 11/12 | 1/12 | **0/12** |
| 120 h | 12/12 | 12/12 | 11/12 | 1/12 | **0/12** |

So what actually changed against v1 is narrower than "everything now fits":

- **`very_heavy` improved and `extremely_heavy` did not.** The pooled bin fit
  clears 5 contributing days for `very_heavy` at every lead on the v2
  training window; on v1's 14-day weekly window it fell back from 72 h.
- **`extremely_heavy` still never fits**, at any lead, at either
  granularity — pooled over regions or per (bin, region). More data did not
  rescue it.
- **BMA is strictly harder than EMOS-CSG** because it must fit six regions per
  bin: `very_heavy` succeeds in only 1–3 of 12 bin×region×fold combinations
  at longer leads.

The v1 table in `docs/phase4-data-and-combiner-scope.md` counted
`extremely_heavy` at 0–3 contributing days over 14 `seasonal_block_split`
train days. The v2 training window is a whole season (~118–122 days), which
is what rescues `very_heavy`. Phase 4 is corrected to scope its claim to that
window.

**Discrepancy to raise.** #74's EVT evaluation reports that "all 5 rain bins
fitted in every fold and lead", which does not agree with the table above.
The likely cause is that EVT fits per-cell EMOS-CSG with its own binning
while Tier 2 fits per-bin EMOS-CSG and per-(bin, region) BMA, so the two are
counting different things. Neither number is wrong; they are different
questions, and this table is the one step 07 asks for. Worth reconciling with
arvnd before either is quoted as "the" fittability result.

## 8. Audit: every text still quoting a v1 number

`docs/preregistration.md` (23 numbers) is the pre-registration and is **not**
edited — it is the record of what was declared before the runs, and changing
it after seeing results would void the only thing that makes H1–H3 mean
anything.

Result docs carrying v1 numbers, which each need a dated v2 section:
`tier0-baseline-results.md`, `tier1-regional-weights-results.md`,
`tier2-hierarchical-baseline-results.md`, `single-source-and-independence-results.md`,
`phase2-ensemble-baseline-results.md`, `phase5-regime-conditioned-results.md`,
`scorecard-and-significance.md`. History and method docs
(`development-log.md`, `baseline-store.md`, `phase-1-data-requirements.md`,
`phase-3-cv-and-regional-scheme.md`, `phase5-regime-covariate-scope.md`,
`step-06-data-handoff.md`) get a dated pointer only.

A v1-vs-v2 table must **state** the cadence difference rather than set the
columns side by side. v1 was weekly end to end (`n_train=14`, `n_test=4`,
`split=seasonal_block_split`, test dates 2020-09-07/14/21/28) and v2 is
daily; most of the apparent change is the sampling, not the method.