# v2 Evidence Base Results

Two-season daily LOYO evidence base, regenerated under step 07.
**Two folds, about 120 test days each, 234–242 test days per lead.** Every number below is
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

> **Regenerated on the corrected forecast-observation pairing (#106).** Every
> number in this document now pairs each forecast with the IMD day it covers.
> Before #106, `weavr.grid` paired it with the day *before*: IMD labels a day by
> the date its 08:30 IST window ends, and the code assumed the start date (see
> `docs/grid-and-time-convention.md`). Skill was understated by that offset,
> so **every method scores better below than in earlier versions of this file**
> (Tier 0 RMSE at 24 h 14.61 -> 13.22 mm, EMOS-GraphCast CRPS 4.546 -> 4.139
> mm). One verdict moved: H2 against EMOS-IFS, FAIL -> INSUFFICIENT_DATA
> (§3). Test days are 242/240/238/236/234 per lead, two fewer than before,
> because the correct pairing needs the observed day after each forecast's
> valid window. The pre-#106 numbers are in git history at a9bc7d2.
>
> The regeneration also exposed an EMOS-CSG defect that only the corrected
> pairing triggered: see §7, "Found during the #106 regeneration".

---

## Summary

| | |
|---|---|
| Tiers regenerated | 0, 1, 2, 3, single-source, Phase 2, independence, daily verification |
| Test days per lead | 242 / 240 / 238 / 236 / 234 (leads 24→120h) |
| Scorecard comparisons | 6342, 66.6% significant (now including steps 11–13's methods) |
| Degenerate bootstraps | **0** |
| Verdicts reached | **5** of 12 registered claims: H1 PASS x2, H2 FAIL (vs EMOS-GraphCast) / INSUFFICIENT_DATA (vs EMOS-IFS), H3 FAIL |

**All three headline claims are decided; H1 passes and H3 fails, and H2 fails
against one comparator and is unproven against the other.** H1 passes twice
over: the blend beats the best single ensemble member on RMSE and on CRPS at 5
of 5 leads each. H2 fails against EMOS-GraphCast, which beats the multi-source
combiner significantly at every lead. Against EMOS-IFS, BMA is worse at 24 and
48 h but better at 72–120 h, significantly so at only 96 and 120 h, so the rule
records it as unproven rather than failed. H3 fails on its Brier leg: at 64.5 mm the
blend's Brier is 3.4× worse than climatology (BSS −2.4, where the claim needs
> 0). Its SEDI leg is mixed: worse than the best member at 64.5 mm at 3 of 5
leads, but significantly so only at 24 h.

So the blend is better than a single raw member (H1), worse than the best
single-source calibration (H2), and worse than climatology on heavy-rain
probability (H3). The declared headline, `tier2_bma`, is still behind
EMOS-GraphCast on both probabilistic claims it was declared to win.

---

## 1. What the evidence base is

Two-season leave-one-year-out. Each fold trains on one season and tests on
the other, so no season scores itself. Per-day scores carry both folds —
242 days at lead 24 falling to 234 at lead 120, because a longer lead pushes
the last initialisation times' valid times past the end of the observed
window. That arithmetic is asserted in `scripts/gate_step07_results.py`.

Both seasons' stores are daily, 122 initialisations × 50 IFS members × 5
leads on the 129×135 IMD grid.

## 2. Aggregate scores

Pooled across both folds, from `results/tier2_hierarchical_baseline.csv`:

| lead | tier0 RMSE | tier1 RMSE | EMOS-CSG CRPS | EMOS-IFS CRPS | BMA CRPS |
|---|---|---|---|---|---|
| 24 h | 13.22 | 12.66 | 4.139 | 4.155 | 4.279 |
| 48 h | 13.67 | 12.80 | 4.163 | 4.285 | 4.328 |
| 72 h | 14.07 | 12.95 | 4.199 | 4.408 | 4.376 |
| 96 h | 14.46 | 13.21 | 4.303 | 4.547 | 4.489 |
| 120 h | 14.64 | 13.41 | 4.389 | 4.623 | 4.572 |

Per fold, Tier 0 RMSE is 12.78–13.81 mm on the 2018 fold and 13.65–15.42 mm
on 2020. Tier 1 beats Tier 0 at every lead on both folds, by 0.59–1.00 mm on
2018 and 0.54–1.55 mm on 2020. (An earlier version of this paragraph said
0.35–0.82 mm; that range did not match its own CSV, which gave 0.70–1.16 mm
before #106.) 2020 is the harder season throughout — it is the wetter one
(land mean 7.45 mm against 6.29 mm in 2018).

Every score now degrades with lead, as it should. Before #106 they were nearly
flat in lead (Tier 0 14.61 → 14.83 mm), which is what scoring each forecast
against the neighbouring day would do: it removes most of the skill a shorter
lead adds.

### Two properties of these numbers worth stating plainly

**The probabilistic scores do beat climatology on CRPS -- a correction.** An
earlier version of this section said they did not, quoting a constant
seasonal-mean forecast at 3.33 mm (2018) and 3.45 mm (2020). Those numbers are
not reproducible: no script produces them, and no valid CRPS can. A constant
per-cell seasonal mean is a point forecast, so its CRPS is its MAE, 7.45 / 8.30
mm on land cells; even a climatology fitted *in-sample* on the test season
itself scores 4.63 / 5.24 mm. Against the project's real reference -- the
day-of-season IMD climatological ensemble with the test year excluded
(`weavr.climatology`), which is what the scorecard scores as `climatology` --
every calibrated method is better at every lead, every interval excluding 0:

| lead | EMOS-CSG | EMOS-IFS | BMA |
|---|---|---|---|
| 24 h | −0.866 | −0.849 | −0.726 |
| 48 h | −0.854 | −0.732 | −0.689 |
| 72 h | −0.835 | −0.625 | −0.657 |
| 96 h | −0.741 | −0.497 | −0.556 |
| 120 h | −0.667 | −0.434 | −0.485 |

(CRPS difference against `climatology`, mm, pooled over both folds; negative =
method better.) The heavy-rain *probabilities* are a different story: H3's
Brier leg below is decisively worse than climatology at 64.5 mm and above.

**BMA sits clearly behind EMOS-GraphCast, and crosses EMOS-IFS.** 4.279
against 4.139 mm (EMOS-CSG) and 4.155 mm (EMOS-IFS) at lead 24. EMOS-IFS
degrades with lead about twice as fast as the other two (4.155 → 4.623 mm,
against 4.139 → 4.389 for EMOS-GraphCast), so BMA overtakes it from 72 h on.
EMOS-GraphCast stays ahead of BMA by 0.14–0.19 mm at every lead. Since
`tier2_bma` is the declared headline, this directly shapes the verdicts below.

## 2b. v1 vs v2, and why the columns must not simply be compared

| | v1 | v2 |
|---|---|---|
| cadence | weekly | daily |
| seasons | 2020 only | 2018 + 2020 |
| splits | `seasonal_block_split`, 1 fold | leave-one-year-out, 2 folds |
| train days | 14 | 122 |
| test days | 3–4 | 117–121 per fold, 234–242 pooled |
| intervals | not computable | paired block bootstrap |

| lead | tier0 RMSE v1 → v2 | tier1 RMSE v1 → v2 | BMA CRPS v1 → v2 |
|---|---|---|---|
| 24 h | 11.55 → 13.22 | 11.08 → 12.66 | 3.721 → 4.279 |
| 48 h | 13.59 → 13.67 | 12.53 → 12.80 | 3.802 → 4.328 |
| 72 h | 13.15 → 14.07 | 12.05 → 12.95 | 3.830 → 4.376 |
| 96 h | 14.15 → 14.46 | 13.34 → 13.21 | 4.643 → 4.489 |
| 120 h | 15.35 → 14.64 | 14.90 → 13.41 | 4.454 → 4.572 |

**Most cells got worse, and that is not a regression.** v1 scored 4
test days, all in the last four weeks of September
(2020-09-07/14/21/28), on a weekly cadence. v2 scores ~120 days per fold
across the whole season, including the early-season days that are harder to
predict. The v1 numbers were flattered by the sample, and the v2 numbers are
the honest ones. v2 is better at the longest leads (Tier 0 and Tier 1 at
120 h, Tier 1 and BMA at 96 h), which is where v1's handful of test days was
least representative; it is not evidence that v2's methods improved there.

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

The #106 regeneration (Apple M3, 16 GB, 2026-10-30) ran Tier 2 and Tier 3 side
by side in **1 h 30 m** each (peak 7.7 GB and 7.5 GB RSS). The scorecard,
now 6342 comparisons, takes 40 s.

## 3. Verdicts

From `results/preregistration_verdicts.csv`. Claims H4–H11 belong to later
steps (08, 09, 11/12, 13, 14, 19, 21) and are correctly out of scope here.

### H1-RMSE — **PASS**

A WEAVR blend beats the best single member chosen on the training fold, on
RMSE. `tier2_bma` improves at 5 of 5 leads with the CI excluding 0 at every
one: by 0.62 mm at 24 h, narrowing to 0.25–0.28 mm at 96–120 h.

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
| 24 h | 6.2738 | 6.5799 | −0.3061 | [−0.3987, −0.2302] | yes |
| 48 h | 6.3617 | 6.8106 | −0.4489 | [−0.5470, −0.3739] | yes |
| 72 h | 6.4474 | 6.9977 | −0.5504 | [−0.6337, −0.4796] | yes |
| 96 h | 6.6391 | 7.0973 | −0.4582 | [−0.5398, −0.3873] | yes |
| 120 h | 6.7977 | 7.2425 | −0.4448 | [−0.5181, −0.3776] | yes |

5 of 5 leads, all intervals excluding 0.

### H2 — **FAIL** against EMOS-GraphCast, **INSUFFICIENT_DATA** against EMOS-IFS

The multi-source combiner does not beat the best single-source EMOS on CRPS.
It is significantly worse than EMOS-GraphCast at every lead. Against EMOS-IFS
it is worse at the short leads and better at the long ones.

| lead | vs EMOS-GraphCast | 95% CI | vs EMOS-IFS | 95% CI |
|---|---|---|---|---|
| 24 h | +0.1399 (BMA worse) | [+0.120, +0.162] | +0.1234 (BMA worse) | [+0.109, +0.139] |
| 48 h | +0.1652 | [+0.146, +0.186] | +0.0430 | [+0.013, +0.066] |
| 72 h | +0.1773 | [+0.157, +0.199] | −0.0319 (BMA better) | [−0.075, +0.003] |
| 96 h | +0.1854 | [+0.160, +0.213] | **−0.0586** | [−0.112, −0.018] |
| 120 h | +0.1824 | [+0.155, +0.211] | **−0.0513** | [−0.106, −0.010] |

Against EMOS-GraphCast every interval excludes 0, in BMA's disfavour. Against
EMOS-IFS, BMA is better at 3 of 5 leads but its interval excludes 0 at only 2
(96 and 120 h), one short of the pre-registered 3, so the scorecard records
INSUFFICIENT_DATA: unproven, not disproven.

**This is the one verdict #106 moved.** Before the corrected pairing, BMA was
significantly worse than EMOS-IFS at all five leads (+0.154 → +0.085 mm). What
changed is EMOS-IFS, not BMA: on the correct days it degrades with lead about
twice as fast as EMOS-GraphCast or BMA (§2). It does not move the claim, which is
about the *best* single-source EMOS. That is EMOS-GraphCast, and BMA fails
against it decisively.

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
The claim needs both legs. The Brier leg fails outright, and the SEDI leg does
not pass either.

**Brier/BSS leg — fails at every threshold and every lead.** `tier2_bma` is
*worse* than IMD climatology on Brier throughout, all significant:

| threshold | tier2_bma | climatology | BSS (needs > 0) |
|---|---|---|---|
| 7.5 mm | 0.1258 | 0.0429 | −1.93 |
| 64.5 mm | 0.01168 | 0.00345 | **−2.38** |
| 115.6 mm | 0.00250 | 0.00070 | −2.58 |
| 204.5 mm | 0.00027 | 0.00007 | −2.80 |

(lead 24 h.) BSS at 64.5 mm is −2.38 to −2.50 across the five leads, where the claim needs
it above 0. Climatology is a strong forecast for a rare event precisely
because it is close to the base rate, and a 24–120 h probabilistic blend does
not beat simply knowing how often 64.5 mm falls.

**SEDI leg — not better at the claim's own threshold.** Against the best member:

| threshold | tier2_bma better at | significant at |
|---|---|---|
| 7.5 mm | 5/5 leads | 4/5 (better; not at 48 h) |
| 64.5 mm | 2/5 (72, 120 h) | 1/5 (worse, at 24 h) |
| 115.6 mm | 1/5 (120 h) | 2/5 (worse, at 24 and 48 h) |
| 204.5 mm | — | — (BMA's predictive mean never scores a hit at 204.5 mm, so its SEDI is undefined) |

The blend's SEDI advantage is reliable only at 7.5 mm — light rain. At the
64.5 mm threshold the claim names, the two are within noise at four leads and
the blend is significantly worse at 24 h (0.475 against 0.520). Before #106 it
was significantly worse at all five leads; the corrected pairing narrowed the
gap but did not reverse it. Against climatology it is better at 7.5 mm; above that climatology's
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
| 24 h | 6.31 | 7.10 | 3.81 / 4.46 |
| 48 h | 6.29 | 7.22 | 3.84 / 4.48 |
| 72 h | 6.36 | 7.31 | 3.88 / 4.52 |
| 96 h | 6.56 | 7.52 | 3.99 / 4.61 |
| 120 h | 6.73 | 7.64 | 4.10 / 4.68 |

The regime-conditioned OLS blend loses to EMOS-CSG by 59–66% at every lead
and fold. Both now degrade with lead (before #106 the regime blend was flat,
7.02 → 6.91 on 2018). A point-forecast OLS blend conditioned on a binary
active/break regime is decisively worse than a calibrated censored-gamma.
Neither more data nor the corrected pairing changes the verdict.

## 5. Independence: three models, about 1.1 independent opinions

From `results/independence_diagnostic.csv`, per region, lead and fold: $N_{eff}$
1.09–1.23, mean pairwise error correlation 0.72–0.87 (domain-wide: 1.13–1.19
and 0.76–0.82). The corrected pairing barely moves it; the models' errors are
as correlated on the right day as on the wrong one. This reproduces the
plan's F3 finding on the daily two-season base. Practical reading: adding a
fourth model should be expected to add little, which is directly relevant to
H4's NEPS-G go/no-go.

## 6. Drift: no drift flagged

`run_daily_verification.py` reports **0 of 15 (source, lead) cells flagged**
(`results/daily_verification.csv`). It runs on both seasons concatenated, so
the rolling window crosses the 2018 → 2020 GraphCast checkpoint change, and
drift does not fire across it at daily cadence.

## 7. Open defects found while regenerating

### Resolved: the "degenerate fits" -- two causes, found in two passes

`scripts/gate_step07_results.py` used to flag two cells as degenerate, in both
the by-bin and by-region tables. Pooled CRPS, lead 24 h:

| bin | region | as first flagged | after 9a44833 | after 636ac86 | after #106 |
|---|---|---|---|---|---|
| heavy | NE1 | 119.5 mm | 48.5 mm | 33.2 mm | **33.3 mm** |
| heavy | SI | 101.3 mm | 38.3 mm | 26.4 mm | **31.1 mm** |

For scale, the other regions in the same bin sit at 21.5–30.7 mm. **The gate
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

### Still open -- now closed by #106

Two cells used to be mildly high rather than absurd: `heavy` x CI at 120 h
(+28.9 mm bias) and `heavy` x NE1 at 96 h (+28.4 mm). On the corrected pairing
they are at +1.0 mm and −18.5 mm. The largest remaining BMA bias in a fitted
heavy-rain cell is `heavy` x SI at 120 h, +12.0 mm over 164 cells. These cells
looked like a forecast-defined-bin problem; at least part of it was the pairing.

### Bin-level CRPS scales with the bin

Among fitted rows (pooled over folds, all three combiners), `heavy` (target
64.5–115.6 mm) sits at 21–37 mm CRPS and `very_heavy` at 23–38 mm, while the
median row is 9.3 mm. This is expected, and it is why the gate compares a
region against its own bin's group rather than against a fixed threshold.
Fallback rows go much higher, up to 204 mm for `very_heavy` x NE1 at 96 h. Those are 1–37
test cells each, in a bin BMA could not fit, scored with the pooled fallback.
(An earlier version quoted 22–30, 30–41 and 8.3 mm, which no row selection of
the committed CSVs reproduces.)

### Bin fittability under `MIN_TRAIN_DAYS_PER_BIN = 5`

**Correction.** An earlier draft of this file said all five bins now fit in
every fold and lead with no fallback remaining. That was wrong, and it
conflated two different granularities. `scripts/measure_bin_fittability.py`
measures the actual fit outcomes on the v2 train folds. Per bin, EMOS-CSG
pooled over regions (2 folds × 2 sources = 4 fits per cell):

| lead | dry | light | heavy | very_heavy | extremely_heavy |
|---|---|---|---|---|---|
| 24 h | 4/4 | 4/4 | 4/4 | 4/4 | **0/4** |
| 48 h | 4/4 | 4/4 | 4/4 | 4/4 | **0/4** |
| 72 h | 4/4 | 4/4 | 4/4 | 4/4 | **0/4** |
| 96 h | 4/4 | 4/4 | 4/4 | 2/4 | **0/4** |
| 120 h | 4/4 | 4/4 | 4/4 | 2/4 | **0/4** |

Per (bin, region), BMA (6 regions × 2 folds = 12 fits per cell). (This table and
the one above used to carry each other's captions.)

| lead | dry | light | heavy | very_heavy | extremely_heavy |
|---|---|---|---|---|---|
| 24 h | 12/12 | 12/12 | 12/12 | 3/12 | **0/12** |
| 48 h | 12/12 | 12/12 | 12/12 | 3/12 | **0/12** |
| 72 h | 12/12 | 12/12 | 12/12 | 1/12 | **0/12** |
| 96 h | 12/12 | 12/12 | 11/12 | 1/12 | **0/12** |
| 120 h | 12/12 | 12/12 | 11/12 | 1/12 | **0/12** |

So what actually changed against v1 is narrower than "everything now fits":

- **`very_heavy` improved and `extremely_heavy` did not.** The pooled bin fit
  clears 5 contributing days for `very_heavy` in every fit at 24–72 h and in
  2 of 4 at 96–120 h on the v2 training window; on v1's 14-day weekly window
  it fell back from 72 h.
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

### Found during the #106 regeneration

Three defects, none caused by #106, surfaced when the corrected pairing was run
end to end. Each fix has a test that fails on the old code.

1. **EMOS-CSG scored a whole bin as a perfect forecast (`weavr.emos.csgd_crps`).**
   At 120 h on the 2020 fold, the `dry` bin's climatological fit collapsed to
   a mean of 1e-6 mm, so the predicted gamma shape `kappa` reached ~1e-8 and the
   scale `theta` ~4e8. `_gamma_cdf` and `beta_fn` clipped the shape, but the rest
   of the closed form used the raw `kappa`. The two halves disagreed, and the
   CRPS came out as exactly 0 on all 312,251 test cells of the bin, while their
   bias was +399.8 mm. This hit EMOS-GraphCast and EMOS-IFS alike. Before #106
   the same fit landed on a healthy shape, which is why no earlier run showed
   it. `csgd_crps` now floors `kappa` at `_TINY` and derives `theta` from it,
   so both halves see one distribution. Refit, the bin's train CRPS is 1.69 mm,
   and the closed form matches brute-force integration. The 2018 fold is
   byte-identical, and healthy cells move by at most 4e-13 mm.
2. **The result gate could not see it.** Every number was finite, and a CRPS of
   0 is a legal value. `gate_step07_results.py` now blocks any fitted
   (non-fallback) breakdown row with CRPS ≤ 1e-6 mm (`check_zero_crps`). On the
   broken run it flags exactly the two affected rows.
3. **The scorecard had never been run with step 13's per-day files present.**
   `tier2b_*` methods score only probabilistically, so their daily MSE and
   contingency counts are empty by design, and `run_scorecard.py` raised on the
   first all-NaN RMSE comparison. RMSE and CSI/SEDI are now skipped for a
   method with nothing finite to compare, as every other metric already was.
   Steps 11–13's methods now appear in `results/scorecard.csv`. No verdict reads
   them: H10 is decided in `docs/tier2b-combined-results.md`, and its scorecard
   spec still names an undeclared method.

The same regeneration fixed defects outside step 07, recorded where they
belong: Tier 2b's per-bin train CRPS and its nomination weight
(`docs/tier2b-combined-results.md` §7; H10 now passes 5 of 5), and the
dashboard map's bin classification and per-cell tail markers
(`docs/tail-repair-results.md` §3).

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