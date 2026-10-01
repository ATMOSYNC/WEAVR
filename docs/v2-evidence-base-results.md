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
| Scorecard comparisons | 3047, 58.9% significant |
| Degenerate bootstraps | **0** |
| Verdicts reached | **4** of 12 registered claims: H1 PASS x2, H2 FAIL, H3 FAIL |

**All three headline claims are now decided, and two of the three fail.** H1
passes twice over: the blend beats the best single ensemble member on RMSE at
3 of 5 leads and on CRPS at 5 of 5. H2 fails — the multi-source combiner does
**not** beat the best single-source EMOS calibration on CRPS. H3 fails on both
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
| 24 h | 14.24 | 13.42 | 4.546 | 4.547 | 4.643 |
| 48 h | 14.12 | 13.30 | 4.554 | 4.554 | 4.577 |
| 72 h | 13.97 | 13.17 | 4.560 | 4.560 | 4.569 |
| 96 h | 13.89 | 13.15 | 4.596 | 4.596 | 4.597 |
| 120 h | 13.90 | 13.20 | 4.622 | 4.622 | 4.627 |

Per fold, Tier 0 RMSE is 13.90–14.24 mm on the 2018 fold and 14.98–15.71 mm
on 2020. Tier 1 beats Tier 0 at every lead on both folds, by 0.35–0.82 mm.
2020 is the harder season throughout — it is the wetter one (land mean 7.45 mm
against 6.29 mm in 2018).

### Two properties of these numbers worth stating plainly

**The probabilistic scores do not beat climatology.** A constant forecast at
the seasonal mean scores CRPS 3.33 mm on 2018 land cells and 3.45 mm on 2020;
the calibrated methods score 4.55 and 4.62. This is not a v2 regression — the
committed v1 weekly numbers had the same property (3.76 mm against a 3.45 mm
floor for 2020). Nothing pre-registered turns on beating climatology on CRPS:
H1 and H2 are *relative* comparisons against other WEAVR configurations.
It belongs here because a reader of this table will otherwise assume the
probabilistic layer is skilful against climatology, and it is not.

**BMA sits marginally behind single-source EMOS-CSG.** 4.643 against 4.546 mm
at lead 24, converging to a dead heat by lead 96. Since `tier2_bma` is the
declared headline, this directly shapes the verdicts below.

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
| 24 h | 11.55 → 14.61 | 11.08 → 13.86 | 3.721 → 4.643 |
| 48 h | 13.59 → 14.71 | 12.53 → 13.86 | 3.802 → 4.577 |
| 72 h | 13.15 → 14.77 | 12.05 → 13.84 | 3.830 → 4.569 |
| 96 h | 14.15 → 14.78 | 13.34 → 13.85 | 4.643 → 4.597 |
| 120 h | 15.35 → 14.83 | 14.90 → 13.89 | 4.454 → 4.627 |

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
| scorecard, 2827 comparisons × 1000 bootstrap resamples | **~1 min** |
| result gate | **~2 s** |
| **full end-to-end rehearsal on miniature stores** | **~7 min** |

The rehearsal figure is the important one: every runner can be executed for
real, CLI and exit codes included, against schema-faithful stores at ~1/1000
scale in about the time it takes to make coffee. Run it before any multi-hour
regeneration, not after the next loss.

Peak RSS was 2.6 GB for Tier 2 and 4.7 GB for Tier 3, both inside the memory
budget the blocked CRPS and cell-blocked BMA work was built for.

## 3. Verdicts

From `results/preregistration_verdicts.csv`. Claims H4–H11 belong to later
steps (08, 09, 11/12, 13, 14, 19, 21) and are correctly out of scope here.

### H1-RMSE — **PASS**

A WEAVR blend beats the best single member chosen on the training fold, on
RMSE. `tier2_bma` improves at 3 of 5 leads with the CI excluding 0, at leads
72, 96 and 120 h.

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
| 24 h | 7.2333 | 7.4312 | −0.1979 | [−0.3888, −0.0275] | yes |
| 48 h | 7.1243 | 7.4614 | −0.3371 | [−0.4816, −0.2194] | yes |
| 72 h | 7.1288 | 7.5628 | −0.4340 | [−0.5821, −0.3167] | yes |
| 96 h | 7.1794 | 7.5871 | −0.4076 | [−0.5325, −0.3102] | yes |
| 120 h | 7.2379 | 7.6138 | −0.3759 | [−0.4878, −0.2747] | yes |

5 of 5 leads, all intervals excluding 0.

### H2 — **FAIL** against EMOS-GraphCast, **INSUFFICIENT_DATA** against EMOS-IFS

The multi-source combiner does not beat the best single-source EMOS on CRPS.

| lead | vs EMOS-GraphCast | vs EMOS-IFS |
|---|---|---|
| 24 h | +0.0975 (BMA worse) | +0.0641 (BMA worse) |
| 48 h | +0.0234 | −0.0208 |
| 72 h | +0.0089 | −0.0461 |
| 96 h | +0.0008 | −0.0692 |
| 120 h | +0.0045 | −0.0891 |

Not one lead reaches significance against either comparator. Against
GraphCast-EMOS the difference is non-negative at every lead — BMA is never
even nominally better, hence FAIL. Against IFS-EMOS it is nominally better at
4 of 5 leads but never significantly, hence INSUFFICIENT_DATA: the direction
leans BMA's way and the data cannot resolve it.

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
| 7.5 mm | 0.1438 | 0.0429 | −2.35 |
| 64.5 mm | 0.01238 | 0.00343 | **−2.61** |
| 115.6 mm | 0.00257 | 0.00069 | −2.73 |
| 204.5 mm | 0.00029 | 0.00007 | −2.90 |

BSS at 64.5 mm is −2.61 to −2.63 across the five leads, where the claim needs
it above 0. Climatology is a strong forecast for a rare event precisely
because it is close to the base rate, and a 24–120 h probabilistic blend does
not beat simply knowing how often 64.5 mm falls.

**SEDI leg — fails at the claim's own threshold.** Against the best member:

| threshold | tier2_bma better at | significant at |
|---|---|---|
| 7.5 mm | 5/5 leads | 5/5 |
| 64.5 mm | **0/5** | 4/5 (worse) |
| 115.6 mm | **0/5** | 3/5 (worse) |
| 204.5 mm | 0/5 | 0/5 |

The blend's SEDI advantage exists only at 7.5 mm — light rain. At the 64.5 mm
threshold the claim names, it is worse at every lead and significantly so at
four of five. Against climatology it is better at 7.5 mm and worse at all
three higher thresholds.

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

### The headline's per-(bin, region) fits include degenerate cases

`scripts/gate_step07_results.py` flags two, in both the by-bin and by-region
tables:

| bin | region | lead | CRPS | reference | RMSE |
|---|---|---|---|---|---|
| heavy | NE1 | 24 h | 119.5 mm | 27.9 mm (3.6×… 4.3×) | 666.8 mm |
| heavy | SI | 24 h | 101.3 mm | 27.9 mm (3.6×) | 748.4 mm |

Observed daily accumulations reach ~400 mm at most, so an RMSE of 667–748 mm
is arithmetically impossible as a forecast error: the fitted mixture's scale
has degenerated. Other regions in the same bin and lead are healthy
(22–36 mm CRPS), so this is regional, not global.

These cell-days are ~0.2% of the ~564,000 scored, so the cell-weighted domain
CRPS is 4.6 mm and **no verdict above moves**. They are reported as advisories
rather than blocking failures for exactly that reason. Median by-bin CRPS is
8.3–8.4 mm and consistent across all three combiners, so the bulk of the
breakdown is sound.

**Still unresolved, and an earlier root-cause claim in this file's history was
wrong.** The first attempt blamed `gamma_mean_intercept`/`gamma_mean_slope` as
the only unbounded parameters, and `weavr.bma` now caps the predicted
cube-root mean at the training maximum. That cap **provably does not bind here**
— re-running Tier 2 with it in place reproduced `heavy`×SI to six decimal
places (CRPS 145.263122, bias +277.369185), which is how the misdiagnosis was
caught. The mean regressors are in fact well conditioned; measured on the
2020-trained fold, `heavy`×SI has mean intercepts +3.31/+1.66/+2.13 against
+2.33 for healthy `heavy`×NE2.

What *does* differ is the variance: `heavy`×SI's cube-root variance intercepts
are 2.75/2.96/1.09 where `heavy`×NE2's are 0.58/1.16/1.10. Since the sampler
draws `Gamma(kappa = mean_ct²/variance_ct, theta = variance_ct/mean_ct)` and
then cubes, a fat gamma tail in cube-root space produces rare enormous draws
whose sample mean dominates a 453-cell fold's pooled bias. That is the
mechanism consistent with a +277 mm bias on a bin whose own scale is ~90 mm,
but it is **not confirmed**, and no fix is claimed.

The bound was kept because it is a legitimate guard that is inert on
well-conditioned fits (asserted to 1e-9 in `tests/test_bma_mean_bound.py`),
not because it fixed this.

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