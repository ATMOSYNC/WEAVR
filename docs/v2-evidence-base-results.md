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
| Scorecard comparisons | 2827, 56.0% significant |
| Degenerate bootstraps | **0** |
| Verdicts reached | **3** of 12 registered claims (2 PASS, 1 FAIL/INSUFFICIENT) |

**The headline result is a split verdict, and the interesting half is the
failure.** H1 passes twice over: the WEAVR blend beats the best single
ensemble member on RMSE at 3 of 5 leads and on CRPS at 5 of 5. H2 fails: the
multi-source combiner does **not** beat the best single-source EMOS
calibration on CRPS. The declared headline, `tier2_bma`, is the weaker of the
two probabilistic methods.

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

### H3 — **NOT_YET_TESTED** (Brier blocked; SEDI measurable and mixed)

The claim has two legs. The SEDI leg is measurable and **threshold-dependent**:

- At **115.6 mm**, `tier2_bma` has higher SEDI than the best member at all 5
  leads (e.g. lead 24: 0.583 vs 0.546, CI [−0.043, −0.032]) and higher than
  climatology at all 5 leads (0.583 vs 0.412, CI [−0.208, −0.132]).
- At **204.5 mm** the ordering reverses: BMA is worse than the best member at
  4 of 5 leads, significantly so at leads 24, 48, 120.

The Brier/BSS leg cannot be computed. Brier needs exceedance probabilities,
and `tier2`'s per-day files carry only hits/misses/false alarms/correct
negatives. Those reconstruct CSI, ETS and SEDI inside each bootstrap
replicate — which is why the SEDI leg works — but **not** Brier, whose
squared-probability error is not a ratio of counts. `climatology` does carry
`brier_*` columns and Tier 2 does not, so no `tier2_bma`-vs-`climatology`
Brier comparison exists to test. Closing this needs Tier 2 to pass
`probabilities=` into `per_day_scores`, i.e. a re-run.

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

**Unexplained.** Why `heavy`×NE1 and `heavy`×SI degenerate at lead 24 while
`heavy`×WC and `heavy`×CI do not is not yet root-caused. The fitting code was
deliberately not changed on the strength of numbers observed after the fact.

### Bin-level CRPS scales with the bin

`heavy` (target 64.5–115.6 mm) sits at 22–30 mm CRPS and `very_heavy` at
30–41 mm, while the median row is 8.3 mm. This is expected, and it is why the
gate compares a region against its own bin's group rather than against a
fixed threshold.

### Bin fittability under `MIN_TRAIN_DAYS_PER_BIN = 5`

All five rain bins now fit in every fold and lead on the two-season base, with
no point-mass fallback remaining — against the v1 weekly base, where
`extremely_heavy` never reached 5 contributing days at any lead. The constant
is unchanged. The phase 4 counts that produced the old claim were measured
over 14 `seasonal_block_split` train days; the v2 training window is a whole
season (~118–122 days). `docs/phase4-data-and-combiner-scope.md` is corrected
to scope that claim, and the extreme-probability map caption inherits the
correction.

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