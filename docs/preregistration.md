# Pre-registration of WEAVR's headline claims

**Registered: 2026-09-29.** Merged before the v2 evidence base
(steps 05–07) produced a single number.

This document fixes, in advance, every claim WEAVR hopes to make and exactly
how each will be judged. It exists so that the criteria provably came before
the results — the git history of this file is the proof.

Phase 5 did this once: Tier 3's go/no-go was stated in the script's docstring
before the script was run, and the answer came back a clean NO-GO, which was
published as prominently as a pass would have been
([docs/phase5-regime-conditioned-results.md](phase5-regime-conditioned-results.md),
step 4). That is the precedent. This document makes it the rule for every
claim that reaches a slide.

Verdicts are computed by `scripts/run_scorecard.py` (step 04) into
`results/preregistration_verdicts.csv`. Nothing in this file is a result.

---

## The rules

1. **Only claims that PASS may appear on a slide as findings.** Everything
   else is labelled **exploratory**, in those words.
2. **A failed claim is reported with the same prominence as a passed one**,
   in the results doc of the step that tests it. Not a footnote.
3. **Criteria change only in a separate, dated commit that says why, and
   never after that claim's results have been computed.** Every amendment is
   logged in §5 below.
4. **No claim may be judged on a configuration chosen after seeing test
   scores.** See "the selection rule" below — this is the loophole that
   makes pre-registration meaningless if left open.
5. If a claim cannot be evaluated (no data, too few events), it is recorded
   as **NOT EVALUABLE** with the reason. That is not a pass.

---

## 1. Shared procedure

Every claim below inherits this unless it states otherwise.

**Data.** JJAS (1 June – 30 September), 2018 and 2020, daily cadence, on
WEAVR's locked 0.25° India grid, verified against `imd_observed.rain`.
Leads 24, 48, 72, 96, 120 h. Rainfall sources: `graphcast`, `hres`,
`ifs_ens_mean`, plus `neps_g` where a claim names it.

**Split.** Leave-one-year-out over 2018 and 2020 — both folds, with the
daily scores pooled. This is the first time `leave_one_year_out` is usable
in this project; every result before step 07 used `seasonal_block_split` on
a single season.

**Confidence intervals.** 95% paired block bootstrap on daily domain-wide
scores, 1,000 resamples, block length **7 days**.

> The block length is a placeholder for a measurement. If step 07 measures a
> different daily-score autocorrelation length, this document is updated in
> its **own commit, before any verdict is computed**, and the change is
> logged in §5.

**Default pass rule.** A claim passes when the improvement holds at **≥ 3 of
5 leads**, *and* the 95% CI of the paired difference **excludes 0** at those
same leads. Both conditions, not either.

The "≥ 3 of 5 leads" framing is the one every prior tier in this project has
used for its headline comparison, kept for continuity. The CI condition is
new in v2 and is what stops a 3–4 day sampling artefact from counting as a
result — step 01 found the train-chosen and test-chosen best source
disagreeing at 2 of 5 leads on the old test sets, which is exactly what this
rule is designed to catch.

### The selection rule

Several claims below compare against "the best" of a set. Choosing that
member after seeing test scores would make any claim passable, so:

- **Every "best of" is chosen on training data only**, never on test. Step
  01's `best_single_member_on_train` is the pattern, and it is deliberately
  the weaker, honest baseline: its ranking disagreed with the test ranking
  at 2 of 5 leads.
- **The headline WEAVR configuration is declared before verdicts are
  computed.** Step 07 scores every tier (Tier 0, Tier 1, Tier 2 EMOS-CSG,
  Tier 2 BMA, and Tier 2b if step 13 lands) and reports them all — but H1
  and H2 are decided by the *one* pre-declared configuration, recorded in
  `docs/v2-evidence-base-results.md` before the verdict run. Reporting five
  configurations and claiming the best is five chances at a 1-in-20 error,
  not one.

---

## 2. The H1 metric decision

**Decided 2026-09-29, before any v2 result existed. H1 is judged on RMSE and
CRPS separately, as two verdicts — `H1-RMSE` and `H1-CRPS` — which are never
merged into a single answer.**

The tradeoff, stated plainly so the decision can be audited:

- **RMSE alone** is the traditional deterministic view and what most of the
  literature reports. But the research brief's §1.3 notes that MSE-trained
  models look better on RMSE because it rewards smooth, hedged forecasts,
  and AI models are MSE-trained. Step 01 measured exactly that: raw
  GraphCast beats every blend on RMSE at 24–96 h **while detecting none of
  the 115.6 mm events at any lead**. Judging H1 on RMSE alone risks failing
  it for a reason that has little to do with whether the forecast is useful.
- **CRPS alone** is what the brief recommends for blends (§1.2: operational
  blends blend probabilities, because extremes guidance needs exceedance
  probabilities). But adopting it *immediately after* step 01 showed RMSE is
  unfavourable would look like moving the goalposts — the precise criticism
  pre-registration exists to prevent.
- **Both, separately** keeps the metric we are currently losing on the
  record, which is the point. It also matches operational practice: BoM's
  IMPROVER study fitted MSE-optimal weights per lead time and then checked
  CRPS as well.

The cost is accepted knowingly: the answer to "is your blend better?" may be
two-part ("it loses on RMSE at short leads and wins on CRPS"), which is
harder to put on a slide than a single number. That is the honest shape of
the result.

---

## 3. The claims

| ID | Claim | Tested by | Type |
|---|---|---|---|
| H1 | A WEAVR blend beats the best single member (chosen on train) | 07 | Pass/fail ×2 |
| H2 | The multi-source probabilistic combiner beats the best single-source EMOS on CRPS | 07, 13 | Pass/fail |
| H3 | P(≥64.5 mm) has BSS > 0 vs climatology, and higher SEDI than the best single member | 07 | Pass/fail |
| H4 | Adding NEPS-G improves CRPS at 24 h | 08 | Pass/fail |
| H5 | WEAVR vs HEPPI's published NEPS-G EMOS on CRPS at 24 h | 08 | Report either way |
| H6 | Independence-aware weights beat OLS `fit_region_weights` on RMSE | 09 | Pass/fail |
| H7 | Tail repair improves tw-CRPS or SEDI without harming light-rain Brier | 11, 12 | Pass/fail |
| H8 | District Orange/Red warnings have higher CSI than the same rule on the best raw ensemble | 14 | Pass/fail |
| H9 | Event replay: first lead at which Orange and Red are issued | 19 | Report either way |
| H10 | A stacked or per-bin-selected combiner beats both EMOS and BMA on CRPS | 13 | Pass/fail |
| H11 | Online weights beat static Tier 1 weights on a sequential replay | 21 | Pass/fail |

---

### H1 — A WEAVR blend beats the best single member

**Claim.** The pre-declared WEAVR headline blend has lower error than the
best single rainfall source, where "best" is chosen on the training fold.

- **Data.** 2018 + 2020 JJAS daily; leads 24–120 h; `graphcast`, `hres`,
  `ifs_ens_mean`.
- **Split.** LOYO, both folds pooled.
- **Metrics.** **RMSE and CRPS, separately** (see §2). Two independent
  verdicts, `H1-RMSE` and `H1-CRPS`.
- **Comparison.** `best_single_member_on_train`, regenerated by
  `scripts/run_single_source_baseline.py` under the v2 split. For CRPS, the
  single-member comparator is that source's own EMOS-CSG calibration, not
  the raw deterministic source. A deterministic forecast has no predictive
  distribution, so its CRPS can only be scored via the point-mass identity
  (CRPS = absolute error) — and
  [the Tier 3 results doc](phase5-regime-conditioned-results.md) records why
  that is a structural disadvantage rather than a fair comparison: a point
  forecast has no way to hedge an uncertain day, so part of any gap is the
  shape of the output, not the skill behind it. Beating a handicapped
  comparator would not be evidence of anything.
- **Pass.** Default rule, applied to each metric independently.
- **Recorded as.** `H1-RMSE`, `H1-CRPS`. A split verdict (one passes, one
  fails) is a legitimate outcome and is reported as such — not rounded to
  "WEAVR wins".

### H2 — The multi-source combiner beats the best single-source EMOS

**Claim.** The best multi-source probabilistic combiner beats the best
single-source EMOS-CSG on CRPS.

- **Data / split.** As H1.
- **Metric.** CRPS (domain-wide, pooled daily).
- **Comparison.** Multi-source candidate (BMA, or the stacked combiner from
  step 13) **selected on the training fold**, against the best single-source
  EMOS-CSG, also selected on the training fold.
- **Pass.** Default rule.
- **Why this matters.** Step 01 found the only blends beating raw GraphCast
  were Tier 2's EMOS runs — which are *single-source calibrations*. H2 is
  the direct test of whether combination adds anything beyond calibration.

### H3 — Heavy-rain probability has real skill

**Claim.** WEAVR's P(≥ 64.5 mm) has a Brier skill score above zero against
IMD climatology, **and** a higher SEDI at 64.5 mm than the best single
member.

- **Data / split.** As H1.
- **Metrics.** Brier skill score (reference: IMD climatological exceedance
  frequency per gridpoint, from `data/imd_seeps_climatology_jjas.zarr`,
  **built excluding the test year** of the fold being scored); SEDI at
  64.5 mm.
- **Pass.** Both conditions, each under the default rule. BSS > 0 with a CI
  excluding 0, and SEDI higher than the best single member.
- **Note on the threshold.** 64.5 mm is the highest IMD threshold with
  enough events for a standalone probabilistic skill claim. Step 01, on 3–4
  test days per lead, measured **zero** test cells in the 204.5 mm bin at
  every lead; at 115.6 mm, HRES detected some events (POD 0.118 / 0.027 /
  0.167 at 24 / 48 / 72 h) while **GraphCast detected none at any lead**.
  115.6 mm therefore has events but very few, which is why it appears in H7
  only as one of two alternative improvement routes and not as a claim of
  its own. See §4 for 204.5 mm.

### H4 — NEPS-G adds skill

**Claim.** Adding NEPS-G to the blend improves CRPS at 24 h over the same
blend without it.

- **Data.** **2018 JJAS only**, 24 h lead.
- **Split.** `seasonal_block_split` within 2018, *not* LOYO — see the scope
  note below.
- **Metric.** CRPS.
- **Comparison.** Identical blend configuration, identical days, with and
  without NEPS-G as a member.
- **Pass.** The CI of the paired CRPS difference excludes 0, in the
  improving direction. The "≥ 3 of 5 leads" condition does not apply: this
  is a single lead.

> **Scope note, checked rather than assumed.** HEPPI covers JJAS 2018 and
> 2019. WEAVR's stores will cover 2018 and 2020. The overlap is **2018
> only**, and it cannot be widened: GraphCast's two WeatherBench 2 windows
> are 2017-11-16…2019-02-01 and 2019-11-16…2021-01-31, so **there is no
> GraphCast JJAS 2019** and a 2019 WEAVR store is impossible. H4 is
> therefore a single-season, single-lead claim, and is registered as such
> rather than being quietly evaluated on a season that doesn't exist.

### H5 — WEAVR against HEPPI's published benchmark

**Claim.** None. This is a **report-either-way** comparison.

- **Data.** 2018 JJAS, 24 h, on identical days to HEPPI's published
  evaluation.
- **Metric.** CRPS.
- **Comparison.** WEAVR's blend against HEPPI's published NEPS-G EMOS
  (Angus et al. 2024).
- **Reported.** The result is published whichever way it falls. A **win is
  claimed only if the CI of the paired difference excludes 0**; anything
  else is reported as "no measurable difference" or as a loss.
- **Constraint.** HEPPI raw data, and grids derived from it, are not
  committed or published without the user confirming the licence. Derived
  scores and metadata are fine, with a citation.

### H6 — Independence-aware weighting beats OLS

**Claim.** Weights fitted from a shrinkage estimate of the error covariance
beat OLS `fit_region_weights` on RMSE. Step 09's go/no-go.

- **Data / split.** As H1.
- **Metric.** RMSE (domain-wide). Per-region RMSE reported alongside, but
  the verdict is domain-wide.
- **Pass.** Default rule.
- **Prior.** Step 01 measured N_eff ≈ 1.1 of 3. With that little diversity
  there may be very little for any weighting scheme to exploit; a NO-GO here
  is a plausible and publishable outcome, and the accompanying Shapley
  contributions are expected to be unstable across regions. That instability
  is reported, not smoothed.

### H7 — Tail repair helps the tail without breaking the rest

**Claim.** Tail repair (quantile mapping and/or the extreme-value tail)
improves threshold-weighted CRPS at t = 64.5 mm **or** SEDI at 115.6 mm,
**without** making the Brier score at 7.5 mm worse.

- **Data / split.** As H1.
- **Metrics.** tw-CRPS (t = 64.5 mm); SEDI at 115.6 mm; Brier at 7.5 mm.
- **Pass.** *Both* of:
  - **Improvement:** at least one of tw-CRPS or SEDI passes the default rule.
  - **Non-inferiority:** the 95% CI of the paired Brier difference at 7.5 mm
    does **not** lie entirely in the worse direction, at any lead.
- **Why the guard.** A tail repair that inflates every forecast would
  improve extreme detection and quietly wreck the light-rain forecast that
  most users see most days. The non-inferiority condition is what makes this
  claim meaningful.
- **Pre-registered in advance:** the GPD threshold selection rule for step 12
  is fixed **before** fitting, and recorded in `docs/tail-repair-results.md`.
  A threshold chosen after seeing the fit would decide the result.

### H8 — District warnings beat the same rule on a raw ensemble

**Claim.** District-level Orange/Red warnings derived from WEAVR have higher
CSI than the identical warning rule applied to the best single raw ensemble.

- **Data / split.** As H1, aggregated to district polygons.
- **Metric.** CSI for Orange and Red, reported separately.
- **Comparison.** The **same** rule and thresholds applied to the best raw
  ensemble (chosen on train). The rule is held fixed across both sides; only
  the input forecast changes.
- **Pass.** Default rule, for Orange and Red independently.

### H9 — Event replay

**Claim.** None. This is a **report-either-way** result.

- **Reported.** The first lead time at which WEAVR issues Orange, and the
  first at which it issues Red, over the affected districts — shown next to
  the same figures for each raw model.
- **No pass/fail.** It is shown whatever the outcome, including "WEAVR never
  issued Red".
- **Fixed in advance.** The event and the affected district list are fixed
  in step 19 **before** any replay result is viewed, and recorded in
  `docs/event-replay-results.md`.
- **Prior.** Step 01 found no source detecting any 204.5 mm event and
  GraphCast detecting no 115.6 mm event at any lead. A replay showing WEAVR
  missing the peak is a realistic outcome and is presented honestly; "here
  is what we would have warned, and here is what we would have missed" is a
  stronger answer to a jury than a result that collapses under one question.

### H10 — Stacking beats both parents

**Claim.** A stacked or per-bin-selected combiner beats **both** EMOS-CSG
and BMA on CRPS.

- **Data / split.** As H1.
- **Metric.** CRPS.
- **Pass.** Default rule, against **each** parent separately. Beating one
  but not the other is a fail, and is reported as such.
- **Note.** The per-bin selection must be fitted on the training fold. A
  per-bin winner chosen on test is the "strongest combiner per bin" claim
  that step 01 removed from the slides.

### H11 — Online weights beat static weights

**Claim.** Online (sequentially updated) weights beat static Tier 1 weights
on a sequential replay of the held-out season.

- **Data.** The held-out LOYO fold, replayed day by day in chronological
  order.
- **Metric.** RMSE and CRPS, cumulative over the replay.
- **Comparison.** Static Tier 1 weights fitted on the training fold,
  applied unchanged.
- **Pass.** Cumulative score lower at the end of the replay, with the CI of
  the paired daily difference excluding 0.
- **Constraint.** The replay is strictly causal: the weights at day *t* use
  only observations up to day *t−1*. Any leak invalidates the claim.

---

## 4. What is deliberately not registered

Stating these prevents them being quietly claimed later:

- **Any claim at 204.5 mm**, for now. Step 01 measured zero test cells in
  that bin at every lead — but that was on 3–4 test days, and v2 moves to
  roughly 240 daily samples across two seasons, so the event count is
  genuinely unknown until step 07 exists. This is registered as "not
  claimed" rather than "unevaluable": **if v2 turns out to contain enough
  204.5 mm events, a claim may be added by dated amendment under §5, before
  that claim's verdicts are computed.** Barring it permanently on the
  strength of a 3-day sample would be its own kind of over-claiming.
- **Any claim that WEAVR is "operationally ready"** or deployed at IMD.
- **Any economic or rupee-value figure** as a measured result. Step 15's
  value bands are illustrative and labelled as such.
- **Any temperature or wind claim** unless step 24 runs and registers one by
  amendment.
- **Any claim about NCUM.** NEPS-G via HEPPI is the only NCMRWF source in
  scope.

---

## 5. Amendment log

Every change to a criterion, dated, with its reason. Empty at registration.

| Date | Claim | Change | Reason | Results already computed? |
|---|---|---|---|---|
| 2026-09-29 | — | Initial registration | — | No |

A row with "Results already computed? **Yes**" would invalidate that claim.
There should never be one.
