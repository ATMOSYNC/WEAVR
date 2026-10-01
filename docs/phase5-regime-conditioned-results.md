# Phase 5 step 3: regime-conditioned weighting -- assessment and fit

Started here per
[`solving issues/06-phase-5-regime-conditioned-weighting/03-fit-regime-conditioned-weighting.md`](../../solving%20issues/06-phase-5-regime-conditioned-weighting/03-fit-regime-conditioned-weighting.md);
finished in step 4 once Tier 3's own numbers exist to score against Tier 2.

## Step 3.1: is there real evidence a regime covariate explains Phase 4's error pattern?

Checked directly against the real train/test data every Tier 2 combiner
was fit and scored on
(`scripts/run_tier2_hierarchical_baseline.py`'s own `seasonal_block_split`,
same stores), not assumed from issue #7's framing alone.

**The held-out test window has essentially zero regime diversity, at every
lead.** `seasonal_block_split`'s trailing test block puts all 3-4 test
samples for every lead in the same narrow late-September window (e.g. lead
24h's test days are 2020-09-07, -14, -21, -28). Looking up
`src/weavr/regimes.py`'s own labels for those exact days:

| Lead (h) | Test days | Monsoon phase | Depression present |
|---|---|---|---|
| 24  | 09-07, 09-14, 09-21, 09-28 | neutral, neutral, neutral, break | False (all 4) |
| 48  | 09-08, 09-15, 09-22, 09-29 | neutral, neutral, neutral, break | False (all 4) |
| 72  | 09-09, 09-16, 09-23, 09-30 | neutral, neutral, neutral, break | False (all 4) |
| 96  | 09-10, 09-17, 09-24 | neutral, neutral, neutral | False (all 3) |
| 120 | 09-11, 09-18, 09-25 | neutral, neutral, neutral | False (all 3) |

No test day at any lead is ever "active" or depression-present; every test
window is neutral-or-break only. **A regime-stratified model literally
cannot have produced the reported EMOS-vs-BMA heavy-bin alternation**
(EMOS-CSG winning at 24h/72h/96h, BMA at 48h/120h,
`docs/tier2-hierarchical-baseline-results.md`) via a monsoon-phase or
depression-presence signal, because that signal barely varies across the
test samples that alternation was measured on. (MJO phase does vary
day-to-day across the test window, but with only 3-4 test days per lead
and 8 possible phases, there's no way to attribute a domain-wide CRPS
difference to a specific phase from this few points without pure
overfitting.)

Also checked directly: within each lead's test window, the "heavy" bin's
cells are overwhelmingly dominated by a single real synoptic event (the
09-21/-22/-23-ish day each lead's test window contains -- e.g. 146 of the
72h lead's heavy-bin test cells come from 2020-09-23 alone, versus single
digits from the other 3 test days combined). The bin-level EMOS/BMA split
is best explained as **which combiner happens to fit this one dominant
event best at each lead's own re-sampling of it**, not a systematic regime
effect -- consistent with the test window's own lack of regime diversity
just established.

**Training data shows a real, but modest and noisy, correlation between
monsoon-active days and elevated heavy-rain cell counts.** Per-lead
training pools are small (14 weekly samples), but looking at real
per-day heavy-bin cell counts (GraphCast-mean classification, lead 24h)
against `src/weavr/regimes.py`'s labels for the same days:

| Day | Heavy-bin cells | Monsoon phase | MJO phase | Depression |
|---|---|---|---|---|
| 06-01 | 170 | neutral | 5 | **True** |
| 06-08 | 1 | neutral | 5 | False |
| 06-15 | 3 | neutral | 3 | False |
| 06-22 | 4 | neutral | 7 | False |
| 06-29 | 0 | neutral | 6 | False |
| 07-06 | 62 | neutral | 3 | **True** |
| 07-13 | 11 | neutral | 8 | False |
| 07-20 | 239 | neutral | 8 | False |
| 07-27 | 7 | neutral | 3 | False |
| 08-03 | 177 | neutral | 3 | False |
| 08-10 | 30 | neutral | 4 | False |
| 08-17 | 78 | **active** | 5 | False |
| 08-24 | 253 | **active** | 7 | False |
| 08-31 | 8 | neutral | 8 | **True** |

Both real "active" training days (08-17, 08-24) show above-median heavy-bin
counts (78, 253), consistent with Rajeevan et al.'s own framing of active
spells as widespread-heavy-rain events. But the two single highest-count
days (07-20: 239, 08-03: 177) are both "neutral" -- so monsoon phase alone
does not cleanly separate heavy from non-heavy training days here, only
weakly enriches for it. Depression-present days are similarly mixed (170
high, 62 medium, 8 low). MJO phase shows no visible pattern at all
(mjo=8 appears at counts 11, 239, and 8). With only 14 training days (2 of
them "active"), this is real but weak, noisy evidence -- enough to justify
trying the simplest possible regime-stratified fit, not enough to expect
a large, reliable improvement, and nowhere near enough to justify a
heavier learned-gating model without first seeing whether the simple
version even fits (rather than falling back for lack of samples).

**Conclusion, honestly stated**: there is a real, literature-grounded,
weakly-supported-by-this-project's-own-data reason to try conditioning on
monsoon active/break (the only covariate showing any real signal, and the
only one free to compute). There is no real evidence that a regime
covariate explains Phase 4's specific reported heavy-bin alternation --
that pattern is better explained by a single dominant event being
resampled differently at each lead, and the test window's own lack of
regime diversity rules out the regime hypothesis for it specifically. MJO
phase and monsoon-depression presence, also built in step 2, show no
comparably clean pattern in this small a sample and are not conditioned on
in this step's fit (see `src/weavr/regime_weighting.py`'s own docstring).

## Step 3.2/3.3: the simple regime-stratified fit, and the heavier-model decision

`src/weavr/regime_weighting.py`'s `fit_regime_weights` reuses
`weavr.weighting.fit_region_weights`'s exact OLS-clip-renormalize-fallback
machinery, conditioned on `weavr.regimes.classify_monsoon_active_break`'s
per-day label instead of a per-gridpoint region (see that module's own
docstring for why the pooling axis differs -- region is spatial, regime is
temporal). Tested in `tests/test_regime_weighting.py` against a synthetic
case with a source that is only accurate in one regime category (recovers
the expected per-category weight split) and the documented sparse-category
fallback (mirroring `tests/test_weighting.py`'s own pattern).

Given only 2 of 14 real training days are ever "active" at any lead, the
real `fit_regime_weights` result against this project's own data sits
right at `weavr.weighting.MIN_TRAIN_SAMPLES = 2` for the "active" category
-- a real fit may be attempted rather than falling back outright, but from
the smallest usable sample this module's own threshold allows, and
whichever it does is reported plainly in step 4's own results, not assumed
here.

Per this file's own step 3.1 assessment (weak, noisy evidence for the one
covariate with any signal; zero evidence the specific reported alternation
is regime-driven; a training set too small to support anything heavier),
whether to build the optional GBM/ViT-style gating model was put to the
user via `AskUserQuestion` rather than assumed -- see this file's own
"Heavier-model decision" section below for the real answer.

## Heavier-model decision

Put to the user via `AskUserQuestion`, informed by this file's own step 3.1
assessment above. **Decision: do not build the heavier model.** The
evidence gathered here doesn't support it -- the test window shows zero
regime diversity (ruling out a regime explanation for Phase 4's specific
reported alternation), training data shows only a weak, noisy correlation
for the one covariate with any signal at all, and only 2 of 14 training
days are ever "active" per lead, far too few to fit anything heavier than
a two-category weight split without overfitting. Issue #7's own condition
("only if Phase 4 plateaus" and only with genuine headroom) is not met.
Step 4 therefore evaluates only the simple regime-stratified reweighting
(`src/weavr/regime_weighting.py`) against Phase 4's real Tier 2 numbers.

## Step 4: the pre-stated go/no-go criterion and the real result

Produced by
[`scripts/run_tier3_regime_conditioned_baseline.py`](../scripts/run_tier3_regime_conditioned_baseline.py),
recomputing Phase 4's own EMOS-CSG/BMA fits and scores on the identical
aligned samples and `seasonal_block_split` the regime-conditioned blend
uses (per Tier 1/Tier 2's own precedent -- see that script's docstring for
why Phase 4's own recorded numbers are not read directly). Full numbers in
[`results/tier3_regime_conditioned_baseline.csv`](../results/tier3_regime_conditioned_baseline.csv)
(domain-wide, per lead) and
[`results/tier3_regime_conditioned_baseline_by_bin.csv`](../results/tier3_regime_conditioned_baseline_by_bin.csv)
(per bin).

**Criterion, decided before running the script** (stated in that script's
own docstring, not adjusted afterward): the regime-conditioned model
"beats Phase 4" if its domain-wide CRPS is the lowest of the four methods
at a majority of leads (>= 3 of 5) -- the same "count leads won" framing
every prior tier in this project has used for its own headline comparison.

**A real, checked scoring note**: the regime-conditioned model
(`weavr.regime_weighting`) is structurally a Tier-1-shaped model -- a
deterministic OLS blend of `graphcast`/`hres`/`ifs_ens_mean`, conditioned
on monsoon phase instead of region -- not a fitted predictive distribution
like EMOS-CSG or BMA. Its CRPS is scored via the real identity "CRPS of a
point-mass distribution equals absolute error," the correct proper score
for the distribution it actually outputs, not an approximation. This means
part of any gap below is structural (a point forecast has no way to hedge
uncertain days the way a fitted distribution can), not solely about
whether monsoon-phase conditioning itself helped -- named plainly so the
result isn't misread as "regime conditioning failed" when part of it is
"a point-forecast blend was compared to two properly probabilistic
models."

### Domain-wide CRPS (mm), all four methods, same split

| Lead (h) | Regime-conditioned | EMOS-graphcast | EMOS-ifs_ens | BMA | Winner |
|---|---|---|---|---|---|
| 24  | 5.95 | 3.76 | 3.76 | **3.72** | BMA |
| 48  | 6.36 | **3.74** | 3.83 | 3.80 | EMOS-graphcast |
| 72  | 6.18 | **3.64** | 3.90 | 3.83 | EMOS-graphcast |
| 96  | 6.80 | **4.41** | 4.64 | 4.64 | EMOS-graphcast |
| 120 | 7.22 | 4.46 | 4.55 | **4.45** | BMA |

**The regime-conditioned model wins 0 of 5 leads.** By the pre-stated
criterion (needs >= 3 of 5), this is a clean, unambiguous **NO GO** --
not a close call decided by which metric was chosen after the fact.

### Per-bin: the same result holds in every stratum, not just on aggregate

Pooled CRPS by lead and bin (dry/light/heavy -- `very_heavy`/
`extremely_heavy` are all-fallback or near-empty at every lead, as Phase 4
step 6 already found):

| Lead (h) | Bin | Regime-conditioned | EMOS-graphcast | EMOS-ifs_ens | BMA |
|---|---|---|---|---|---|
| 24  | dry   | 3.16 | 2.01 | 2.05 | 1.96 |
| 24  | light | 13.48 | 8.49 | 8.38 | 8.41 |
| 24  | heavy | 40.70 | 23.02 | 23.96 | 46.49 |
| 48  | dry   | 3.02 | 1.78 | 1.85 | 1.80 |
| 48  | light | 14.29 | 8.56 | 8.73 | 8.80 |
| 48  | heavy | 62.88 | 23.29 | 21.91 | 19.33 |
| 72  | dry   | 3.11 | 1.75 | 1.90 | 1.79 |
| 72  | light | 12.78 | 7.71 | 8.19 | 7.99 |
| 72  | heavy | 46.50 | 26.82 | 30.10 | 45.80 |
| 96  | dry   | 4.01 | 2.46 | 2.60 | 2.51 |
| 96  | light | 12.06 | 8.07 | 8.43 | 8.40 |
| 96  | heavy | 38.00 | 27.05 | 30.99 | 61.01 |
| 120 | dry   | 4.14 | 2.62 | 2.63 | 2.60 |
| 120 | light | 12.61 | 7.69 | 7.92 | 7.71 |
| 120 | heavy | 33.74 | 15.11 | 15.79 | 13.97 |

The regime-conditioned model does not win a single (lead, bin) cell above
-- consistent with the domain-wide result, not an artifact of aggregation.
One real exception, noted for completeness but excluded from the verdict:
at 96h, the `very_heavy` bin (a single test cell) shows the regime model
at 26.3mm against EMOS-graphcast/EMOS-ifs_ens's 111.5mm -- but that EMOS
cell `is_fallback=True` (a point-mass-at-zero prediction against a huge
real observation, per `docs/tier2-hierarchical-baseline-results.md`'s own
description of this exact cell), so this is a single degenerate-fallback
comparison, not evidence the regime model is broadly competitive at
extreme rain -- the same reasoning Phase 4 step 6 already used to exclude
an analogous all-fallback cell from its own per-bin table.

### Honest conclusion

**No go.** The regime-conditioned model does not beat Phase 4's real
EMOS-CSG/BMA combiners by the criterion stated before this script was run,
at any lead, in any bin. Per this file's own established framing (Phase 3's
regional blend not winning everywhere; Phase 4's two combiners splitting
which they win), this is reported as this phase's real, valid outcome, not
adjusted or re-scoped to look better. Contributing factors, both real and
worth separating for anyone extending this work: (1) step 3.1 already found
only weak, noisy evidence that monsoon phase explains this project's own
error pattern, so a large win was never expected; (2) the model itself is
a deterministic point-forecast blend, structurally at a CRPS disadvantage
against two models that fit an actual predictive distribution -- the same
"same small sample size, worse here" caveat every prior tier has
stated applies doubly, since regime-conditioning can only shrink an
already-small training pool (14 samples, 2 "active") further. This is a
"no go on today's data," not proof monsoon-phase conditioning can never
help -- a different, honest conclusion this file is careful to keep
separate, per this project's own established practice.

Issue #7's own go/no-go criterion has been applied, honestly, to a real fit
against real data: Phase 5's regime-conditioned model is not adopted.

---

## v2 update (2026-10-01)

**This file's numbers above are v1 and are now superseded.** They were measured
on the **weekly, single-season 2020** evidence base: `n_train=14`, `n_test=3-4`,
`split=seasonal_block_split`, test dates 2020-09-07/14/21/28, and no confidence
intervals were computable. They are kept above as a labelled record.

Step 07 regenerated every result on the **two-season daily LOYO** base
(2018 + 2020, 122 train days per fold, 236-244 test days per lead, paired block
bootstrap CIs). Read them in
[`docs/v2-evidence-base-results.md`](v2-evidence-base-results.md).

Do not set the v1 and v2 columns side by side without stating the cadence
difference: v1 scored four late-September days on a weekly split, v2 scores a
whole season daily. Every RMSE in this file improved or worsened for sampling
reasons before any method changed.

Headline v2 findings: **H1 passes** (the blend beats the best single member on
RMSE at 3 of 5 leads and on CRPS at 5 of 5, every interval excluding zero),
**H2 fails** (the multi-source combiner does not beat the best single-source
EMOS on CRPS at any lead), and **H3 is not yet testable** (its Brier leg needs
exceedance probabilities that Tier 2's per-day files do not carry).
