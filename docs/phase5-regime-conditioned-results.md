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
