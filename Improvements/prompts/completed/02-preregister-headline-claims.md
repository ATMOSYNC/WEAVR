# Step 02: Pre-register the headline claims

**Type**: design/decision prompt (a doc-only PR).
**Plan items**: A4 (pre-registration), plan §3 ("Accountable").
**Depends on**: step 01. The best-single-member baseline must be defined
before claims can reference it.

## Goal

Before any new data (steps 05–07) or new method (steps 08–13, 19) produces a
single number, write down each headline claim WEAVR hopes to make, and
exactly how it will be judged. Merge that as its own PR, so the git history
proves the criteria came before the results.

Phase 5 did this once (Tier 3's go/no-go was fixed before the run). This
step makes it the rule for every claim that reaches a slide.

## Why now

- Moving from 18 weekly samples to about 240 daily samples will change
  every number. Without fixed criteria, any improvement found afterwards is
  open to "you picked the metric that looked best".
- Pre-registration is WEAVR's strongest credibility asset in front of a
  scientific jury. A timestamped, merged criterion doc is proof no other team
  will have.
- It also forces one real decision now: which metric "beats the best single
  model" is judged on. F2 showed RMSE currently favours raw GraphCast. The
  research brief's §1.3 says RMSE favours smooth forecasts.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Route one genuine tradeoff to the user via AskUserQuestion.
   Question: for H1 ("the blend beats the best single model"), what is the
   primary metric?
     (a) RMSE only (the traditional deterministic view; currently favours
         raw GraphCast, plan section 2 F2)
     (b) CRPS only (the probabilistic view; the research brief's
         recommendation for blends)
     (c) both, reported separately, with H1 judged on each and never merged
         into one verdict
   Describe the tradeoff plainly; the brief's section 1.2-1.3 has the
   background. Recommend (c).

2. Write docs/preregistration.md. Each claim gets: an ID, the exact claim,
   the data (seasons, leads, sources), the split, the metric(s), the
   comparison, and the pass criterion.
   Shared procedure for every claim:
   - Split: leave-one-year-out over 2018 and 2020 (both folds, pooled
     daily scores).
   - CIs: 95% paired block bootstrap on daily domain-wide scores, 1,000
     resamples, block length 7 days unless step 07 measures a different
     autocorrelation length. If it does, update this doc in its own commit
     BEFORE the verdicts are computed.
   - Default pass rule: the improvement holds at >= 3 of 5 leads, AND the
     CI of the paired difference excludes 0 at those leads.
   Claims:
   - H1  A WEAVR blend beats the best single member (chosen on train) on
         the metric chosen in item 1.
   - H2  The multi-source probabilistic combiner (best of BMA / stacked)
         beats the best single-source EMOS on CRPS.
   - H3  WEAVR's P(>=64.5 mm) has a Brier skill score > 0 against IMD
         climatology (built excluding the test year), and a higher SEDI at
         64.5 mm than the best single member.
   - H4  At 24 h (2018/2019), adding NEPS-G to the blend improves CRPS over
         the same blend without it.
   - H5  At 24 h on identical days, WEAVR vs HEPPI's published NEPS-G EMOS
         (Angus et al. 2024) on CRPS. Report the result either way; claim a
         win only if the CI excludes 0.
   - H6  Independence-aware (shrinkage-covariance) weights beat OLS
         fit_region_weights on RMSE (step 09's go/no-go).
   - H7  Tail repair (quantile mapping and/or the extreme-value tail)
         improves threshold-weighted CRPS (t = 64.5 mm) or SEDI at
         115.6 mm, WITHOUT making the Brier score at 7.5 mm worse (steps
         11-12).
   - H8  District Orange/Red warnings from WEAVR have higher CSI than the
         same rule applied to the best single raw ensemble (step 14).
   - H9  Event replay (step 19): report the first lead at which WEAVR
         issues Orange and Red over the affected districts, next to each
         raw model. It is shown whatever the outcome; no pass/fail, but the
         event and districts are fixed in step 19 BEFORE its results are
         viewed.
   - H10 A stacked or per-bin-selected combiner beats both EMOS and BMA on
         CRPS (step 13).
   - H11 Online weights beat static Tier 1 weights on a sequential replay
         of the held-out season (step 21).

3. Add a rules section:
   - Only claims that PASS may appear on slides as findings. Everything
     else is "exploratory".
   - A failed claim is reported with the same prominence as a passed one,
     in the results doc of the step that tests it.
   - Criteria change only in a separate, dated commit that says why, and
     never after that claim's results have been computed.
   - The Tier 3 go/no-go (docs/phase5-regime-conditioned-results.md) is the
     precedent; link it.

4. Add a README.md paragraph linking the doc. Open one PR containing only
   this doc and the README change.
```

## Done when

`docs/preregistration.md` is merged to `main` before step 07 runs. It
must contain H1–H11 with exact metrics, splits and pass rules, and the
user's H1 metric decision.
