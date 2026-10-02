# Scorecard, significance and the honest baselines

Step 04 of the improvement plan. This document describes the machinery that
turns WEAVR's numbers into claims with error bars, and reports what that
machinery says about the **existing** evidence base.

The headline result is uncomfortable and is the point of running this now:

> **On today's weekly store, not one of the 1,691 method comparisons in
> `results/scorecard.csv` has a computable confidence interval.** With 3–4
> test days per lead there is not enough data to resample. Every comparison
> WEAVR has published to date — every "Tier 1 beats Tier 0", every
> "Tier 2 beats Tier 0" — rests on a sample too small to put an interval
> around.

Nothing here is a new forecast claim. It is a measurement of how much the
old evidence base could ever have supported, and the machinery step 07 will
re-run on daily, two-season data where the answers will mean something.

---

## 1. What was built

| Module | What it provides |
|---|---|
| [`weavr.significance`](../src/weavr/significance.py) | Moving-block bootstrap, paired difference CIs, Diebold–Mariano, lag-1 autocorrelation |
| [`weavr.verify`](../src/weavr/verify.py) (additions) | SEDI, threshold-weighted CRPS, PIT (ensemble and CSGD), reliability tables, BSS/CRPSS, raw contingency counts, a memory-safe CRPS |
| [`weavr.climatology`](../src/weavr/climatology.py) | Climatological ensembles and exceedance probabilities, with the test year excluded |
| [`weavr.score_io`](../src/weavr/score_io.py) | Per-day domain-wide scores, written by every scoring script |
| [`scripts/run_scorecard.py`](../scripts/run_scorecard.py) | `results/scorecard.csv`, `results/preregistration_verdicts.csv`, `results/block_length_diagnostic.csv` |

### Three decisions that determine whether any of it is trustworthy

**Ratios are never averaged; counts are stored.** The CSI of a week is not
the mean of its daily CSIs, because a ratio of sums is not the sum of
ratios. `weavr.score_io` therefore records hits / misses / false alarms /
correct negatives per day, and the scorecard rebuilds CSI and SEDI *inside*
each bootstrap replicate. The same applies to RMSE: per-day files store
**MSE**, and `paired_difference_ci(aggregate="rmse")` takes the square root
after averaging. Averaging daily RMSEs is biased low by Jensen's inequality.

**Days are resampled in blocks.** Daily forecast errors are autocorrelated —
a monsoon spell is wet for a week at a time — so an i.i.d. bootstrap would
produce intervals that are far too narrow and make everything look
significant. `docs/preregistration.md` fixes the block length at 7 days.

**The bootstrap refuses to invent an interval it cannot have.** See §3; this
is the single most important thing in this document.

---

## 2. The metrics, and why these ones

Beyond the Phase 1 set, the research brief's §3.5–3.6 recommends these and
`docs/preregistration.md` depends on them:

- **SEDI** (Ferro & Stephenson 2011). CSI and ETS both degenerate towards 0
  as an event gets rarer, so they cannot distinguish a skilful rare-event
  forecast from a useless one — exactly the regime WEAVR cares about. SEDI
  is base-rate independent. It returns **NaN** when the hit rate or false
  alarm rate is 0 or 1, which is common at high thresholds and is reported
  as "not estimable", never as zero skill.
  *Note:* SEDI needs the false alarm **rate** (`FA / (FA + CN)`), not the
  false alarm **ratio** that `contingency_scores` reports as `far`. They
  have different denominators, and substituting one silently produces a
  plausible wrong number. `tests/test_verify_significance_additions.py`
  checks SEDI against hand-computed counts for this reason.
- **Threshold-weighted CRPS** at 64.5 mm (Allen et al. 2023), via the
  chaining identity `twCRPS(F, y; t) = CRPS(max(members, t), max(y, t))`.
  Plain CRPS is dominated by the many dry and light days, so a change that
  only affects heavy rainfall barely moves it. Verified against brute-force
  numerical integration of the weighted definition.
- **PIT / rank histograms**, with randomized tie-breaking. Essential for
  precipitation specifically: most cells are exactly 0 in both forecast and
  observation, and deterministic ranking would pile every dry day into one
  bin and make a perfectly calibrated forecast look broken. The CSGD variant
  randomizes over the point mass at zero for the same reason.
- **Reliability tables**, which are what make an "X% chance of >115.6 mm"
  statement meaningful. Step 01's slide audit marked that example
  illustrative precisely because nothing had measured it.
- **Climatological references**, because a skill score needs one. A rare
  event gets a low Brier score from forecasting "never", so an absolute
  score proves nothing.

### A memory limit found, not predicted

`weavr.verify.crps` delegates to `properscoring`, which forms the full
`m × m` matrix of pairwise member differences. Fine for a 50-member IFS
ensemble; impossible for the **434-member** climatological reference, which
needs roughly **83 GB** over the 129×135 grid. Measured: 200 members over
2,000 cells already peaks at 2.0 GB, and the first attempt to score
climatology was killed by the OS.

`crps_large_ensemble` computes the same number from the sorted-ensemble
identity

```
CRPS = (1/m) Σ|xᵢ − y| − (1/m²) Σ (2i − m + 1) x₍ᵢ₎     (x sorted, i = 0…m−1)
```

in `O(m log m)` time and `O(m)` memory. It is an exact algebraic
rearrangement, not an approximation, and the tests check it against `crps`
itself to floating-point precision (agreement ~1e-15) across ensemble sizes.
The 434-member case then runs in 0.12 s at 0.4 GB.

---

## 3. The result that matters: no interval is computable

A moving block of length `L` over `n` days has `n − L + 1` possible start
positions. With **n = 3–4 test days** and the registered **L = 7**, there is
exactly **one** — so every bootstrap replicate is the original series, and
the interval has *zero width*.

A zero-width interval is not a narrow interval. It is no interval at all,
and it excludes 0 for **any** non-zero difference. A first run of this
scorecard reported **965 of 1,691 comparisons (57%) as significant** on
three days of data, which is nonsense, and is exactly the failure mode this
step exists to prevent.

`weavr.significance.bootstrap_is_degenerate` now detects this case, and such
comparisons are written with NaN bounds, `degenerate_bootstrap=True` and
`significant=False`:

```
Distinct days per lead: {24: 4, 48: 4, 72: 4, 96: 3, 120: 3}

Wrote results/scorecard.csv (1691 comparisons)
Significant (CI excludes 0): 0 of 1691 (0.0%)
No interval computable (too few days to resample): 1691 of 1691 (100.0%)
```

**Every comparison. All five leads. All twelve methods.**

### The machinery does work — checked separately

To show that 0-of-1691 reflects the data and not a broken implementation,
the same scorecard was run with a deliberately invalid 1-day block (which
ignores autocorrelation entirely and is *not* the registered setting). Real
intervals appear — and even then, most comparisons are still not
significant. RMSE against climatology, mm:

| Lead | Method | Method | Climatology | Diff | 95% CI | Significant |
|---|---|---|---|---|---|---|
| 24 h | graphcast | 11.49 | 11.81 | −0.32 | (−1.23, +0.56) | no |
| 24 h | tier0 | 11.66 | 11.81 | −0.15 | (−1.46, +1.36) | no |
| 24 h | tier2_emos_graphcast | 10.86 | 11.81 | −0.95 | (−1.41, −0.59) | **yes** |
| 48 h | tier0 | 13.54 | 12.52 | **+1.01** | (−1.20, +3.76) | no |
| 72 h | graphcast | 11.80 | 12.52 | −0.71 | (−1.00, −0.29) | **yes** |
| 72 h | tier0 | 13.38 | 12.52 | **+0.86** | (−0.34, +1.77) | no |
| 96 h | tier2_emos_graphcast | 12.95 | 14.77 | −1.82 | (−3.56, +0.03) | no |
| 120 h | tier0 | 15.30 | 15.33 | −0.03 | (−0.83, +1.51) | no |

Two things are visible even under this over-generous setting, and both are
consistent with step 01:

- **Tier 0's equal-weight mean has a *higher* RMSE than climatology at 48 h
  and 72 h.** Not significantly so — but a blend that does not clearly beat
  "what the climate would have said" is not yet earning its complexity.
- The only comparisons reaching significance are **single-source
  calibrations** (EMOS on GraphCast) and **raw GraphCast**, never a
  multi-source blend. That is step 01's finding again, now with intervals
  attached.

These numbers are **exploratory** under `docs/preregistration.md`'s rules:
the 1-day block is not the registered procedure, and nothing here may appear
on a slide as a finding.

### Block length

`results/block_length_diagnostic.csv` reports the lag-1 autocorrelation of
each method's daily MSE, which is what the pre-registration says to use if
the 7-day default needs revising. Today it is uninformative — the mean is
−0.30 to −0.12 across leads, computed from 3–4 points, where a lag-1
estimate has essentially no meaning. Step 07 re-measures it on ~240 days;
if it disagrees with 7 days, `docs/preregistration.md` is amended in its own
commit **before** any verdict is computed.

---

## 4. Pre-registration verdicts

`results/preregistration_verdicts.csv`, judged against
[`docs/preregistration.md`](preregistration.md). All twelve entries (H1 is
two verdicts) are **NOT_YET_TESTED**, each naming what is missing:

| Claim | Status | Why |
|---|---|---|
| H1-RMSE, H1-CRPS | NOT_YET_TESTED | The headline WEAVR configuration must be declared in step 07 before verdicts are computed |
| H2 | NOT_YET_TESTED | The single-source EMOS comparator must be selected on the **training** fold; today's per-day scores are test-only |
| H3 | NOT_YET_TESTED | Needs a blend with calibrated exceedance probabilities (step 07) |
| H4, H5 | NOT_YET_TESTED | Step 08 (NEPS-G) |
| H6 | NOT_YET_TESTED | Step 09 |
| H7 | NOT_YET_TESTED | Steps 11–12 |
| H8 | NOT_YET_TESTED | Step 14 |
| H9 | NOT_YET_TESTED | Step 19; report-either-way, no pass/fail defined |
| H10 | NOT_YET_TESTED | Step 13 |
| H11 | NOT_YET_TESTED | Step 21 |

The verdict writer refuses to run if a claim it implements has no section in
`docs/preregistration.md` — the doc is the source of truth, and an
unregistered claim must not be able to masquerade as a registered one.

The status vocabulary distinguishes two things that are easy to conflate:

- **FAIL** — the improvement holds at fewer than 3 of 5 leads. The direction
  itself fails. A real negative result.
- **INSUFFICIENT_DATA** — the direction holds at ≥ 3 leads, but the
  intervals do not exclude 0. **Unproven is not disproven**, and reporting
  it as FAIL would misrepresent a sample-size limit as evidence against the
  claim.

---

## 5. Per-day scores

Every scoring script now also writes `results/per_day/<method>__lead<h>.csv`
— one row per IMD day, scored over the whole domain. Twelve methods × five
leads = 55 files today.

Methods: `graphcast`, `hres`, `ifs_ens_mean`,
`best_single_member_on_train`, `climatology`, `tier0`, `tier1_regional`,
`tier2_emos_graphcast`, `tier2_emos_ifs_ens`, `tier2_bma`,
`tier3_regime_conditioned`, `phase2_lagged_graphcast`.

Each method is written by exactly one script. Tier 2 and Tier 3 both
recompute Tier 0, Tier 1 and the EMOS/BMA combiners (to score them on
identical samples), but only the owning script writes them — two files under
one method name would leave the scorecard silently using whichever script
ran last.

**Every existing aggregated CSV is byte-for-byte unchanged.** All six
scripts were re-run and diffed; this change is purely additive.

---

## 6. Reproducing

```bash
python scripts/run_tier0_baseline.py
python scripts/run_single_source_baseline.py     # also writes climatology
python scripts/run_tier1_regional_baseline.py
python scripts/run_tier2_hierarchical_baseline.py
python scripts/run_tier3_regime_conditioned_baseline.py
python scripts/run_phase2_ensemble_baseline.py
python scripts/run_scorecard.py
```

## 7. References

- Ferro, C. A. T. & Stephenson, D. B. (2011), "Extremal dependence indices",
  *Weather and Forecasting* 26. — SEDI.
- Allen, S., Ginsbourger, D. & Ziegel, J. (2023), "Evaluating forecasts for
  high-impact events using transformed kernel scores". — threshold-weighted
  CRPS and the chaining identity.
- Künsch, H. R. (1989), "The jackknife and the bootstrap for general
  stationary observations", *Annals of Statistics* 17. — moving-block
  bootstrap.
- Diebold, F. X. & Mariano, R. S. (1995), "Comparing predictive accuracy",
  *JBES* 13; Newey, W. K. & West, K. D. (1987) for the HAC variance.
- Hersbach, H. (2000), "Decomposition of the CRPS for ensemble prediction
  systems", *Weather and Forecasting* 15. — the CRPS decomposition behind
  `crps_large_ensemble`.

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
