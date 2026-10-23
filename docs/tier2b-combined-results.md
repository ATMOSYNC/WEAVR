# Tier 2b Combined Combiners (EMOS-CSG + BMA Stacking)

## 1. Overview and Motivation

Phase 4 evaluated two advanced probabilistic combiners:
- **EMOS-CSG** (`src/weavr/emos.py`): Parametric Censored Shifted Gamma Distribution regression per rain-intensity bin. Excels at discrimination and sharp threshold forecasts.
- **Hierarchical BMA** (`src/weavr/bma.py`): Mixture modeling with point-mass at zero and power-transformed Gamma components. Excels at calibration and reliability across spread regimes.

As demonstrated by **Javanshiri (2021)** and **Ji (2025)**, parametric EMOS and mixture BMA exhibit complementary trade-offs: EMOS provides superior discrimination for heavy events, whereas BMA produces better-calibrated probabilities. Slide 2 previously asserted that WEAVR *"uses the strongest combiner per bin"*, but no such combined selector existed in the codebase. Step 13 closes this gap with two principled combination methods implemented in `src/weavr/stacking.py`.

---

## 2. Combination Methods

### A. Per-Bin x Lead Selection
Selects the combiner with the lowest mean CRPS on the training fold for each `(rain_bin, lead)` stratum:
$$\text{Combiner}^*(b, l) = \arg\min_{c \in \{\text{EMOS-CSG}, \text{BMA}\}} \overline{\text{CRPS}}_{\text{train}}(b, l, c)$$

- **Tie-break rule**: If CRPS values match within $10^{-9}$, prefers `emos_csg` to favor parametric efficiency.
- **Sparse bin fallback**: If a bin has insufficient training cases (e.g. extreme precipitation), falls back to the domain-wide pooled-best combiner.

### B. Quantile Averaging (Vincentization)
Rather than averaging probabilities (linear pooling), which artificially flattens sharp distributions and leads to over-dispersion, **Lichtendahl, Grushka-Cockayne & Winkler (2013)** demonstrated that **quantile averaging (Vincentization)** preserves distribution shape, unimodality, and sharpness:
$$q_{\text{avg}}(\tau) = (1 - w) \cdot q_{\text{EMOS}}(\tau) + w \cdot q_{\text{BMA}}(\tau), \quad \tau \in (0, 1)$$

- **Monotonicity**: Because $q_{\text{EMOS}}(\tau)$ and $q_{\text{BMA}}(\tau)$ are monotone non-decreasing in $\tau$ and $w \in [0, 1]$, $q_{\text{avg}}(\tau)$ is monotone non-decreasing by construction.
- **CSGD Quantiles**: Inverted analytically from the CSGD survival function, accounting for the point mass at zero:
  $$p_0 = F_X(-\delta / \theta; \kappa), \quad q_{\text{EMOS}}(\tau) = \begin{cases} 0 & \text{if } \tau \le p_0 \\ \max(0, \delta + \theta \cdot F_X^{-1}(\tau; \kappa)) & \text{if } \tau > p_0 \end{cases}$$
- **BMA Quantiles**: Exact, by inverting the closed-form mixture CDF (`weavr.bma.bma_predictive_quantiles`): interpolated on a value grid reaching 1e5 mm, and solved by bisection for any level the grid does not reach. An earlier version read them from 500 Monte Carlo draws, which understated the upper quantiles by enough to matter at H10's margins.

### C. Linear Pool (Reference Baseline)
Standard mixture of predictive probabilities:
$$P_{\text{pool}}(Y > t) = (1 - w) P_{\text{EMOS}}(Y > t) + w P_{\text{BMA}}(Y > t)$$

---

## 3. Quantile-Based CRPS Approximation

To evaluate quantile-averaged predictive distributions without closed forms, `crps_from_quantiles` integrates the pinball loss across 99 quantile levels:
$$\text{CRPS}(F, y) = 2 \int_0^1 (y - q(\tau)) \cdot (\tau - \mathbb{I}(y < q(\tau))) \, d\tau$$
Verified against closed-form `csgd_crps` with $< 1.5\%$ relative error across all test conditions in `tests/test_stacking.py`.

---

## 3a. Measured Results

Two-season daily LOYO (2018 JJAS / 2020 JJAS), leads 24–120 h, Tier 1 blend
scope, against the IMD observed grid. Pooled scores are cell-weighted over both
folds and all five leads. Reproduce with
`scripts/run_tier2b_combined.py --fit-weight` (`--fit-weight` adds the
train-fitted `quantile_avg` arm alongside the fixed w=0.5 one; without it both
arms use 0.5). Results: `results/tier2b_combined.csv`, `_paired.csv`,
`_train.csv` and `results/per_day/tier2b_*`.

| arm | pooled CRPS (mm) | PIT mean | role |
|---|---|---|---|
| `linear_pool` | **4.5412** | 0.492 | comparison only |
| `quantile_avg_fixed` (w=0.5) | 4.5420 | 0.492 | combination |
| `quantile_avg` (w fitted on train) | 4.5484 | 0.493 | combination |
| `per_bin` | 4.5591 | 0.498 | combination |
| `bma` | 4.5645 | 0.498 | parent |
| `emos_csg` | 4.5657 | 0.499 | parent |

Brier score by IMD threshold (cell-weighted, lower is better):

| arm | 7.5 mm | 35.5 mm | 64.5 mm | 115.6 mm | 204.5 mm |
|---|---|---|---|---|---|
| `linear_pool` | 0.14308 | **0.03879** | **0.01240** | 0.00259 | 0.00027 |
| `quantile_avg_fixed` | 0.14309 | 0.03880 | 0.01243 | 0.00259 | 0.00027 |
| `quantile_avg` | **0.14307** | 0.03891 | 0.01249 | 0.00260 | 0.00027 |
| `per_bin` | 0.14341 | 0.03899 | 0.01248 | 0.00261 | 0.00028 |
| `bma` | 0.14348 | 0.03905 | 0.01253 | 0.00261 | 0.00028 |
| `emos_csg` | 0.14401 | 0.03899 | 0.01243 | 0.00259 | 0.00027 |

The Brier spread across arms is small -- the fifth decimal at 204.5 mm, the
third at 7.5 mm -- so Brier does not separate these methods. CRPS does, which
says the trade-off is about the *shape* of the predictive distribution rather
than threshold decisions.

**Every arm is calibrated on average: PIT means 0.49–0.50.** An earlier version
of this file reported 0.61–0.63 for every arm and read it as a shared
over-forecast upstream of the combination rule. It was the PIT itself: a dry
observation was scored as `F(0) = p0`, the top of the point mass at zero, when
every level in `[0, p0]` is equally consistent with it. A forecast that is
calibrated by construction scored 0.638 under that rule. `pit_values` now
scores the point mass as the randomised PIT's expectation, and a regression
test pins it (`tests/test_run_tier2b_combined.py`). Only `pit_mean` changed:
every CRPS, Brier and paired difference is byte-identical between the two
codes.

**What changed against the previous run.** Both parents and every combination
moved because BMA did. The previous run used BMA quantiles from a library with
two defects, fixed on #102: a variance floor compounded by an independent
`theta` clip, and Monte Carlo quantiles. With them fixed, BMA's pooled CRPS
fell from 4.5918 to 4.5645 mm and it now edges EMOS-CSG (4.5657), so the two
parents are close -- which is exactly the regime where combining them can
help.

---

## 4. Pre-registered Hypothesis H10

- **Null Hypothesis**: The combined combiner (quantile averaging or per-bin selection) does not improve upon both individual constituent combiners.
- **Pass Rule**: Significantly lower CRPS than both standalone EMOS-CSG and BMA at $\ge 3$ of 5 leads, with the 95% paired block-bootstrap CI strictly excluding 0, for the **primary arm nominated on train-fold CRPS**.

### 4.1 H10 verdict: **FAIL** (2 of 5 leads) -- but quantile averaging beats both parents at 5 of 5

The primary arm is nominated on **train**-fold CRPS, never on test. It was
`per_bin` at 24, 48 and 72 h and `quantile_avg` at 96 and 120 h. Paired daily
CRPS differences, negative meaning the combination is better, 95% paired
moving-block bootstrap over both seasons (236–244 test days per lead):

| lead | nominated | vs `emos_csg` | verdict | vs `bma` | verdict |
|---|---|---|---|---|---|
| 24 h | `per_bin` | −0.00428 CI[−0.02123, +0.01340] | fail | −0.01040 CI[−0.02098, −0.00296] | **pass** |
| 48 h | `per_bin` | −0.00178 CI[−0.01845, +0.01375] | fail | −0.00699 CI[−0.01496, −0.00183] | **pass** |
| 72 h | `per_bin` | −0.00890 CI[−0.03058, +0.01223] | fail | −0.00701 CI[−0.01499, −0.00178] | **pass** |
| 96 h | `quantile_avg` | −0.02479 CI[−0.04439, −0.00717] | **pass** | −0.01514 CI[−0.02396, −0.00833] | **pass** |
| 120 h | `quantile_avg` | −0.02075 CI[−0.03944, −0.00597] | **pass** | −0.01502 CI[−0.02187, −0.00842] | **pass** |

**2 of 5 leads beat both parents, short of the pre-registered 3. H10 fails as
registered.** The failure is entirely the nomination: `per_bin` has the lowest
*train* CRPS at the three short leads, by 0.001–0.007 mm pooled over the folds,
and then never separates from EMOS-CSG on test.

Every arm is reported separately, so the verdict on the nominated arm cannot
hide a method that did beat both parents:

| arm | leads where it beat **both** parents |
|---|---|
| `per_bin` | **0 / 5** |
| `quantile_avg` (fitted) | **4 / 5** |
| `quantile_avg_fixed` (w=0.5) | **5 / 5** |
| `linear_pool` | **5 / 5** (comparison only) |

`quantile_avg_fixed` against each parent, every interval excluding 0:

| lead | vs `emos_csg` | vs `bma` |
|---|---|---|
| 24 h | −0.0180 CI[−0.0279, −0.0086] | −0.0241 CI[−0.0387, −0.0124] |
| 48 h | −0.0184 CI[−0.0270, −0.0107] | −0.0236 CI[−0.0352, −0.0130] |
| 72 h | −0.0254 CI[−0.0362, −0.0154] | −0.0235 CI[−0.0390, −0.0102] |
| 96 h | −0.0303 CI[−0.0430, −0.0201] | −0.0206 CI[−0.0384, −0.0060] |
| 120 h | −0.0264 CI[−0.0389, −0.0164] | −0.0207 CI[−0.0333, −0.0083] |

**The previous verdict was PASS (3 of 5), and it does not survive.** It rested
on BMA quantiles from the defective library, under which the combinations
beat BMA by 0.02–0.07 mm and `quantile_avg` was the nominee at four leads. With
BMA fixed, every arm's margin over BMA shrinks and the nominations change; the
result above is the one the corrected parents support.

**Nomination is now pooled over folds.** At 48 h and 96 h the two folds' train
CRPS prefer different arms, and the runner used to take whichever fold came
first. `nominate_by_lead` now averages each candidate's train CRPS over the
folds (ties to `quantile_avg`; `per_bin` ineligible where it collapsed onto one
parent). On this run it nominates exactly the same arms, so the verdict does not
depend on the change -- but it no longer depends on fold order either.

---

## 5. What the verdict does and does not license

**It does not license H10.** The pre-registered claim fails, and it is not
re-litigated here by swapping in the arm that did best on test. Choosing the
fixed-weight arm *after* seeing these results is exactly what nominating on
train was meant to prevent.

**It does license a narrower, honest statement:** averaging the two parents'
quantile functions with the pre-registered fixed weight of 0.5 beats both
parents at every lead, by 0.018–0.030 mm of daily CRPS, every interval
excluding 0. That arm was declared before the run (w=0.5 is the
pre-registered weight), so it is a real result -- but it is not the H10
primary, and a follow-up claim built on it needs its own pre-registration.

**It does not license the claim Step 13 was partly built to rescue.** Slide 2
says WEAVR "uses the strongest combiner per bin". That is `per_bin`, and
`per_bin` beats both parents at **0 of 5** leads. Selecting a winner per rain
bin on train CRPS and applying it unconditionally throws away the hedge exactly
where it is needed. Slide 2 should carry no per-bin claim.

**Three qualifications:**

1. **Fitting the Vincentization weight still does not help.** The train-fitted
   weight (0.55–0.80 by lead and fold) beats both parents at 4 of 5 leads; the
   fixed 0.5 at 5 of 5. On this evidence the extra freedom costs a lead, so
   w=0.5 stays the runner's default.
2. **The margin is small.** About 0.02–0.03 mm of daily CRPS against a ~4.5 mm
   total: real and significant under the paired test, not the magnitude of a
   method change.
3. **Day alignment (#106).** Every forecast here is paired with the IMD day
   before the one it covers, as in every runner today. Both parents and every
   combination pay that penalty on the same days, so the paired differences
   are like for like, but the absolute CRPS is understated and H10 should be
   re-decided once #106 is resolved.

---

## 6. Open problems this step did not fix

- **The CRPS is a 99-level quantile approximation** for all six arms,
  deliberately uniform so that no arm is advantaged by a better estimator.
- **The linear pool's inverse is interpolated**, not closed-form (§2 C). It is
  kept comparison-only for that reason, and a test pins it inside the parents'
  CRPS range so an inversion failure cannot pass as a pooling result.

**Resolved since the first version of this file:**

- The Tier 2 `heavy` x NE1/SI "variance blow-up" was a defect in the BMA parent
  -- a negative-slope variance regression floored at `_TINY`, compounded by an
  independent clip of `theta` that moved the component mean to ~1850 mm. It is
  fixed on #102 (merged into this branch), and BMA quantiles are now exact
  (§2 B) rather than 500-draw Monte Carlo.
- The "universal PIT bias" was the dry-observation PIT bug (§3a).

---

## 7. Bugs found by reading the numbers, not the tests

This step needed four runs before this one. Every run exited 0, passed the test suite, and wrote
finite numbers; each was wrong in a way only inspection caught. They are
recorded because the pattern is the finding, not the individual slips.

| Bug | Symptom | Why tests missed it |
|---|---|---|
| `ci_low > 0` used as the improvement test | A combination winning by `d=-0.047`, CI `[-0.074,-0.022]`, was recorded as a loss | The helper returned a plausible boolean |
| Train table keyed `lead=0` while the test lookup asked for the real lead | `per_bin` silently collapsed to one parent everywhere, showing `0.000000` differences | A collapse looks exactly like a clean result |
| `sum(means) / sum(counts)` used as a mean | `per_bin` train CRPS read 7e-5 mm against `quantile_avg`'s 4.8 mm -- 56,000x -- so `<=` always picked `per_bin`, and H10 was decided on the weakest arm at 4 of 5 leads | Both values were finite and plausible-looking |
| Raw rainfall passed where a binary indicator belonged | Every `brier_*` column read ~281 mm^2 instead of a value in [0,1] | Nothing bounded a Brier score |

Each is now covered by a test built from the numbers that exposed it, and the
third has a runtime guard (`comparable_train_crps`) that aborts the run rather
than nominating an arm from incomparable quantities. The general lesson: the
suite checks that the code runs and that outputs are well-formed, and none of
that catches a computation that is well-formed and means the wrong thing.
Sanity bounds on internal quantities -- a Brier score in [0,1], two CRPS values
on a common scale -- are cheap and would have caught two of these immediately.
