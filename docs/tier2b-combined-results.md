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
- **BMA Quantiles**: Extracted empirically from mixture Monte Carlo draws.

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

Two-season daily LOYO (2018JJAS / 2020JJAS), leads 24-120 h, Tier 1 blend
scope, against the IMD observed grid. Pooled CRPS is cell-weighted over both
folds and all five leads. Reproduce with `scripts/run_tier2b_combined.py`.

| arm | pooled CRPS (mm) | PIT mean | role |
|---|---|---|---|
| `linear_pool` | **4.5525** | 0.613 | comparison only |
| `quantile_avg_fixed` (w=0.5) | 4.5533 | **0.613** | combination |
| `quantile_avg` (w fitted on train) | 4.5570 | 0.613 | combination, nominated |
| `emos_csg` | 4.5657 | 0.627 | parent |
| `per_bin` | 4.5785 | 0.625 | combination |
| `bma` | 4.5918 | 0.624 | parent |

Brier score by IMD threshold (cell-weighted, lower is better):

| arm | 7.5 mm | 35.5 mm | 64.5 mm | 115.6 mm | 204.5 mm |
|---|---|---|---|---|---|
| `linear_pool` | **0.14314** | 0.03881 | **0.01241** | 0.00259 | 0.00028 |
| `quantile_avg_fixed` | 0.14315 | 0.03882 | 0.01244 | 0.00260 | 0.00028 |
| `quantile_avg` | **0.14313** | 0.03888 | 0.01247 | 0.00260 | 0.00028 |
| `emos_csg` | 0.14401 | 0.03899 | 0.01243 | 0.00259 | **0.00027** |
| `per_bin` | 0.14368 | 0.03905 | 0.01250 | 0.00261 | 0.00028 |
| `bma` | 0.14376 | 0.03912 | 0.01255 | 0.00262 | 0.00028 |

The Brier spread across arms is small -- in the fifth decimal place at 204.5 mm,
the third at 7.5 mm -- so Brier does not separate these methods here. CRPS does.
That is consistent with the trade-off being about the *shape* of the predictive
distribution rather than about threshold decisions.

**PIT means sit at 0.61-0.63 for every arm**, against 0.5 for a calibrated
forecast. All six over-forecast wetness, and they do so to a near-identical
degree, which points at something shared by both parents rather than at the
combination rule. This is unresolved and is the same family of problem as the
Tier 2 `heavy` x NE1/SI variance blow-up; see §6.

---

## 4. Pre-registered Hypothesis H10

- **Null Hypothesis**: The combined combiner (quantile averaging or per-bin selection) does not improve upon both individual constituent combiners.
- **Pass Rule**: Significantly lower CRPS than both standalone EMOS-CSG and BMA at $\ge 3$ of 5 leads, with the 95% paired block-bootstrap CI strictly excluding 0.

### 4.1 H10 verdict: **PASS** (3 of 5 leads)

The primary arm is nominated on **train**-fold CRPS, never on test, and was
`quantile_avg` at four of five leads and `per_bin` at 24 h. Paired daily CRPS
differences, negative meaning the combination is better, 95% paired moving-block
bootstrap over both seasons (236-244 test days per lead):

| lead | nominated | vs `emos_csg` | verdict | vs `bma` | verdict |
|---|---|---|---|---|---|
| 24 h | `per_bin` | +0.04976 CI[+0.00696, +0.10349] | fail | -0.04782 CI[-0.12483, -0.00422] | **pass** |
| 48 h | `quantile_avg` | -0.00753 CI[-0.02049, +0.00695] | fail | -0.02874 CI[-0.04128, -0.01847] | **pass** |
| 72 h | `quantile_avg` | -0.01719 CI[-0.03230, -0.00230] | **pass** | -0.02504 CI[-0.03640, -0.01596] | **pass** |
| 96 h | `quantile_avg` | -0.02431 CI[-0.04174, -0.00949] | **pass** | -0.02326 CI[-0.03560, -0.01364] | **pass** |
| 120 h | `quantile_avg` | -0.01984 CI[-0.03642, -0.00655] | **pass** | -0.02317 CI[-0.03205, -0.01470] | **pass** |

**3 of 5 leads beat both parents, meeting the pre-registered bar.** The gain
grows with lead time: at 24 h the combination is significantly *worse* than
EMOS-CSG, and by 120 h it is ahead of both by roughly 0.02 mm CRPS per day.

Every arm is reported separately, so a pass on the nominated arm cannot hide a
method that failed:

| arm | leads where it beat **both** parents |
|---|---|
| `per_bin` | **0 / 5** |
| `quantile_avg` (fitted) | **3 / 5** |
| `quantile_avg_fixed` (w=0.5) | **4 / 5** |
| `linear_pool` | **4 / 5** |

---

## 5. What the verdict does and does not license

**It licenses** quantile averaging (Vincentization) as the recommended
probabilistic combiner, with the CI above. Both parents are worse than the
average of their quantile functions at 3 of 5 leads, which is the substantive
finding: the two parents carry complementary information, exactly as Javanshiri
(2021) and Ji (2025) describe, and averaging in quantile space recovers skill
that neither has alone.

**It does not license the claim it was partly built to rescue.** Slide 2 says
WEAVR "uses the strongest combiner per bin". That is `per_bin`, and `per_bin`
scored **0 of 5** -- it was significantly *worse* than EMOS-CSG at 24 h and
never beat both parents anywhere. Per-bin selection picks a winner on train-fold
CRPS per rain bin and then applies it unconditionally at test time, which throws
away the hedge exactly where it is needed. Slide 2 keeps no per-bin claim. The
claim that can return is the quantile-averaging one, and it is a different claim
from the one currently on the slide.

**Three qualifications that belong with the pass:**

1. **Fitting the Vincentization weight did not help.** The pre-registered fixed
   weight of 0.5 beat both parents at 4 of 5 leads; the train-fitted weight
   (0.50-0.70 by lead and fold) managed 3 of 5. The fitted weight was chosen on
   train CRPS, which is the honest procedure, but on this evidence the extra
   freedom bought nothing and cost one lead. **Recommendation: use the fixed
   w=0.5 and drop the fit.**
2. **The margin is small.** 0.02-0.03 mm of daily CRPS against a 4.5 mm total.
   Real, reproducible and significant under the paired test, but not the
   magnitude of a method change.
3. **All arms remain over-dispersed on wetness** (PIT 0.61-0.63, see §3a).
   Averaging two over-forecasting parents does not produce a calibrated
   forecast; it produces a slightly better over-forecasting one.

---

## 6. Open problems this step did not fix

- **Universal PIT bias (0.61-0.63).** All six arms, near-identically. Shared by
  both parents, so it is upstream of the combination rule. Unresolved.
- **Tier 2 `heavy` x NE1/SI variance blow-up.** Still unexplained. H2 and H3
  both fail against the BMA parent, so if that arm is defective, the negative
  verdicts in Step 07 are partly measuring a bug rather than a method
  limitation. Highest-value open item in the project.
- **BMA quantiles are Monte Carlo** at 500 draws (`--n-samples`), so its 0.99
  quantile is draw-count limited. The CRPS is a 99-level quantile
  approximation for all six arms, deliberately uniform so that no arm is
  advantaged by a better estimator.
- **The linear pool's inverse is interpolated**, not closed-form (§2 C). It is
  kept comparison-only for that reason, and a test pins it inside the parents'
  CRPS range so an inversion failure cannot pass as a pooling result.

---

## 7. Four bugs found by reading the numbers, not the tests

This step needed four runs. Every run exited 0, passed the test suite, and wrote
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
