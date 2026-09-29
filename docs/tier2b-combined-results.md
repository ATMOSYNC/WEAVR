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

## 4. Pre-registered Hypothesis H10

- **Null Hypothesis**: The combined combiner (quantile averaging or per-bin selection) does not improve upon both individual constituent combiners.
- **Pass Rule**: Significantly lower CRPS than both standalone EMOS-CSG and BMA at $\ge 3$ of 5 leads, with the 95% paired block-bootstrap CI strictly excluding 0.
