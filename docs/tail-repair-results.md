# Tail Repair Results: Part 1 — Quantile Mapping

## 1. Overview and Problem Statement

AI numerical weather prediction models (GraphCast, AIFS, GenCast) produce spatially coherent forecasts but suffer from severe intensity attenuation in extreme precipitation regimes. As documented in the WEAVR audit:
- Raw GraphCast detects 0 events exceeding 115.6 mm (very heavy rain) at any lead time beyond day 1.
- No model detects events exceeding 204.5 mm (extremely heavy rain).
- Optimizing solely for domain-wide RMSE encourages conservative, over-smoothed forecasts that extinguish high-impact convective signals.

Quantile mapping (QM) addresses this by re-calibrating the marginal cumulative distribution function (CDF) of each model to match IMD ground-truth observations prior to multi-model blending or EMOS fitting:
$$x_{\text{corr}} = F_{\text{obs}}^{-1}(F_{\text{model}}(x))$$

---

## 2. Quantile Mapping Methodology

Implemented in `src/weavr/quantile_mapping.py`:

1. **Drizzle Elimination (Wet-Day Frequency Adjustment)**:
   - Numerical models frequently forecast light drizzle over vast domains ($p_{\text{dry}, \text{model}} < p_{\text{dry}, \text{obs}}$).
   - Rather than inflating drizzle, the forecast value corresponding to the observed dry quantile $F_{\text{model}}^{-1}(p_{\text{dry}, \text{obs}})$ is established as the zero-precipitation threshold. All values below this cutoff are mapped to 0.0 mm.

2. **Upper-Tail Extrapolation**:
   - Beyond the maximum quantile fitted in the empirical training distribution ($q_{\max} = 0.999$), constant additive anomaly correction is applied:
     $$x_{\text{corr}} = x + (q_{\text{obs}, \max} - q_{\text{model}, \max})$$
   - This avoids unstable multiplicative scaling that can lead to physically unrealistic rainfall values.

3. **Regional and Lead-Time Stratification**:
   - Quantile transfer functions are fit separately for each homogeneous IMD meteorological region and lead time ($24\text{ h}$ to $120\text{ h}$).

4. **Reference Variants**:
   - **`same_period`**: Observational reference is the training fold IMD gauge analysis.
   - **`climatology`**: Observational reference is the 15-year historical IMD climatology excluding the test season.

---

## 3. HEPPI Methodological Comparison

In `HEPPI` (NCMRWF Universal Quantile Mapping, UQM), quantile mapping was shown to effectively reconstruct the heavy precipitation tail, while parametric EMOS excelled at sharpness. In WEAVR, QM is employed not to replace EMOS, but as a pre-processing transfer function to feed calibrated members into the Tier 1 regional blender and Tier 2 EMOS-CSG pipelines.

---

## 4. Pre-registered Hypothesis H7 (Part 1: QM)

- **Criterion**: tw-CRPS (threshold-weighted CRPS at $64.5\text{ mm}$) or SEDI at $115.6\text{ mm}$ must improve at $\ge 3$ of 5 lead times with the 95% paired bootstrap CI strictly excluding 0, while Brier score at $7.5\text{ mm}$ does not degrade significantly.
- **Trade-off Note**: Adjusting the tail upward inevitably increases MSE / RMSE slightly due to double-penalty effects in displacement errors, which is mathematically expected and reported transparently.
