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

### 4.1 Measured H7 verdict (Step 11)

LOYO over both daily seasons (`data/baseline_{2018,2020}_jjas_daily.zarr`), Tier 1 blend scope, paired moving-block bootstrap CIs. Reproduce with `scripts/run_tail_repair_qm.py`.

| variant | tw-CRPS@64.5 Δ | SEDI@115.6 Δ | Brier@7.5 degraded | RMSE Δ | **H7** |
|---|---|---|---|---|---|
| `same_period` | +0.474 … +0.554 (all 5 leads **worse**, CI excludes 0) | −0.112 … −0.256 (**improved**, all 5 leads) | no | +4.09 … +4.71 mm | **PASS** (SEDI branch) |
| `climatology` | +0.698 … +0.802 (all 5 leads **worse**, CI excludes 0) | −0.136 … −0.271 (**improved**, all 5 leads) | no | +5.83 … +6.65 mm | **PASS** (SEDI branch) |

**H7 (QM part) passes for both variants, but only through the SEDI branch — and it is not free.**

- The improvement comes entirely from SEDI@115.6 mm: 5/5 leads improve with the CI excluding 0, for both variants. The pass threshold is ≥3/5.
- tw-CRPS@64.5 mm moves the **wrong way at every lead**, significantly (CIs exclude 0): QM sharpens the extreme tail at the cost of the threshold-weighted score it was supposed to help.
- RMSE increases significantly at every lead, by roughly **+4.1 to +4.7 mm** (`same_period`) and **+5.8 to +6.7 mm** (`climatology`). The `climatology` variant pays roughly 1.5 mm more than `same_period`.
- Brier@7.5 mm does **not** degrade for either variant, so the pre-registered non-inferiority guard is satisfied and the pass is not the artifact of a light-rain forecast being wrecked.

**Correction to an earlier run.** A previous run reported `climatology` as *failing* H7 (Brier@7.5 degrading at 5/5 leads). That was a bug: the runner passed `climatology["rain"].mean(dim="time")` — a single value per grid cell — as the reference, so the fitted quantile map was degenerate. With the full climatological distribution the Brier guard passes. The numbers in this table supersede the earlier ones.

### 4.2 HEPPI directional check — PENDING

Not performed. The check requires NEPS-G raw 2018 / 2019 archives from step 08 (#70); those data are not available. No number is reported in its place.

---

# Tail Repair Results: Part 2 — Extreme-Value Tail (EVT)

## 1. Motivation and Problem Statement

In the baseline calibration pipeline (Tier 2 EMOS-CSG), forecast cells falling into rain-intensity bins with too few historical training observations (notably the `extremely_heavy` bin, $\ge 204.5\text{ mm}$) receive a point-mass fallback at zero. On the extreme-probability dashboard map, this was previously displayed as a grey "not a real probability" fallback mask.

Even with multi-season datasets, sample counts for localized extreme rainfall ($>200\text{ mm}$) are inherently sparse in single forecast training sets. Extreme Value Theory (EVT) provides the standard statistical framework to model exceedance tails above a high threshold without requiring thousands of localized training events.

---

## 2. Pooled Generalized Pareto Distribution (GPD) Formulation

Implemented in `src/weavr/tail.py`:

### Splicing at Threshold $u$
We set $u = 64.5\text{ mm}$ (IMD's official heavy rainfall threshold). For rainfall values $y > u$:
$$P(Y > y \mid \text{forecast}) = P_{\text{CSGD}}(Y > u \mid \text{forecast}) \cdot \left(1 + \xi \frac{y - u}{\sigma_r}\right)^{-1/\xi}$$
where:
- $P_{\text{CSGD}}(Y > u \mid \text{forecast})$ is conditioned on the weather forecast via the nearest fittable EMOS-CSG bin.
- $\xi$ is a **shared national shape parameter** pooled across India to stabilize tail estimation.
- $\sigma_r$ is a **region-specific scale parameter** fit for each of the six Sreekala & Babu monsoon zones (`WC`, `WI`, `SI`, `CI`, `NE1`, `NE2`).

For $y \le u$, the predictive probability is given directly by CSGD:
$$P(Y > y) = P_{\text{CSGD}}(Y > y)$$
This guarantees **exact continuity at $y = u$**, strict monotonicity, and proper bounds in $[0, 1]$.

### Maximum Likelihood Estimation and Diagnostics
The pooled GPD parameters $(\xi, \{\sigma_r\})$ are fit via maximum likelihood estimation (`scipy.optimize.minimize` with L-BFGS-B) on historical IMD gauge analysis exceedances excluding the test year.

### Measured shape parameter
Fitted on both daily seasons, LOYO, at $u = 64.5\text{ mm}$:

| fold | $\hat{\xi}$ at 24 h | 48 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|
| 2018 (train on 2020) | 0.1191 | 0.1190 | 0.1192 | 0.1190 | 0.1179 |
| 2020 (train on 2018) | 0.0989 | 0.0987 | 0.0987 | 0.0984 | 0.0979 |

$\hat{\xi} \approx 0.10$–$0.12$, positive throughout and stable across lead times within a fold, which is the heavy-tailed Fréchet behaviour characteristic of tropical monsoon downpours. The modest fold-to-fold difference (≈0.02) reflects which season the pooled fit sees, not lead-time drift. **No confidence interval is reported for $\hat{\xi}$**: the MLE is not accompanied by a parametric or bootstrap interval in `src/weavr/tail.py`, and none is claimed here.

All 5 rain bins (`dry`, `light`, `heavy`, `very_heavy`, `extremely_heavy`) fitted in every fold and lead — i.e. no `extremely_heavy` point-mass fallback remained.

---

## 3. Cell Classification & Dashboard Mapping

Every grid cell receives a transparent method flag:
- `"csgd"`: Evaluated within the fittable CSGD domain ($y \le u$).
- `"csgd+gpd_tail"`: Repaired extreme probability using the spliced GPD tail ($y > u$ or unfittable forecast bins).
- `"fallback"`: Preserved strictly for true missing/degenerate data.

On the extreme-probability map, cells that previously rendered as fallback grey now show real, calibrated probabilities for $P(\ge 115.6\text{ mm})$ and $P(\ge 204.5\text{ mm})$ with a threshold toggle.

---

## 4. Measured H7 verdict (Step 12, EVT arm)

LOYO over both daily seasons, `csgd_only` (point-mass control) vs `csgd_gpd_tail` (spliced EVT). Reproduce with `scripts/run_tail_repair_evt.py`. Paired moving-block bootstrap CIs; 236–244 test days per lead.

| threshold | false-zero cells (control → EVT) | Brier improved | SEDI improved |
|---|---|---|---|
| 115.6 mm | 193 → **0** | 2/5 leads | **0/5 leads** |
| 204.5 mm | 193 → **0** | 2/5 leads | **0/5 leads** |

**The spliced tail does eliminate every false zero-probability cell, and that is the honest win. It does not improve SEDI.**

- Every cell that previously got a hard 0.0 now receives a real spliced probability, at both thresholds, across all leads and folds.
- SEDI **worsens at 5/5 leads** with CIs excluding 0 (Δ ≈ +0.07 to +0.16). Removing hard zeros inflates false alarms faster than it gains hits, so the skill component drops even though coverage improves.
- Brier at 115.6/204.5 mm improves significantly at only 2/5 leads (24 h and 48 h); the remaining leads are neutral, and none is significantly degraded.
- **H7 is not met by the EVT arm on its own.** The improvement leg fails (0/5 on SEDI, and the tw-CRPS leg is not measured here because the EVT arm produces an exceedance probability rather than a full predictive distribution). The zero-elimination result stands on its own and is what unblocks the dashboard fallback mask.

**Raw IFS-ENS member-counting benchmark: NOT RUN.** Daily IFS-ENS stores for 2018 and 2020 do not exist yet (#69). The runner accepts `--ifs-ensemble-stores` and reports the comparison as unavailable when it is absent. No comparison against IFS-ENS member counting is claimed.

---

## 5. Honest Limitations

- **Climatological Conditioning**: The GPD excess distribution describes the regional climatological tail behavior above $u$. Conditioning on the specific forecast lead and meteorological forcing enters solely through the splice anchor $P_{\text{CSGD}}(Y > u)$.
- **Confidence Intervals**: Exceedance events $\ge 204.5\text{ mm}$ remain statistically rare (a few dozen grid cells per season), resulting in wide paired-bootstrap confidence intervals for the uppermost thresholds.

