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

In `HEPPI` (NCMRWF Universal Quantile Mapping, UQM), quantile mapping was shown to effectively reconstruct the heavy precipitation tail, while parametric EMOS excelled at sharpness. In WEAVR, QM is employed not to replace EMOS, but as a pre-processing transfer function to feed calibrated members into the Tier 1 regional blender and Tier 2 EMOS-CSG pipelines. §4.2 measures the Tier 2 path and finds it hurts.

---

## 4. Pre-registered Hypothesis H7 (Part 1: QM)

- **Criterion**: tw-CRPS (threshold-weighted CRPS at $64.5\text{ mm}$) or SEDI at $115.6\text{ mm}$ must improve at $\ge 3$ of 5 lead times with the 95% paired bootstrap CI strictly excluding 0, while Brier score at $7.5\text{ mm}$ does not degrade significantly.
- **Trade-off Note**: Adjusting the tail upward inevitably increases MSE / RMSE slightly due to double-penalty effects in displacement errors, which is mathematically expected and reported transparently.

### 4.1 Measured H7 verdict (Step 11)

LOYO over both daily seasons (`data/baseline_{2018,2020}_jjas_daily.zarr`), Tier 1 blend scope, paired moving-block bootstrap CIs, 234–242 test days per lead. Reproduce with `scripts/run_tail_repair_qm.py`.

On the corrected pairing (#106) both verdicts stand. Every effect is in the same direction and of similar size: SEDI gains grew slightly (−0.112 … −0.256 → −0.160 … −0.279 for `same_period`), and the tw-CRPS and RMSE penalties shrank by about 10%.

| variant | tw-CRPS@64.5 Δ | SEDI@115.6 Δ | Brier@7.5 degraded | RMSE Δ | **H7** |
|---|---|---|---|---|---|
| `same_period` | +0.421 … +0.513 (all 5 leads **worse**, CI excludes 0) | −0.160 … −0.279 (**improved**, all 5 leads) | no | +3.62 … +4.36 mm | **PASS** (SEDI branch) |
| `climatology` | +0.630 … +0.751 (all 5 leads **worse**, CI excludes 0) | −0.185 … −0.296 (**improved**, all 5 leads) | no | +5.28 … +6.22 mm | **PASS** (SEDI branch) |

**H7 (QM part) passes for both variants, but only through the SEDI branch — and it is not free.**

- The improvement comes entirely from SEDI@115.6 mm: 5/5 leads improve with the CI excluding 0, for both variants. The pass threshold is ≥3/5.
- tw-CRPS@64.5 mm moves the **wrong way at every lead**, significantly (CIs exclude 0): QM sharpens the extreme tail at the cost of the threshold-weighted score it was supposed to help.
- RMSE increases significantly at every lead, by roughly **+3.6 to +4.4 mm** (`same_period`) and **+5.3 to +6.2 mm** (`climatology`). The `climatology` variant pays roughly 1.7 mm more than `same_period`.
- Brier@7.5 mm does **not** degrade for either variant, so the pre-registered non-inferiority guard is satisfied and the pass is not the artifact of a light-rain forecast being wrecked.

**Out-of-sample climatology (correction).** The `climatology` reference is described above as excluding the test season, but until this revision the runner passed the full 2006–2020 climatology, which contains both LOYO test seasons — so each fold's quantile map had seen the observations it was scored against. It now drops every held-out year (`weavr.quantile_mapping.climatology_excluding_years`). Re-measured before #106, the effect was in the third decimal (tw-CRPS +0.698 → +0.699 and SEDI −0.136 → −0.135 at the edge of their ranges; RMSE unchanged at the precision quoted) and **both verdicts stand**. `same_period` never used the climatology and is byte-identical.

**Correction to an earlier run.** A previous run reported `climatology` as *failing* H7 (Brier@7.5 degrading at 5/5 leads). That was a bug: the runner passed `climatology["rain"].mean(dim="time")` — a single value per grid cell — as the reference, so the fitted quantile map was degenerate. With the full climatological distribution the Brier guard passes. The numbers in this table supersede the earlier ones.

### 4.2 QM on the members before EMOS-CSG — makes CRPS worse

§3 describes QM as a pre-processing step feeding calibrated members into Tier 2 EMOS-CSG. `scripts/run_tail_repair_qm_emos.py` tests exactly that: EMOS-CSG fitted on the raw lagged GraphCast members (`raw_members`) against EMOS-CSG refitted on members quantile-mapped per region (`qm_members_same_period`, `qm_members_climatology`), same LOYO folds, same held-out days. The maps are fitted on training-fold forecasts only, and the climatology reference drops both held-out years. Results in `results/tail_repair_qm_emos.csv` and `results/tail_repair_qm_emos_paired.csv`.

`raw_members` reproduces Tier 2's EMOS-GraphCast exactly (pooled CRPS 4.1390 mm at lead 24 in both, and identical at every lead and fold), which is the check that this runner scores the same thing.

Paired CRPS difference, QM arm minus `raw_members` (positive = QM worse), 95% moving-block bootstrap CI:

| lead | `same_period` | 95% CI | `climatology` | 95% CI |
|---|---|---|---|---|
| 24 h | +0.038 | [+0.023, +0.053] | +0.055 | [+0.041, +0.068] |
| 48 h | +0.037 | [+0.023, +0.052] | +0.053 | [+0.040, +0.067] |
| 72 h | +0.036 | [+0.022, +0.050] | +0.053 | [+0.040, +0.067] |
| 96 h | +0.034 | [+0.020, +0.049] | +0.048 | [+0.034, +0.063] |
| 120 h | +0.035 | [+0.020, +0.052] | +0.049 | [+0.034, +0.066] |

**Both arms are significantly worse at every lead.** Alongside, per fold:

- Brier@7.5 mm worsens by +0.0013 … +0.0026 in every lead and fold.
- SEDI at 64.5 and 115.6 mm moves by at most about ±0.01 either way (−0.013 … +0.012) — no tail gain to set against the CRPS loss.
- RMSE of the predictive mean rises by +3.1 … +4.0 mm (`same_period`) and +3.2 … +6.2 mm (`climatology`), the same double-penalty pattern §4.1 reports for the deterministic sources.

The corrected pairing (#106) shrank the CRPS penalty by about 10% and left the verdict unchanged.

**Reading.** The tail improvement QM delivers on deterministic sources (§4.1, SEDI@115.6) does not survive EMOS-CSG. A plausible but untested explanation is that EMOS-CSG's own regression already corrects the members' marginal bias, so a per-region map applied first adds noise to its predictors without adding information. On this evidence QM should not be inserted ahead of EMOS-CSG. This is a measured negative result for the Tier 2 path only; it does not touch the §4.1 H7 verdict, which is scoped to the Tier 1 blend.

### 4.3 HEPPI directional check — low agreement, and HEPPI's UQM is the outlier

Step 11 asks for a directional check, not a reproduction: fit this module's empirical per-region QM on NEPS-G raw 2018 against IMD 2018, apply it to NEPS-G raw 2019, and compare its quantiles with HEPPI's own UQM output for 2019, region by region. Data: the HEPPI release (https://edata.bham.ac.uk/698/, Angus 2021), dates recovered with `docs/heppi-date-map.csv`; 120 days of 2018, 122 of 2019, 23 members, on weavr's grid. Only the aggregate statistics below are reported (`results/heppi_directional_check.csv`); no HEPPI-derived grid is committed. Reproduce with `python scripts/check_heppi_qm_direction.py <HEPPI_DIR>`.

Mean absolute error against IMD 2019, averaged over the six regions:

| statistic | raw NEPS-G | weavr QM | HEPPI UQM |
|---|---|---|---|
| wet-day fraction (≥ 0.1 mm) | 0.28 | **0.02** | 0.29 |
| median (mm) | 4.11 | **0.37** | 5.54 |
| 75th percentile | 4.56 | **1.73** | 7.62 |
| 90th percentile | **2.99** | 3.94 | 9.61 |
| 95th percentile | **5.90** | 5.99 | 12.06 |
| 99th percentile | 19.01 | **14.52** | 21.79 |
| 99.9th percentile | 44.82 | 36.34 | **36.00** |

**The two corrections move each quantile the same way in only 11 of 42 region × statistic cells.** weavr's QM moves the raw forecast towards IMD where it matters most for drizzle and the body of the distribution (the wet-day fraction falls from 0.76–0.98 to within 0.03 of IMD's in every region) and helps at the 99th percentile. HEPPI's UQM moves the other way: its median is above the raw forecast's in all six regions, and over the whole domain it is *wetter* than the raw forecast in both years (wet-day fraction 2018: IMD 0.53, raw 0.87, UQM 0.89; 2019: IMD 0.58, raw 0.88, UQM 0.90).

That is HEPPI's file, not a misreading of it: at every grid cell its UQM values are a perfect monotone transform of the raw ones (rank correlation 1.000; 0.68 with the members reversed, 0.17 with latitude and longitude swapped), as a quantile map should be, so the layout and dates are right. **HEPPI's own code explains the direction.** `Generalized_QM.m` (Angus 2020, file 3 of the release) implements five variants -- `Gamma`, `GEV`, `DGQM` (Pasten-Zapata et al. 2020's double gamma, split at the 90th percentile; the script's default), `GEVG` (gamma below the observed 90th percentile, GEV above) and `Emp` (empirical) -- and all five share two choices:

- **Dry days: only values above 0 mm are mapped, and there is no wet-day frequency correction.** At each grid cell the forecast's wet values are fitted to the observations' wet values; a forecast of 0 stays 0, and a forecast of 0.3 mm stays wet and is mapped onto IMD's *wet-day* distribution, whose values are far larger because IMD has fewer, heavier wet days. Drizzle cannot be removed and is inflated instead -- exactly the wetter-than-raw output measured above. weavr's QM matches the dry fraction first (§2), which is why the two move in opposite directions in the body of the distribution.
- **Beyond the training range: no explicit rule.** The parametric variants extrapolate through the fitted distribution's inverse CDF (CDF values of exactly 1 are nudged below 1); `Emp` snaps to the nearest observed value, so it cannot exceed the training maximum.

It also fits per grid cell rather than per region, and on all days pooled (the script notes the forecast days and members were "smashed together"), so not under leave-one-year-out. **HEPPI's UQM output therefore cannot serve as a validation target for weavr's QM**: it answers a different question (map wet-day intensities, leave occurrence alone). The check is reported as run, with that result.

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
| 2018 (train on 2020) | 0.1190 | 0.1192 | 0.1190 | 0.1179 | 0.1181 |
| 2020 (train on 2018) | 0.0987 | 0.0987 | 0.0984 | 0.0979 | 0.0978 |

$\hat{\xi} \approx 0.10$–$0.12$, positive throughout and stable across lead times within a fold: a heavy-tailed (Fréchet-type) excess distribution. The tail is fitted on IMD observations only, so lead time can only enter through which days fall in the fold; the fold-to-fold difference (≈0.02) is the larger effect. #106 shifts each lead's training days by one, so each value above is the one the next lead had before; the block-bootstrap diagnostics below fit every training day and are unchanged.

### How uncertain $\hat{\xi}$ is

`fit_pooled_gpd` returns an information-matrix standard error, now emitted by the runner as `tail_shape_xi_se`, `tail_shape_xi_ci_lo` and `tail_shape_xi_ci_hi`. Two corrections were needed before it could be quoted:

1. **The standard error came from the wrong curvature.** It was taken from the $\xi$–$\xi$ entry of the Hessian alone, which ignores the correlation between $\xi$ and the six regional scales. It is now the $\xi$ diagonal of the **inverse** Hessian. On simulated pooled GPD data the old interval was 26% too narrow with 85% coverage; the corrected one gives SE 0.0175 against a theoretical 0.0174 and 94.5% coverage (`tests/test_tail.py`).
2. **The information interval assumes independent exceedances, and they are not.** One monsoon day produces exceedances in dozens of neighbouring cells, and wet spells span consecutive days. `scripts/diagnose_evt_tail.py` refits on a moving-block bootstrap that resamples whole days in 7-day blocks (`results/tail_evt_diagnostics.csv`).

At the runner's $u = 64.5$ mm:

| fold | $\hat{\xi}$ | information 95% CI | **block-bootstrap 95% CI** | width ratio |
|---|---|---|---|---|
| 2018 (train on 2020) | 0.119 | [0.094, 0.145] | **[0.064, 0.149]** | ×1.67 |
| 2020 (train on 2018) | 0.099 | [0.073, 0.125] | **[0.036, 0.155]** | ×2.31 |

**The block-bootstrap interval is the one to quote.** The information interval is 1.7–2.3× too narrow because it treats correlated exceedances as independent. Even so, both block intervals exclude 0, so the heavy-tail sign is supported; the magnitude is uncertain by roughly a factor of two.

### Threshold choice

A GPD is appropriate only above a threshold where $\hat{\xi}$ has stabilised. Refitting across candidate thresholds on each training season:

| $u$ (mm) | 35.5 | 50 | **64.5** | 80 | 100 | 115.6 |
|---|---|---|---|---|---|---|
| $\hat{\xi}$, fold 2018 | 0.176 | 0.146 | **0.119** | 0.085 | 0.052 | 0.058 |
| $\hat{\xi}$, fold 2020 | 0.142 | 0.108 | **0.099** | 0.101 | 0.070 | 0.068 |
| exceedances, fold 2018 | 27,619 | 14,457 | **8,200** | 4,803 | 2,584 | 1,688 |

$\hat{\xi}$ drifts downward from 0.14–0.18 at 35.5 mm to 0.06–0.07 at 115.6 mm. On the 2020 fold it is flat from 50 to 80 mm and every higher-threshold estimate lies inside the 64.5 mm block-bootstrap interval; on the 2018 fold it is still falling at 64.5 mm, and the 100 and 115.6 mm estimates (0.052, 0.058) fall below that interval. So **$u = 64.5$ mm sits at the lower edge of the stable region, not inside it**. It is defensible as IMD's own "heavy" boundary and because it keeps ~8,000 exceedances per fold, but a higher $u$ would give a lighter tail: at 115.6 mm, $\hat{\xi}$ is about half the 64.5 mm value. The probabilities at 204.5 mm, which extrapolate furthest, are the most sensitive to this choice.

In this runner's own binning, all 5 rain bins fit in every fold and lead (`fallback bins unfitted: []` in every log line), so no `extremely_heavy` point-mass fallback remains here. This counts something different from Tier 2's per-bin and per-(bin, region) fits, where `extremely_heavy` never fits (`docs/v2-evidence-base-results.md` §7); neither number is wrong, and they should not be quoted interchangeably.

---

## 3. Cell Classification & Dashboard Mapping

Every grid cell receives a transparent method flag:
- `"csgd"`: Evaluated within the fittable CSGD domain ($y \le u$).
- `"csgd+gpd_tail"`: Repaired extreme probability using the spliced GPD tail ($y > u$ or unfittable forecast bins).
- `"fallback"`: Preserved strictly for true missing/degenerate data.

On the extreme-probability map, cells that previously rendered as fallback grey now show real, calibrated probabilities for $P(\ge 115.6\text{ mm})$ and $P(\ge 204.5\text{ mm})$ with a threshold toggle.

**The map is out of sample (#74).** `scripts/export_dashboard_example_grids.py --fit-store … --fit-lagged-store … --fit-ifs-ensemble-store …` fits EMOS-CSG (`ifs_ens`) and the GPD tail on 2018 and draws a 2020 day; the NPZ records the fitting store under `fitted_on`.

**Correction: the map's bins.** The export fitted each bin on cells classified by the GraphCast ensemble mean (the pipeline's convention, as in Tier 2), but classified the displayed day's cells by the IFS ensemble mean, so a cell could be scored with a bin model fitted on different cells. Above $u$ the bin picks which CSGD supplies $P(Y > u)$ for the splice, so this changed the map. The displayed day is now classified by the GraphCast mean too, with the IFS mean and spread kept as the predictors (`test_dashboard_probability_grid_classifies_bins_like_the_fit`). This touched only the dashboard: the evaluated EVT runner fits and tests on one source throughout.

**Correction: the map's tail markers.** Both thresholds lie above $u$, so every fitted cell is `csgd+gpd_tail`, and the map used to put a dot on all 17,415 cells (near-black squares in geography mode), which marked nothing and hid the probability colours. The dashboard now draws per-cell markers only when some cells use the tail and others do not, and otherwise states once, in the legend and banner, that the whole map at this threshold is a tail extrapolation (`extremeProbabilityTailIsUniform`).

---

## 4. Measured H7 verdict (Step 12, EVT arm)

LOYO over both daily seasons, `csgd_only` (point-mass control) vs `csgd_gpd_tail` (spliced EVT). Reproduce with `scripts/run_tail_repair_evt.py` (command in §4c). Paired moving-block bootstrap CIs; 234–242 test days per lead. Results: `results/tail_repair_evt.csv`, `results/tail_repair_evt_paired.csv`.

| threshold | false-zero cell-days, control → EVT | Brier improved | SEDI improved |
|---|---|---|---|
| 115.6 mm | 193 → **0** | 3/5 leads | **0/5 leads** (worse at 5/5) |
| 204.5 mm | 193 → **0** | 2/5 leads | **0/5 leads** (worse at 5/5) |

The 193 is a total over all five leads and both folds. The control's false zeros are not spread evenly: the 2018 fold has none at any lead, and the 2020 fold has 1 at 24 h, **116 at 96 h** and 76 at 120 h. Every bin fits in every fold and lead here (`fallback bins unfitted: []` throughout), so these are cells where the fitted CSGD's exceedance probability is exactly 0, not unfitted bins. The count and its split are the same before and after #106.

**The spliced tail eliminates every false zero-probability cell-day, and that is the honest win. It does not improve SEDI.**

- Every cell-day that got a hard 0.0 now receives a real spliced probability, at both thresholds.
- SEDI is **worse at 5/5 leads** at both thresholds, every CI excluding 0: by 0.073–0.092 at 115.6 mm and 0.121–0.163 at 204.5 mm. §4b shows why.
- Brier improves significantly at 3 of 5 leads at 115.6 mm (24, 48 and 72 h) and 2 of 5 at 204.5 mm (96 and 120 h). The rest are neutral; none is significantly worse.
- **H7 is not met by the EVT arm on its own.** The improvement leg fails (0/5 on SEDI). The tw-CRPS leg is not measured, because the EVT arm produces an exceedance probability rather than a full predictive distribution. The zero-elimination result stands on its own and is what unblocks the dashboard fallback mask.

---

## 4a. Against raw IFS-ENS member counting

`raw_ifs_ens_member_counting` is the empirical fraction of the 50 IFS-ENS members above the threshold: no calibration, the thing a forecaster gets for free. It needs the daily full-member IFS-ENS stores for both seasons, which step 07 built (#69). Averaged over the ten lead-fold cells:

| arm | false-zero cell-days per lead-fold, @115.6 | @204.5 | BSS @115.6 | BSS @204.5 | SEDI @115.6 | SEDI @204.5 |
|---|---|---|---|---|---|---|
| `csgd_only` (point-mass control) | 0–116 | 0–116 | −0.044 | −0.020 | 0.321 | 0.319 |
| `csgd_gpd_tail` (spliced EVT) | **0** | **0** | **−0.018** | **−0.007** | 0.238 | 0.181 |
| `raw_ifs_ens_member_counting` | **1.96–2.07 million** | **2.02–2.10 million** | −0.284 | −0.255 | 0.333 | 0.256 |

Each lead-fold cell scores 2.04–2.11 million grid cell-days (the full 129×135 grid, sea included, times 117–121 days), so **raw member counting assigns probability exactly 0.0 to almost every cell-day**. With 50 members an exceedance at 115.6 or 204.5 mm needs at least one member to clear the bar, which almost never happens at a daily accumulation that size. (An earlier version of this table gave 2,087,404 as if it were a total; it is one lead-fold cell, and the benchmark rows were also written four times over -- see the runner fix in this branch.) Its Brier skill is negative and far worse than either calibrated arm.

So the tail is not merely better than member counting on Brier: it is the **only** arm that assigns a non-zero probability to essentially every extreme cell-day.

**What the benchmark does not rescue:**

- The tail is **still worse than climatology** on Brier at both thresholds (BSS −0.018 and −0.007). Being the least-bad probabilistic forecast is not the same as being a good one.
- Its SEDI is the **worst** of the three arms at both thresholds (0.238 at 115.6 mm against 0.321 and 0.333; 0.181 at 204.5 mm against 0.319 and 0.256).

Net: the tail is the only arm that gives a non-degenerate probability at these thresholds, and it beats raw member counting on Brier. It still loses to climatology, costs SEDI, and over-forecasts (§4b).

---

## 4b. Reliability: the tail forecasts extremes about three times too often

`results/tail_repair_evt_reliability.csv` bins every cell-day's forecast probability and records the observed frequency per bin. Summed over all cell-days, expected events (Σ forecast probability) against observed events:

| arm | 115.6 mm | 204.5 mm |
|---|---|---|
| `csgd_only` | 1.92× | 1.29× |
| `csgd_gpd_tail` | **2.86×** | **3.02×** |
| `raw_ifs_ens_member_counting` | 2.14× | 3.32× |

The ratio is stable across leads (tail: 2.89× → 2.85× at 115.6 mm from 24 h to 120 h). Within the bins the pattern holds too: at 115.6 mm, lead 24, the control's 0.1–0.2 probability bin verifies at 5.8%, and its 0.2–0.5 bin at 8.9%.

That is the mechanism behind §4's SEDI loss. Lifting every hard zero to a spliced probability spreads mass onto cell-days that stay dry, so false alarms grow faster than hits. A tail that over-forecasts by ~3× is not calibrated, whatever its Brier score says relative to the control.

Two causes were open. One is now ruled out:

1. **The forecast–observation day alignment (#106) — ruled out.** Every arm used to be fitted and scored against the IMD day before the one each forecast covered, which could have pushed the calibration towards over-dispersion. On the corrected pairing the over-forecast is essentially unchanged (2.91× → 2.86× at 115.6 mm, 3.07× → 3.02× at 204.5 mm), so the misalignment was not the cause.
2. **The threshold choice** (§2) — still open: a lower $u$ gives a heavier tail, and $u = 64.5$ mm sits at the low edge of the stable region. This is now the first thing to test.

---

## 4c. What was wrong with the benchmark before this run

`--ifs-ensemble-stores` existed on the runner and was accepted, but the flag was
never resolved into paths. `ifs_available` was decided on a non-`None`
argument that nothing then read, the runner printed **"IFS-ENS member-counting
benchmark: enabled"**, and no benchmark row was ever emitted. A reader had no
way to tell the difference between "ran and produced nothing" and "enabled and
ran".

That is a worse failure than an absent feature, because the message claimed it.
Fixed in this branch: the stores are resolved, the arm is computed, and the run
now **raises** if the stores were supplied but the benchmark produced no rows,
rather than reporting success.

Four more runner defects were fixed before this run, each with a test in `tests/test_run_tail_repair_evt.py`:

- **The benchmark rows were written four times** -- once per inner-loop iteration rather than once per fold -- so any total summed over the CSV was 4× too large. The committed CSV now has 60 rows, not 120.
- **The lagged stores were resolved with the IFS-ENS default list**, so an explicit `--lagged-stores` could be silently replaced.
- **Test-time ensemble spread used `ddof=0` against `ddof=1` in training**, a small train/test mismatch. Fixing it moved SEDI by at most 0.002 and Brier by under 1e-6.
- **`benchmark_row` counted an exceedance with `>`** where both other arms use `>=`.

Reproduce with:

```
python scripts/run_tail_repair_evt.py \
  --baseline-stores data/baseline_2018_jjas_daily.zarr data/baseline_2020_jjas_daily.zarr \
  --lagged-stores   data/lagged_ensemble_inputs_2018_jjas_daily.zarr data/lagged_ensemble_inputs_2020_jjas_daily.zarr \
  --ifs-ensemble-stores data/ifs_ens_2018_jjas_daily.zarr data/ifs_ens_2020_jjas_daily.zarr \
  --climatology data/imd_seeps_climatology_jjas.zarr \
  --results-dir results --force
python scripts/diagnose_evt_tail.py --results-dir results --force   # §2 threshold and xi-interval diagnostics, ~4 min
```

## 5. Honest Limitations

- **Day alignment (#106) — resolved.** Every number in this document is measured on the corrected forecast-observation pairing. No verdict moved: H7 (QM) still passes on SEDI only, QM still hurts EMOS-CSG, and the EVT arm still removes every false zero while costing SEDI and over-forecasting about 3×.
- **Climatological Conditioning**: The GPD excess distribution describes the regional climatological tail behavior above $u$. Conditioning on the specific forecast lead and meteorological forcing enters solely through the splice anchor $P_{\text{CSGD}}(Y > u)$.
- **Confidence Intervals**: Exceedance events $\ge 204.5\text{ mm}$ remain statistically rare (about 150 cell-days per season across the whole grid), resulting in wide paired-bootstrap confidence intervals for the uppermost thresholds.

