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

LOYO over both daily seasons (`data/baseline_{2018,2020}_jjas_daily.zarr`), Tier 1 blend scope, paired moving-block bootstrap CIs. Reproduce with `scripts/run_tail_repair_qm.py`.

| variant | tw-CRPS@64.5 Δ | SEDI@115.6 Δ | Brier@7.5 degraded | RMSE Δ | **H7** |
|---|---|---|---|---|---|
| `same_period` | +0.474 … +0.554 (all 5 leads **worse**, CI excludes 0) | −0.112 … −0.256 (**improved**, all 5 leads) | no | +4.09 … +4.71 mm | **PASS** (SEDI branch) |
| `climatology` | +0.699 … +0.802 (all 5 leads **worse**, CI excludes 0) | −0.135 … −0.271 (**improved**, all 5 leads) | no | +5.83 … +6.65 mm | **PASS** (SEDI branch) |

**H7 (QM part) passes for both variants, but only through the SEDI branch — and it is not free.**

- The improvement comes entirely from SEDI@115.6 mm: 5/5 leads improve with the CI excluding 0, for both variants. The pass threshold is ≥3/5.
- tw-CRPS@64.5 mm moves the **wrong way at every lead**, significantly (CIs exclude 0): QM sharpens the extreme tail at the cost of the threshold-weighted score it was supposed to help.
- RMSE increases significantly at every lead, by roughly **+4.1 to +4.7 mm** (`same_period`) and **+5.8 to +6.7 mm** (`climatology`). The `climatology` variant pays roughly 1.5 mm more than `same_period`.
- Brier@7.5 mm does **not** degrade for either variant, so the pre-registered non-inferiority guard is satisfied and the pass is not the artifact of a light-rain forecast being wrecked.

**Out-of-sample climatology (correction).** The `climatology` reference is described above as excluding the test season, but until this revision the runner passed the full 2006–2020 climatology, which contains both LOYO test seasons — so each fold's quantile map had seen the observations it was scored against. It now drops every held-out year (`weavr.quantile_mapping.climatology_excluding_years`). Re-measured, the effect is in the third decimal (tw-CRPS +0.698 → +0.699 and SEDI −0.136 → −0.135 at the edge of their ranges; RMSE unchanged at the precision quoted) and **both verdicts stand**. `same_period` never used the climatology and is byte-identical.

**Correction to an earlier run.** A previous run reported `climatology` as *failing* H7 (Brier@7.5 degrading at 5/5 leads). That was a bug: the runner passed `climatology["rain"].mean(dim="time")` — a single value per grid cell — as the reference, so the fitted quantile map was degenerate. With the full climatological distribution the Brier guard passes. The numbers in this table supersede the earlier ones.

### 4.2 QM on the members before EMOS-CSG — makes CRPS worse

§3 describes QM as a pre-processing step feeding calibrated members into Tier 2 EMOS-CSG. `scripts/run_tail_repair_qm_emos.py` tests exactly that: EMOS-CSG fitted on the raw lagged GraphCast members (`raw_members`) against EMOS-CSG refitted on members quantile-mapped per region (`qm_members_same_period`, `qm_members_climatology`), same LOYO folds, same held-out days. The maps are fitted on training-fold forecasts only, and the climatology reference drops both held-out years. Results in `results/tail_repair_qm_emos.csv` and `results/tail_repair_qm_emos_paired.csv`.

`raw_members` reproduces Tier 2's EMOS-GraphCast to the fourth decimal (pooled CRPS 4.5457 mm at lead 24 in both), which is the check that this runner scores the same thing.

Paired CRPS difference, QM arm minus `raw_members` (positive = QM worse), 95% moving-block bootstrap CI:

| lead | `same_period` | 95% CI | `climatology` | 95% CI |
|---|---|---|---|---|
| 24 h | +0.042 | [+0.027, +0.057] | +0.058 | [+0.046, +0.072] |
| 48 h | +0.043 | [+0.029, +0.058] | +0.059 | [+0.046, +0.073] |
| 72 h | +0.043 | [+0.029, +0.058] | +0.058 | [+0.045, +0.072] |
| 96 h | +0.043 | [+0.030, +0.057] | +0.056 | [+0.044, +0.070] |
| 120 h | +0.046 | [+0.032, +0.061] | +0.058 | [+0.046, +0.073] |

**Both arms are significantly worse at every lead.** Alongside, per fold:

- Brier@7.5 mm worsens by +0.0016 … +0.0026 in every lead and fold.
- SEDI at 64.5 and 115.6 mm moves by less than ±0.01 either way — no tail gain to set against the CRPS loss.
- RMSE of the predictive mean rises by +3.6 … +4.5 mm (`same_period`) and +3.7 … +6.7 mm (`climatology`), the same double-penalty pattern §4.1 reports for the deterministic sources.

**Reading.** The tail improvement QM delivers on deterministic sources (§4.1, SEDI@115.6) does not survive EMOS-CSG. A plausible but untested explanation is that EMOS-CSG's own regression already corrects the members' marginal bias, so a per-region map applied first adds noise to its predictors without adding information. On this evidence QM should not be inserted ahead of EMOS-CSG. This is a measured negative result for the Tier 2 path only; it does not touch the §4.1 H7 verdict, which is scoped to the Tier 1 blend.

### 4.3 HEPPI directional check — PENDING

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

$\hat{\xi} \approx 0.10$–$0.12$, positive throughout and stable across lead times within a fold: a heavy-tailed (Fréchet-type) excess distribution. The tail is fitted on IMD observations only, so lead time can only enter through which days fall in the fold; the fold-to-fold difference (≈0.02) is the larger effect.

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

---

## 4. Measured H7 verdict (Step 12, EVT arm)

LOYO over both daily seasons, `csgd_only` (point-mass control) vs `csgd_gpd_tail` (spliced EVT). Reproduce with `scripts/run_tail_repair_evt.py` (command in §4c). Paired moving-block bootstrap CIs; 236–244 test days per lead. Results: `results/tail_repair_evt.csv`, `results/tail_repair_evt_paired.csv`.

| threshold | false-zero cell-days, control → EVT | Brier improved | SEDI improved |
|---|---|---|---|
| 115.6 mm | 193 → **0** | 2/5 leads | **0/5 leads** (worse at 5/5) |
| 204.5 mm | 193 → **0** | 2/5 leads | **0/5 leads** (worse at 5/5) |

The 193 is a total over all five leads and both folds. The control's false zeros are not spread evenly: the 2018 fold has none at any lead, and the 2020 fold has 1 at 24 h, **116 at 96 h** and 76 at 120 h. Every bin fits in every fold and lead here (`fallback bins unfitted: []` throughout), so these are cells where the fitted CSGD's exceedance probability is exactly 0, not unfitted bins.

**The spliced tail eliminates every false zero-probability cell-day, and that is the honest win. It does not improve SEDI.**

- Every cell-day that got a hard 0.0 now receives a real spliced probability, at both thresholds.
- SEDI is **worse at 5/5 leads** at both thresholds, every CI excluding 0: by 0.070–0.089 at 115.6 mm and 0.106–0.163 at 204.5 mm. §4b shows why.
- Brier improves significantly at 2 of 5 leads at each threshold (115.6 mm: 24 h and 48 h; 204.5 mm: 24 h and 120 h). The rest are neutral; none is significantly worse.
- **H7 is not met by the EVT arm on its own.** The improvement leg fails (0/5 on SEDI). The tw-CRPS leg is not measured, because the EVT arm produces an exceedance probability rather than a full predictive distribution. The zero-elimination result stands on its own and is what unblocks the dashboard fallback mask.

---

## 4a. Against raw IFS-ENS member counting

`raw_ifs_ens_member_counting` is the empirical fraction of the 50 IFS-ENS members above the threshold: no calibration, the thing a forecaster gets for free. It needs the daily full-member IFS-ENS stores for both seasons, which step 07 built (#69). Averaged over the ten lead-fold cells:

| arm | false-zero cell-days per lead-fold, @115.6 | @204.5 | BSS @115.6 | BSS @204.5 | SEDI @115.6 | SEDI @204.5 |
|---|---|---|---|---|---|---|
| `csgd_only` (point-mass control) | 0–116 | 0–116 | −0.036 | −0.018 | 0.279 | 0.293 |
| `csgd_gpd_tail` (spliced EVT) | **0** | **0** | **−0.017** | **−0.006** | 0.200 | 0.155 |
| `raw_ifs_ens_member_counting` | **1.98–2.09 million** | **2.03–2.12 million** | −0.302 | −0.263 | 0.316 | 0.245 |

Each lead-fold cell scores 2.09–2.12 million grid cell-days (the full 129×135 grid, sea included, times 118–122 days), so **raw member counting assigns probability exactly 0.0 to almost every cell-day**. With 50 members an exceedance at 115.6 or 204.5 mm needs at least one member to clear the bar, which almost never happens at a daily accumulation that size. (An earlier version of this table gave 2,087,404 as if it were a total; it is one lead-fold cell, and the benchmark rows were also written four times over -- see the runner fix in this branch.) Its Brier skill is negative and far worse than either calibrated arm.

So the tail is not merely better than member counting on Brier: it is the **only** arm that assigns a non-zero probability to essentially every extreme cell-day.

**What the benchmark does not rescue:**

- The tail is **still worse than climatology** on Brier at both thresholds (BSS −0.017 and −0.006). Being the least-bad probabilistic forecast is not the same as being a good one.
- Its SEDI is the **worst** of the three arms at 204.5 mm (0.155, against 0.293 and 0.245) and second-worst at 115.6 mm.

Net: the tail is the only arm that gives a non-degenerate probability at these thresholds, and it beats raw member counting on Brier. It still loses to climatology, costs SEDI, and over-forecasts (§4b).

---

## 4b. Reliability: the tail forecasts extremes about three times too often

`results/tail_repair_evt_reliability.csv` bins every cell-day's forecast probability and records the observed frequency per bin. Summed over all cell-days, expected events (Σ forecast probability) against observed events:

| arm | 115.6 mm | 204.5 mm |
|---|---|---|
| `csgd_only` | 1.91× | 1.14× |
| `csgd_gpd_tail` | **2.91×** | **3.07×** |
| `raw_ifs_ens_member_counting` | 2.14× | 3.32× |

The ratio is stable across leads (tail: 2.97× → 2.88× at 115.6 mm from 24 h to 120 h). Within the bins the pattern holds too: at 115.6 mm, lead 24, the control's 0.1–0.2 probability bin verifies at 6.1%, and its 0.2–0.5 bin at 6.3%.

That is the mechanism behind §4's SEDI loss. Lifting every hard zero to a spliced probability spreads mass onto cell-days that stay dry, so false alarms grow faster than hits. A tail that over-forecasts by ~3× is not calibrated, whatever its Brier score says relative to the control.

Two causes are open, and the order to check them in is clear:

1. **The forecast–observation day alignment (#106).** Every arm here is fitted and scored against the IMD day before the one each forecast covers. A calibration fitted to the wrong day sees heavy-rain forecasts that mostly verify dry, which should push it towards over-dispersion. This affects every arm, so it has to be ruled in or out first.
2. **The threshold choice** (§2): a lower $u$ gives a heavier tail, and $u = 64.5$ mm sits at the low edge of the stable region.

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

- **Day alignment (#106).** Every number in this part pairs each forecast with the IMD day before the one it covers, as every runner in the project does today. The tail's fit is to IMD observations only, so $\hat{\xi}$ and the regional scales are unaffected. The splice anchor $P_{\text{CSGD}}(Y > u)$, every Brier and SEDI, and the over-forecast in §4b are not, and should be re-measured once #106 is resolved.
- **Climatological Conditioning**: The GPD excess distribution describes the regional climatological tail behavior above $u$. Conditioning on the specific forecast lead and meteorological forcing enters solely through the splice anchor $P_{\text{CSGD}}(Y > u)$.
- **Confidence Intervals**: Exceedance events $\ge 204.5\text{ mm}$ remain statistically rare (about 150 cell-days per season across the whole grid), resulting in wide paired-bootstrap confidence intervals for the uppermost thresholds.

