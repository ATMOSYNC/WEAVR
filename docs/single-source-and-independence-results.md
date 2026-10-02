# Single-source baselines and the independence diagnostic

Produced by
[`scripts/run_single_source_baseline.py`](../scripts/run_single_source_baseline.py)
→ `results/single_source_baseline.csv`, and
[`scripts/run_independence_diagnostic.py`](../scripts/run_independence_diagnostic.py)
→ `results/independence_diagnostic.csv`.

This document answers a question the tier results never asked: **is blending
worth doing at all?** Every tier in this project is justified against
another blend. Nothing recorded what the raw sources score on their own. A
reviewer can compute that in five minutes with this repo's own functions, so
it is computed here, and the uncomfortable half of the answer is stated
first.

Both scripts reuse the tier scripts' code rather than re-implementing it:
alignment and the metres→millimetres conversion come from
`run_tier1_regional_baseline.load_aligned_forecasts_and_obs`, scoring from
`run_tier0_baseline.score_lead`, and the train/test split from
`run_tier0_baseline.train_test_masks`. Every number below is therefore on
literally the same held-out days as the Tier 0/1/2 numbers it is compared
against, not on a split that merely should line up.

---

## 1. No blend beats raw GraphCast on RMSE at 24–96 h

Test-split RMSE (mm), `seasonal_block_split`, `test_fraction=0.2`,
2020 JJAS store. **Bold** is the best in each row.

| Lead | GraphCast | IFS-ENS | HRES | Tier 0 | Tier 1 | Tier 2 EMOS-GraphCast | Tier 2 EMOS-IFS-ENS | n_test |
|---|---|---|---|---|---|---|---|---|
| 24 h | 11.49 | 11.76 | 13.50 | 11.66 | 11.55 | 10.86 | **10.80** | 4 |
| 48 h | 12.45 | 13.78 | 16.42 | 13.54 | 12.62 | **12.08** | 12.31 | 4 |
| 72 h | 11.80 | 13.85 | 17.97 | 13.38 | 12.96 | **11.27** | 12.24 | 4 |
| 96 h | 12.71 | 14.48 | 19.29 | 14.27 | 13.65 | **12.95** | 13.73 | 3 |
| 120 h | 14.83 | 15.38 | 17.65 | 15.30 | **14.82** | 14.88 | 15.08 | 3 |

Read plainly:

- **Raw GraphCast beats both Tier 0 and Tier 1 at every lead from 24 h to
  96 h.** At 72 h the gap is large: 11.80 mm against Tier 0's 13.38 mm.
- The multi-source blends (Tier 0, Tier 1) never win a row.
- The only blends that beat raw GraphCast are Tier 2's EMOS runs — and
  **those are single-source too**: EMOS-GraphCast is a statistical
  calibration of GraphCast alone, not a combination of models. What helps
  here is calibration, not combination.
- Tier 1 does consistently beat Tier 0, so *weighting* the mean is better
  than not weighting it. That is a real result; it is just a much smaller
  one than "the blend is better than the models".

**These test sets are 3–4 days.** No claim here should be read as
established; the point is that the headline "the blend beats the sources" is
not supported by what is currently measured, and must not be asserted until
it is. Steps 05–07 (daily cadence, a second season, leave-one-year-out) and
step 04 (confidence intervals) exist to give these rows enough samples to
mean something.

### The honest baseline is worse than "GraphCast"

Picking GraphCast because it won on the test set is the leak
`src/weavr/splits.py` exists to prevent. The defensible baseline is **the
best single member chosen on the training split**, which the script records
as `best_single_member_on_train`:

| Lead | Chosen on train | Train RMSE | Its test RMSE | Test-best source |
|---|---|---|---|---|
| 24 h | ifs_ens_mean | 15.34 | 11.76 | graphcast (11.49) |
| 48 h | graphcast | 14.42 | 12.45 | graphcast |
| 72 h | ifs_ens_mean | 15.24 | 13.85 | graphcast (11.80) |
| 96 h | graphcast | 16.03 | 12.71 | graphcast |
| 120 h | graphcast | 15.22 | 14.83 | graphcast |

**The train ranking and the test ranking disagree at 24 h and 72 h.** On a
3–4 day test set that is exactly what sampling noise looks like, and it is
the strongest single piece of evidence that these RMSE differences are not
yet resolvable. It also means the honest single-source baseline that later
steps must beat is the middle column, not the best number in the table.

---

## 2. RMSE is not the verdict, and here is the measurement that shows it

The research brief's section 1.3 caveat applies directly: RMSE rewards
**smooth, MSE-trained output**. A model that hedges — spreading a
100 mm event over a wider, lighter area — scores better on RMSE while being
*less* useful for a warning. Blends must therefore also be judged on CRPS,
heavy-rain POD/ETS and reliability.

That is not a theoretical hedge here. It is visible in this very run:

**POD at 115.6 mm (IMD "very heavy rainfall"), test split:**

| Lead | GraphCast | HRES | IFS-ENS |
|---|---|---|---|
| 24 h | **0.000** | 0.118 | 0.118 |
| 48 h | **0.000** | 0.027 | 0.000 |
| 72 h | **0.000** | 0.167 | 0.000 |
| 96 h | **0.000** | 0.000 | 0.000 |
| 120 h | **0.000** | 0.000 | 0.000 |

**GraphCast — the RMSE winner at every lead from 24 to 96 h — detects none
of the very-heavy-rain events, at any lead.** HRES, which has the worst RMSE
in the whole table and which Tier 1 assigns zero weight in 25 of 30 cells,
is the only source that detects any of them.

At 204.5 mm ("extremely heavy"), POD is **0 for every source at every
lead**. Nothing in this system currently detects that category at all.

So the two findings in section 1 and section 2 are the same finding seen
from two sides: optimising the leaderboard metric selects the source that is
least able to warn anyone. This is the argument for tail repair (steps
11–12) and for scoring extremes separately (step 04), and it is why "lowest
RMSE" is not allowed to be this project's headline.

---

## 3. The sources are not independent: N_eff ≈ 1.1 of 3

Computed on the **train split only** — the correlation structure is an input
to how the blend is built, so measuring it on test days would leak the test
set into the design. Errors are forecast minus IMD observation, in mm.

Domain-wide, per lead:

| Lead | GC–HRES | GC–IFS | HRES–IFS | Mean | N_eff | Participation ratio |
|---|---|---|---|---|---|---|
| 24 h | 0.800 | 0.904 | 0.891 | 0.865 | 1.099 | 1.200 |
| 48 h | 0.730 | 0.886 | 0.830 | 0.815 | 1.140 | 1.283 |
| 72 h | 0.639 | 0.850 | 0.771 | 0.753 | 1.197 | 1.395 |
| 96 h | 0.619 | 0.838 | 0.765 | 0.740 | 1.209 | 1.420 |
| 120 h | 0.691 | 0.885 | 0.787 | 0.788 | 1.165 | 1.331 |

Across all 30 lead × region cells, N_eff stays in **1.057 – 1.298**.

**Three sources are worth about 1.1 independent ones.** Averaging correlated
errors cancels almost nothing, which is the mechanical reason section 1's
blends fail to beat their own best member: there is very little diversity
for a mean to exploit. The one encouraging trend is that independence
*grows* with lead time (N_eff 1.10 → 1.21 from 24 h to 96 h), so whatever
value blending has is at longer leads, where the sources finally start
disagreeing.

This is the measurement that motivates going after genuinely independent
members — NEPS-G (steps 03, 08), GenCast/FuXi (step 22) — rather than
re-weighting the three that are already nearly the same forecast. It is also
why step 09 fits weights against the error *covariance* instead of
source-by-source skill.

### Why HRES gets zero weight in 25 of 30 Tier 1 cells

`results/tier1_regional_weights.csv` gives HRES a weight of exactly 0 in 25
of the 30 region × lead cells. That is **collinearity, not uselessness**.

HRES and IFS-ENS are both ECMWF's own model — the deterministic run and the
ensemble mean of the same system — and their errors correlate at **0.765 to
0.891 domain-wide**, up to **0.933** in individual regions (the minimum over
any region × lead cell is still 0.66). `fit_region_weights` is unconstrained
OLS with negatives clipped to zero, so when two predictors carry nearly the
same information, the fit keeps the one that fits marginally better and
drives the other to the boundary. A zero weight there means *"redundant
given the other sources in this fit"*, and nothing more.

Section 2 is the direct evidence for that reading: HRES is the **only**
source in the store that detects any 115.6 mm event, while carrying the
worst RMSE and a zero weight. An RMSE-fitted weight of zero and "this source
is useless" are not the same statement, and the slides must not conflate
them.

---

## 4. What this does and does not establish

**Established, on this store:**

- Raw GraphCast is not beaten by Tier 0 or Tier 1 at 24–96 h.
- The three sources' errors correlate at 0.62–0.90; N_eff ≈ 1.1 of 3.
- HRES's zero Tier 1 weight is explained by its correlation with IFS-ENS.
- GraphCast detects no 115.6 mm events; no source detects any 204.5 mm ones.

**Not established, and not to be claimed:**

- That GraphCast is "the best model". The train-chosen baseline disagrees
  with the test ranking at 2 of 5 leads.
- Any ranking at all, at significance. There are 3–4 test days per lead and
  no confidence intervals yet (step 04).
- That blending cannot work. It has not been tested against genuinely
  independent members, on enough days, on metrics that reward warnings
  rather than smoothness.

## Reproducing

```bash
python scripts/run_single_source_baseline.py
python scripts/run_independence_diagnostic.py
```

Both read `data/baseline_2020_jjas.zarr` (not committed; see `.gitignore`)
and write into `results/`.

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
