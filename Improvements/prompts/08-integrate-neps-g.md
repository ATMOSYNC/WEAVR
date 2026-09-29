# Step 08: Put India's own ensemble (NEPS-G) into the blend

**Type**: implementation + evaluation prompt.
**Plan items**: A3 part 2.
**Depends on**: step 03 (HEPPI date map), step 06 (2018 stores and
builders) and step 07 (v2 tiers and scorecard to compare against).

## Goal

Add **NCMRWF NEPS-G** (23 members, from HEPPI, now dated) as a fifth
rainfall source for the 24 h lead in JJAS 2018 and 2019. Measure three
things:

1. **What it adds:** its independence from the other sources (N_eff) and
   its weight and contribution.
2. **Whether it improves the blend:** pre-registered H4.
3. **How WEAVR compares with the published Indian benchmark:** HEPPI's
   `NCMRWF_EMOS_forecast.nc` and `NCMRWF_UQM_forecast.nc`
   (Angus et al. 2024) on identical days. That is H5.

## Why this is the most persuasive data move

- The jury may include NCMRWF. A blend that includes, and can be measured
  against, *their* ensemble answers "why should we adopt this?" directly.
- NEPS-G is based on the Met Office Unified Model and uses NCMRWF's own
  analysis. That is a different lineage from IFS/ERA5, which GraphCast
  inherits. Hypothesis: lower error correlation and higher N_eff. **It is a
  hypothesis; measure it.**
- Beating (or honestly matching) a peer-reviewed EMOS post-processing of the
  same ensemble on the same days is much stronger than beating an
  equal-weight mean.

## Constraints to respect

- **Lead:** NEPS-G in HEPPI is day-1 only, assumed per Angus et al. 2024
  (step 03 flags this). So NEPS-G enters **only the 24 h blend**.
- **Source availability:** GraphCast has no 2019 JJAS. Run two
  configurations:
  - (A) 2018 + 2019, with sources hres, ifs_ens(_mean) and neps_g present
    in both years (LOYO 2018 ↔ 2019)
  - (B) 2018 only, adding graphcast (block split, labelled as such)
- **Fairness of the HEPPI comparison:** HEPPI-EMOS's training protocol is
  not known from the files. Read `HEPPI/example_EMOS_fit.R` to characterise
  its training window, and state whether its forecasts could be in-sample
  for these days. If they might be, say the comparison favours HEPPI.
- **Licence:** commit scores only. No HEPPI-derived grids go in the repo or
  on the dashboard unless the user confirms the licence.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Data for 2019. Build daily JJAS 2019 stores for hres and ifs_ens (and
   ifs_ens_mean) plus IMD, using step 06's per-(source, year) builders.
   There is no GraphCast for 2019; record that as expected in the manifest.
   Measure cost first (convention 8).

2. Load NEPS-G via weavr.data.heppi_reference with the step-03 date map:
   variant "orig" as the source, and "uqm"/"emos" as the benchmarks.
   Restrict to confirmed JJAS 2018/2019 dates. Set member_dim="member".
   HEPPI values are already mm/day; verify units against IMD before use.
   Align to IMD days using the step-03 validity check.

3. Add scripts/run_neps_g_experiment.py, writing results/neps_g_*.csv and
   per-day outputs (step 04 helper), for configurations (A) and (B):
   a. Independence: weavr.independence error correlations and N_eff, with
      and without neps_g (train folds only).
   b. Deterministic: Tier 1 regional weights with vs without neps_g
      (reuse fit_region_weights and blend_with_region_weights).
   c. Probabilistic: EMOS-CSG on the NEPS-G ensemble (weavr.emos), and BMA
      with neps_g as an ensemble component (weavr.bma). Reuse
      run_tier2_hierarchical_baseline's functions; don't re-implement
      fitting.
   d. Benchmark head-to-head on IDENTICAL days and cells:
      - raw NEPS-G ensemble
      - HEPPI-UQM
      - HEPPI-EMOS
      - WEAVR EMOS/BMA with and without neps_g
      - IFS-ENS EMOS
      - climatology
      Metrics: CRPS, Brier at 7.5/64.5/115.6 mm, SEEPS (ensemble mean),
      reliability tables, PIT. CIs via step 04.

4. Verdicts: compute H4 and H5 per docs/preregistration.md; update
   results/preregistration_verdicts.csv.

5. Write docs/neps-g-results.md:
   - the data (how NEPS-G was recovered and dated, and the day-1
     assumption)
   - the independence numbers
   - weights with vs without NEPS-G
   - the benchmark table
   - H4/H5 verdicts
   - the HEPPI-EMOS fairness characterisation
   Plain language; a loss is reported as prominently as a win. README
   paragraph.

6. Operational note: the daily pipeline has no live NEPS-G feed (none is
   public). State this in the doc. NEPS-G is an evaluation and
   hindcast-blend result, not a live input.

7. Route one decision via AskUserQuestion: whether NEPS-G-derived results
   may appear on the (eventually public, step 20) dashboard as scores only
   (recommended, since licence risk is minimal), as scores + weights, or
   not at all until the HEPPI authors confirm the licence.

8. Outside the repo: if H4 or the configuration (A)/(B) results support
   it, update ppt/content's NEPS-G wording to be accurate (e.g. "includes
   NCMRWF NEPS-G at day 1 for 2018-2019 hindcasts"). List the change in
   the PR.
```

## Done when

- NEPS-G is a dated, working 24 h source in configurations (A) and (B).
- Its N_eff contribution, weights and EMOS/BMA results are measured.
- The head-to-head against HEPPI-UQM and HEPPI-EMOS exists with CIs and a
  fairness note.
- H4 and H5 have verdicts.
- The licence handling decision is recorded.
