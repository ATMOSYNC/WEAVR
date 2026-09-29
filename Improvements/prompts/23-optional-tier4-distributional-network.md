# Step 23 (optional): Tier 4 — a learned distributional blend (the "AI" in AI–NWP blending)

**Type**: method + evaluation prompt, with a pre-registered go/no-go.
**Plan items**: B4.
**Depends on**: step 04 (significance) and step 07 (v2 data with two
seasons). It benefits from steps 08 and 22 (more sources as features).

## Goal

A judge will ask "where is the AI in *your* system?" Tier 3's gating model
was declined in Phase 5 because the training set had 2 "active" days per
lead. After v2 there are about 240 daily samples × about 18.5k land cells
per lead, enough for a **small** model with honest cross-validation.

Build a **distributional regression network (DRN)** after Rasp & Lerch
(2018, MWR):

- A small MLP maps per-cell features to **CSGD parameters**.
- It is trained by minimising WEAVR's **own closed-form `csgd_crps`**,
  ported to a differentiable framework, so the training loss *is* the
  evaluation metric.
- Judge it with the same scorecard and a pre-registered criterion:
  **H12**, added in this step, before training.

## Why a DRN (and its risks)

- It outputs a full predictive distribution (compatible with every
  probabilistic product: district warnings, CAP, value curves), not just a
  point forecast.
- It naturally uses features the linear combiners can't: inter-model
  spread, location, day of season, MJO phase (already in `data/`), and
  elevation.
- **Main risk: overfitting**, because neighbouring cells are strongly
  correlated, so the effective sample size is far below the raw row
  count. Mitigate with year-held-out validation, a small network, early
  stopping on a held-out *block of days* (never random cells), and
  weight decay.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Dependency decision: route via AskUserQuestion. Options: add PyTorch
   (CPU) as a new optional group `ml` (the mature option; large install)
   vs JAX vs a numpy-only implementation with hand-derived gradients (no
   dependency, much more code). Recommend a torch `ml` group. Keep it out
   of the `dev` group, and make its tests importorskip so CI stays light
   (the same pattern as fastapi in tests/test_dashboard_api.py).

2. Pre-register first. Add H12 to docs/preregistration.md as its own
   commit, before any training:
     "Tier 4 DRN beats the best of EMOS-CSG / BMA / the step-13 combined
     method on LOYO CRPS at >= 3/5 leads, with the CI excluding 0, AND is
     not worse on the Brier score at 64.5 mm."

3. Add src/weavr/drn.py:
   - A differentiable CSGD CRPS matching weavr.emos.csgd_crps. Test that
     the two agree to within 1e-6 on random parameters, and that
     gradients are finite, including at the censoring point.
   - Feature builder (pure, tested): per cell-day, each source's
     forecast, ensemble means and spreads, inter-model std, lat, lon,
     elevation (source it and cite it; if none is available offline, drop
     the feature and say so), region one-hot or embedding,
     day-of-season (sin/cos), lead, and MJO phase and amplitude from
     data/mjo_phase_*.zarr (extend to 2018 via
     scripts/fetch_omi_mjo_index.py if needed).
   - The model: an MLP of 2 hidden layers x 32-64 units, outputting
     (mean > 0, std > 0, shift <= 0) through softplus / negated-softplus
     links matching weavr.emos's parameterisation.
   - Training: one model per lead (or lead as a feature; justify it),
     Adam, weight decay, early stopping on a held-out contiguous block of
     train days, fixed seeds. Report train/val curves.

4. Add scripts/run_tier4_drn.py (LOYO, both folds, pooled):
   CRPS, Brier at IMD thresholds, PIT, reliability, tw-CRPS, and per-day
   outputs; compute the H12 verdict. Explainability: permutation
   importance per feature (overall and per region) on test folds, written
   to results/tier4_feature_importance.csv. Record CPU training time.

5. docs/tier4-drn-results.md: the architecture, features, training
   protocol, results with CIs, the H12 verdict (a no-go reported with the
   same prominence as a go, like Tier 3), and feature importance with a
   plain reading ("inter-model spread matters most in NE regions",
   whatever it truly shows). README paragraph.

6. If H12 PASSES: expose DRN probabilities as an option in the
   extreme-probability and district exports (a flag, not a silent default
   switch), and add the DRN to the scorecard view.
```

## Done when

- H12 is committed before training.
- `weavr.drn` is tested, including CRPS agreement with the closed form.
- LOYO results, the verdict and feature importance are reported
  honestly.
- The dependency decision is documented.
