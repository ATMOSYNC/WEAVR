# Step 21: Self-healing weights (drift → adaptation)

**Type**: method + evaluation + operations prompt.
**Plan items**: D2 part 2, and pre-registered H11.
**Depends on**: step 07 (daily history for a sequential replay) and step 10
(the live pipeline, and whatever real days it has accumulated).

## Goal

Today `weavr.drift` can *detect* a model shift, but nothing *responds* to
one. The daily pipeline also admits a mismatch: its Tier 1 weights were fit
on GraphCast, while AIFS fills that slot live.

Close the loop with **online weight updating**:

- Start from the Tier 1 weights as a prior.
- Update each region × lead's weights daily from verified losses, using an
  exponentially weighted / Hedge-style rule with a **fixed-share** floor
  (Herbster & Warmuth 1998), so a down-weighted model can recover after an
  upgrade.
- Thorey, Mallet & Baudin (2017, QJRMS) apply online learning with the
  CRPS to ensemble forecasting; cite them as the method lineage.

Evaluate it honestly under H11 on a **sequential replay** of a held-out
season. The live pipeline will have only a few weeks of real history, so
the historical daily data is the real test.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Add src/weavr/online.py (pure, tested):
   - OnlineWeights state per (region, lead): weights, learning rate eta,
     fixed-share alpha.
   - update(state, losses_by_source) -> new state:
     w_i <- w_i * exp(-eta * loss_i), normalise, then apply fixed share
     (w <- (1 - alpha) * w + alpha / N).
   - A missing source for a day leaves its weight unchanged, and today's
     blend renormalises via weavr.renormalize (reuse it; don't
     re-implement it).
   - drift_response(state, drift_result): when weavr.drift fires for a
     source, temporarily raise alpha / eta for that region x lead, with a
     documented decay back to baseline.
   Tests: a consistently best source converges to the top weight; after
   a synthetic switch, fixed share lets the new best recover within a
   bounded number of days; a missing source is handled; weights stay
   non-negative and sum to 1.

2. Sequential replay evaluation: add scripts/run_online_weights_replay.py.
   Tune eta and alpha on 2018 (a sequential pass over days), then replay
   2020 day by day with a prior from 2018-fit Tier 1 weights, updating
   only on information available that day. Verification is
   retrospective: a lead-L forecast's loss is known only once its IMD day
   has passed. Compare against static Tier 1 weights on the same days.
   Metrics: RMSE, and CRPS where applicable. CIs via step 04. Compute
   H11 per docs/preregistration.md.

3. Synthetic-upgrade test, clearly labelled SYNTHETIC in every output:
   inject a systematic change into one source mid-2020 (e.g. multiply
   GraphCast by 0.7 from day 60). Show:
   a. when weavr.drift fires (if it does)
   b. how the online weights move
   c. blend error recovery time vs static weights
   Save the figure data to results/online_synthetic_upgrade.csv.

4. Operations: wire into scripts/run_daily_pipeline.py behind a flag
   (--online-weights, default OFF until H11 passes):
   - load results/online_weights_state.json
   - blend with the current weights
   - after the retrospective verification step, apply update(), and
     drift_response() if drift fired
   - write the new state
   Test the end-to-end daily step with fakes. If step 10's scheduled
   workflow exists, extend it only if H11 passed AND the user confirms in
   chat.

5. Dashboard: add a "Weights over time" mode to the weight-map view
   (weightMap.js; keep its module prefix) that plots the online weights per
   region across the replay days, with the synthetic-upgrade scenario as a
   toggle clearly labelled "synthetic test". Add API endpoints with
   tests. Browser-verify on a fresh port.

6. docs/online-weights-results.md: the method, tuning, replay results with
   CIs, the H11 verdict, the synthetic-upgrade behaviour, and what the real
   live history (step 10) shows so far, with day counts. README
   paragraph.
```

## Done when

- `weavr.online` is tested.
- The sequential replay against static weights has CIs and an H11
  verdict.
- The synthetic-upgrade test demonstrates, or honestly fails to
  demonstrate, recovery.
- The pipeline has a flagged online mode.
- The dashboard shows weights over time.
