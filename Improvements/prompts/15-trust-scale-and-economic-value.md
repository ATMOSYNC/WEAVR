# Step 15: The "trust scale" and economic value

**Type**: implementation + evaluation prompt (science and API; the UI is
step 16).
**Plan items**: C2 + C4.
**Depends on**: step 07 (v2 data, FSS already implemented) and step 12
(the best probabilistic forecasts).

## Goal

Two additions that translate skill into something a forecaster or official
can act on:

1. **Trust scale.** For each lead × threshold, find the smallest spatial
   scale at which WEAVR's rainfall forecast is "useful", using Roberts &
   Lean (2008)'s criterion `FSS ≥ 0.5 + f₀/2`, where f₀ is the observed
   event fraction. Add **neighbourhood probabilities**,
   `P(≥ t within r km)`, which avoid the double penalty that pixel scores
   give near-misses.
2. **Relative economic value** (Richardson 2000). `V(C/L)` curves show
   *who* benefits from WEAVR's probabilities, and by how much, compared with
   raw models: low cost/loss users (e.g. a farmer delaying spraying) versus
   high (e.g. an evacuation).

## Formulas to implement exactly (and cite)

- **Useful FSS criterion:** `FSS_useful = 0.5 + f₀/2`, from Roberts & Lean
  (2008).
- **Relative economic value** for cost/loss ratio α, base rate ō, hit rate
  H and false-alarm rate F:
  `V = [min(α, ō) − F·α·(1−ō) + H·ō·(1−α) − ō] / [min(α, ō) − ō·α]`
  For probabilistic forecasts, take the envelope (the maximum V over
  probability thresholds) at each α. Verify this against Richardson (2000)
  before coding; it is the standard form.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Extend src/weavr/verify.py (tested, with formula docstrings):
   - useful_scale(forecast, obs, threshold, neighborhood_sizes) -> the
     smallest size meeting FSS >= 0.5 + f0/2 (NaN if none does), reusing
     fss(). Also return the FSS-vs-size curve.
   - neighbourhood_exceedance_probability(ensemble, threshold,
     radius_cells, member_dim): the fraction of members with any cell >= t
     within the radius. Use a max filter per member, then average across
     members.
   - relative_economic_value(hit_rate, false_alarm_rate, base_rate,
     cost_loss_ratios), plus
     value_envelope(prob_forecast, obs_binary, prob_thresholds,
     cost_loss_ratios).
   Tests: a perfect forecast has V = 1 wherever the event is possible;
   climatology has V = 0 at alpha = base rate; a useful scale of 1 for a
   perfect forecast; a known synthetic displacement gives the expected
   useful scale.

2. Add scripts/run_decision_value.py (LOYO, pooled):
   - Useful scale per lead x threshold (7.5 / 64.5 / 115.6) for raw
     GraphCast, raw IFS-ENS mean, the Tier 1 blend and WEAVR's best
     probabilistic method (via its median or mean). Convert cells to km
     at the domain-mean latitude, and state the conversion.
   - Neighbourhood probabilities at r = 1, 2 and 4 cells for the best
     ensemble method and raw IFS-ENS: Brier score and reliability.
   - Value curves at thresholds 64.5 and 115.6, alpha on a log grid from
     0.01 to 0.9, for: raw GraphCast (deterministic), raw IFS-ENS
     probabilities, NEPS-G (24 h, step 08 data), WEAVR best, and
     climatology. Bootstrap CIs on V at a few representative alpha
     values.
   Write results/useful_scale.csv, results/neighbourhood_probability.csv
   and results/economic_value.csv.

3. User annotations for the value chart: label alpha bands with example
   user types ("low cost/loss, e.g. a farmer delaying pesticide
   spraying"; "high cost/loss, e.g. a precautionary evacuation"), clearly
   marked ILLUSTRATIVE. Do not invent monetary figures or cite cost
   numbers without a real source.

4. API: /api/useful-scale, /api/economic-value?threshold=&lead= (422 on
   bad params), reading the CSVs via pure dashboard/data_loading.py
   functions, with tests.

5. docs/decision-value-results.md: plain statements ("day-3 heavy-rain
   forecasts are useful at >= X km"; "for cost/loss ratios between A and
   B, WEAVR's probabilities are worth more than raw IFS-ENS's, CI ..."),
   including where WEAVR adds nothing. README paragraph.
```

## Done when

- Useful scale, neighbourhood probabilities and economic value are
  implemented, tested and computed under LOYO with CIs.
- The API serves them for step 16.
- The doc states the "trust scale" sentence for each lead from real
  numbers.
