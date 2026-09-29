# Step 24 (optional): Temperature and heat-wave blending

**Type**: scope check, then implementation + evaluation.
**Plan items**: C6.
**Depends on**: steps 05–06 (the daily stores already contain
`2m_temperature` for graphcast, pangu, hres and ifs_ens_mean).

## Goal

The research brief's framing of the problem names **rainfall, temperature
and wind**, but WEAVR blends only rainfall. Temperature is the cheapest
addition:

- the data is already in the stores
- **Pangu finally contributes** (it has temperature but no rainfall)
- Gaussian EMOS (NGR) has a closed-form CRPS, far simpler than rain

Deliver a blended daily-maximum temperature forecast plus heat-wave
probabilities from **IMD's own heat-wave criteria**.

## Caveats to check, not paper over

- **Scope first.** Confirm temperature against the official PS 26081 text.
  If the PS doesn't ask for it, stop after item 1 and report back.
- **Ground truth:** IMD gridded Tmax/Tmin come via `imdlib` (`var_type`
  "tmax"/"tmin") at **1°**, not 0.25°. Coarsen the forecasts to IMD's 1°
  grid. Never refine IMD's field to 0.25°.
- **Tmax proxy:** WeatherBench 2's `2m_temperature` is instantaneous at
  synoptic hours, while Indian daily maxima peak around 09 UTC. The daily
  max over 6-hourly steps under-samples the peak (expect a cold bias). NGR
  learns the offset, but also check whether any archive offers a
  max-temperature variable, e.g. HRES `maximum_2m_temperature…`, and
  prefer it if it exists.
- **Season:** the heat-wave season is mainly April–June. Early June is in
  JJAS, and both GraphCast windows cover April–June of 2018 and 2020. Adding
  April–May needs extra builds; measure the cost first.
- **IMD heat-wave criteria** (verify against IMD's published definition
  before coding):
  - Tmax ≥ 40 °C in the plains (≥ 37 °C coastal, ≥ 30 °C hilly), **and**
    a departure from normal ≥ 4.5 °C
  - or Tmax ≥ 45 °C
  The "normal" needs an IMD Tmax climatology, and plains/coastal/hilly
  needs a station-type mask. Approximate both from gridded data and
  document the approximation.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Scope check: find the official PS 26081 text (ask the user if it isn't
   in the repo or research/) and quote the line covering temperature. If
   it's absent, stop here and report.

2. Data:
   - Check imdlib tmax/tmin availability for the years needed (a live
     test).
   - Build an IMD Tmax climatology store for "normal"
     (e.g. 2006-2020 excluding the test year, via
     build_seeps_climatology.py's pattern).
   - Decide the season (JJAS only vs adding April-May; measure the
     extra build cost and route via AskUserQuestion if it's more than a
     few hours).
   - Coarsen forecasts to IMD's 1 degree grid with a tested, conservative
     area-average helper in weavr.grid.

3. Add src/weavr/ngr.py (pure, tested):
   - gaussian_crps(mu, sigma, y) in closed form, tested against numerical
     integration.
   - fit_ngr(ensemble_or_multi_source, obs, train_mask) with
     mu = a + sum_i b_i * f_i and sigma^2 = c + d * s^2, fitted by
     minimising the mean CRPS, with sensible positivity constraints and
     the project's fallback convention.
   - heatwave_probability(mu, sigma, normal, category_mask) implementing
     the verified IMD criteria.

4. Add scripts/run_temperature_tier.py (LOYO): raw sources (graphcast,
   pangu, hres, ifs_ens_mean) vs the equal mean vs Tier 1-style regional
   weights (reuse fit_region_weights) vs NGR. Metrics: RMSE, bias, CRPS,
   PIT, and the heat-wave Brier score and SEDI, with CIs. Write
   results/temperature_*.csv and per-day outputs. Record the cold bias of
   the 6-hourly Tmax proxy explicitly.

5. Dashboard: a "Heat-wave probability" view (a new prefixed module) with
   P(heat wave) at 1 degree, IMD colours where IMD defines heat-wave
   warning colours (verify; otherwise use a clearly labelled sequential
   scale), and a caption stating the 1 degree resolution, the Tmax proxy
   and the criteria approximation. API with tests. Browser-verify.

6. docs/temperature-results.md, and README: WEAVR now covers rainfall and
   temperature. Wind remains out of scope: there is no IMD gridded wind,
   and ground truth would be ERA5, which this project avoids for
   verification.
```

## Done when

- Temperature scope is confirmed against the PS.
- NGR is tested.
- The temperature tier and heat-wave probabilities are evaluated under
  LOYO with CIs, with the caveats stated.
- A heat-wave view is live.
