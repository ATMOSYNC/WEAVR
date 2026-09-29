# Step 17: District warnings view and the "why this forecast?" explainer

**Type**: frontend + small science prompt. Needs real browser
verification.
**Plan items**: C1 (UI) + C5.
**Depends on**: step 14 (district warnings API and boundaries) and step 09
(per-model contributions).

## Goal

The view a forecaster or disaster manager would actually use:

1. **District warnings map.** An India district choropleth in IMD colours
   for the selected lead, drawn as a GeoJSON fill layer on the existing
   OpenStreetMap basemap (`BasemapMap`, from `/api/districts`), with the
   official national outline on top when it is configured. It shows the
   boundary attribution from `DISTRICTS_LICENSE.md` next to the OSM one.
   A sortable district list beside it gives keyboard and screen-reader
   access, and is also the fallback when the basemap or WebGL is
   unavailable.
2. **District card on click:**
   - `P(≥64.5 / ≥115.6 / ≥204.5)` and the colour
   - IMD-style wording (e.g. "Heavy rain likely at scattered places")
   - **why:** each source's forecast over the district, its weight and its
     contribution (weight × forecast)
   - the ensemble spread and inter-model disagreement
   - the rain bin, and whether a fallback or GPD tail was used
   - a **confidence rating** (High / Medium / Low)
3. **A model-disagreement layer** toggle on the existing blended-map view.

## The confidence rating must be calibrated in Python, not invented in JS

- The rating comes from spread and inter-model disagreement. It must be
  **calibrated on history**. Bin district-days into terciles of a
  spread + disagreement score on train folds, and show that the observed
  error actually increases from High to Low. That is the spread–error
  relationship; it is not decoration.
- If it doesn't, because the spread–error relationship is weak, say so in
  the card ("confidence rating not informative at this lead") rather than
  showing a meaningless badge.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Science: add scripts/run_confidence_calibration.py (LOYO):
   - Define score = standardized ensemble spread + standardized
     inter-model spread, per district-day.
   - Fit tercile cut-points on train, and report the test
     mean-absolute-error (and CSI for >= Orange) per tercile, with CIs.
   - Write results/confidence_calibration.csv with the cut-points per
     lead and an is_informative flag (true only when the error is
     monotone across terciles AND the High-vs-Low difference has a CI
     excluding 0).
   - Test the pure pieces.
   Extend the district export and /api/district-warnings with the
   per-district confidence rating, per-source district-mean forecasts,
   weights, contributions, spread, disagreement, bin and method flags.
   Update the API tests.

2. Frontend: add dashboard-web/js/charts/districtWarnings.js
   (module-prefixed globals, e.g. districtWarningsEl):
   - Map: reuse `BasemapMap.create` (Web Mercator, correct alignment,
     attribution, offline tiles); add the districts as a GeoJSON source
     with a fill layer coloured from the warning colour and a thin
     outline layer, both below the boundary layer. MultiPolygons and holes
     come free. Don't write a separate projection. Geography is on by
     default here (unlike the two grid views), since districts without
     context are hard to read.
   - Fill: IMD colours from /api/colors (reuse getColors(); never hard-code
     the palette).
   - Accessible: the map canvas can't take per-district focus, so the
     district list is the keyboard path (each row: name, colour,
     probabilities; Enter opens the card and highlights the district on
     the map). Map click opens the same card.
   - District card panel (a side panel on desktop, below the map at mobile
     width) with the contents in "Goal". The confidence badge shows "not
     informative at this lead" when is_informative is false.
   - Caption: "Demonstration warnings produced by WEAVR; not official IMD
     warnings", the colour-rule objective and thresholds from
     results/district_colour_rule.csv, and the H8 verification headline
     from docs/district-warnings-results.md (pass or fail, as it is).
   Register it in main.js VIEWS (needsLead) and the nav; add a script tag.

3. Blended map: add a "Model disagreement" toggle, rendering the
   inter-model standard deviation (mm) as a separate canvas layer with its
   own sequential legend. It draws on the same `BasemapMap` (`setCells`)
   and honours the existing "Show geography" toggle. Extend the blended-map export and API to include
   it (422-validated, tested).

4. Browser verification on a fresh port:
   - all leads render
   - a random sample of about 10 districts: card contents match the API
     JSON exactly (check via JS)
   - keyboard focus works
   - the disagreement toggle renders and its legend is correct
   - no console errors
   - district fills line up with the basemap coastline and the official
     outline (spot-check 4 coastal districts; report any offset)
   - the district list and card work with the basemap absent (rename the
     tile file temporarily) and with WebGL unavailable
   - mobile width (375 px): map fits, card stacks, no horizontal scroll
   - map render time for all districts (target under 1 s on a laptop;
     report the real time)
   Screenshots in the PR.

5. Docs: a README paragraph, and docs/district-warnings-results.md gains a
   "Dashboard" subsection. One PR.
```

## Done when

- The district map is built on the OpenStreetMap basemap, with a list
  fallback.
- The district warnings view, with click and keyboard-accessible cards,
  is live and verified in a real browser.
- The confidence rating is calibrated from history (or honestly marked
  not informative).
- The blended map has a disagreement layer.
- The caption states the demonstration-only status and the real
  verification result.
