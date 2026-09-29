# Step 14: District-level IMD colour warnings with a verified rule (science and API)

**Type**: decision + implementation + evaluation prompt. It has two real
user decisions: the boundary source and the rule's objective.
**Plan items**: C1 (science half). Step 17 builds the UI.
**Depends on**: step 04 (verification and CIs) and step 12 (real
`P(≥115.6)` and `P(≥204.5)` probabilities).

## Goal

IMD issues **district-wise, colour-coded** heavy-rain warnings. WEAVR
colours 0.25° pixels by a deterministic bin. Turn WEAVR's probabilities
into **district warnings**:

- per-district `P(≥64.5)`, `P(≥115.6)` and `P(≥204.5)`
- an **explicit probability→colour rule**, with thresholds fitted on train
  data only
- IMD-style spatial-distribution wording
- **out-of-sample verification of the colour calls themselves**
  (pre-registered H8)

## Things to get right

- **Boundaries.** An India map shown to a ministry jury must match the
  Survey of India's official depiction of the national boundary. Licences
  vary between sources (Survey of India releases, community datasets,
  global datasets whose depiction may differ). The district count also
  depends on vintage (census-2011 era vs current). This is the user's call.
- **The rule's objective is a real choice.** Options: maximise CSI/ETS of
  the colour calls; maximise economic value for a chosen cost/loss band
  (step 15); or match IMD's own historical warning frequency. Each yields
  different thresholds, so route it to the user.
- **Defining an observed district event.** With gridded obs, options
  include "any cell ≥ t", or "≥ X% of cells ≥ t". IMD's own
  spatial-distribution terms map onto the fraction of stations reporting:
  isolated (≤25%), scattered (26–50%), fairly widespread (51–75%) and
  widespread (76–100%). Use the fraction of district cells as the gridded
  analogue, verify these category boundaries against an IMD source, and
  document the choice.
- **No new heavy dependency without need.** Point-in-polygon for about
  18k cell centres can be done in numpy (ray casting). Only add shapely if
  that is genuinely insufficient, and say why.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Boundary source: route via AskUserQuestion. First research 2-4 real
   candidates for India district boundaries. For each, report: licence,
   vintage/district count, whether its national boundary depiction matches
   the Survey of India's official map, file size, and attribution
   requirements. Recommend the Survey of India-compliant option with the
   clearest licence. Don't download or commit anything until the user
   picks.

2. After the user picks:
   - Simplify the geometry (topology-preserving; document the tolerance)
     to a committed dashboard/data/districts.geojson, ideally under 2 MB,
     with dashboard/data/DISTRICTS_LICENSE.md stating the source, licence
     and attribution text.
   - Add src/weavr/districts.py:
     - assign_cells_to_districts(lat, lon, geojson) -> (lat, lon) int
       array of district ids (-1 outside); numpy ray casting; tested on a
       synthetic square and a concave polygon
     - district_probability(cell_probs, assignment, method="max" |
       "area_mean")
     - district_exceedance_fraction(cell_values, assignment, threshold) ->
       the fraction of cells >= t, plus IMD's spatial-distribution
       category (isolated / scattered / fairly widespread / widespread),
       citing the IMD source for the boundaries

3. The colour rule: route the objective via AskUserQuestion (CSI/ETS vs
   economic value at a chosen C/L vs matching IMD's historical warning
   frequency). State that every option is fitted on train folds only.
   Then implement in weavr/districts.py:
     fit_colour_rule(train_district_probs, train_district_events,
                     objective) -> ColourRule(p_yellow, p_orange, p_red)
     apply_colour_rule(rule, district_probs) -> colour per district
   Rule shape:
     Red if P(>=204.5) >= p_red,
     elif Orange if P(>=115.6) >= p_orange,
     elif Yellow if P(>=64.5) >= p_yellow,
     else Green.
   Test threshold fitting on synthetic data with a known optimum.

4. Add scripts/run_district_warnings.py (LOYO, pooled; best probabilistic
   WEAVR method from steps 12/13):
   - Choose the aggregation (max vs area_mean) on train; report both.
   - Fit the colour rule on train and apply it on test.
   - Verify district colours: POD/FAR/CSI for ">= Orange" and for "Red",
     plus a contingency table per colour, with CIs.
   - Baselines, using the same fitted procedure: (i) the best single raw
     ensemble (IFS-ENS member-counting probabilities); (ii) a
     deterministic colour from raw GraphCast. Compute H8.
   - Write results/district_warnings_verification.csv and
     results/district_colour_rule.csv (fitted thresholds per lead and
     fold, plus the final all-data fit used for display).

5. Export and API:
   - Extend export_dashboard_example_grids.py (or add
     scripts/export_district_warnings.py) to write
     dashboard/data/example_district_warnings.json: per lead, for each
     district: id, name, state, the three probabilities, colour, spatial
     category, and n_cells.
   - Add /api/district-warnings?lead= (422 on bad lead) and
     /api/districts (the GeoJSON), with tests.

6. docs/district-warnings-results.md: the boundary source and licence, the
   aggregation choice, the rule objective (the user's decision), the
   fitted thresholds, the verification with CIs, and the H8 verdict. State
   plainly that these are demonstration warnings, not official IMD
   warnings. README paragraph.
```

## Done when

- User-approved, licence-documented district boundaries are committed.
- `weavr.districts` is tested.
- A train-fitted colour rule is verified out-of-sample, with H8's verdict.
- The API serves district warnings for step 17's view.
