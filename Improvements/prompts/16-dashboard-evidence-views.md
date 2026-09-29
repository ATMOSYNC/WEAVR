# Step 16: Dashboard evidence views (scorecard, independence, reliability, value and trust scale)

**Type**: frontend prompt (views over already-computed results). Needs real
browser verification.
**Plan items**: A4, B1, C2 and C4 (the UI halves).
**Depends on**: step 04 (scorecard), step 09 (independence and Shapley
API), and step 15 (useful-scale and value API).

## Goal

Show WEAVR's strongest differentiators, its **evidence**, in the
dashboard, next to the existing 4 views:

1. **Scorecard.** An ECMWF-style grid: rows are metric × threshold,
   columns are lead. Each comparison (vs best single member, vs
   climatology, vs Tier 0) shows ▲/▼, coloured only when significant.
   Hover shows the estimate, CI and DM p-value. A panel lists each
   pre-registered claim with PASS / FAIL / INSUFFICIENT / NOT YET TESTED,
   failures included.
2. **Independence.** An error-correlation heat-map by lead, an N_eff-by-lead
   line (with NEPS-G at 24 h, if step 08 ran), and a Shapley contribution
   chart per region × lead. This is the explanation of the weight map.
3. **Reliability.** Reliability diagrams (with counts) and PIT/rank
   histograms by lead × threshold for the probabilistic methods.
4. **Value & trust scale.** Economic value curves with the illustrative
   user bands, and the useful-scale-by-lead chart.

Also update the existing **skill-trends** view to plot the raw single
sources as dashed reference lines. That is F2's honesty, made visible.

## Conventions carried from the frontend migration (all mandatory)

- Plain HTML/CSS/vanilla JS, with hand-rolled SVG/canvas
  (`docs/frontend-migration-scope.md`). No charting library.
- **One self-contained module per view** under `dashboard-web/js/charts/`,
  with **module-prefixed globals** (e.g. `scorecardEl`,
  `SCORECARD_SVG_NS`). The frontend step-4 collision bug broke the whole
  page when two modules shared `el` / `SVG_NS`.
- SVG sizing through CSS (`svg.style.width/height`), never
  `setAttribute("height", "auto")`, which was the frontend step-3 bug.
- Each view states its real finding in a caption, including negatives.
  Reuse the exact wording from the corresponding results doc; don't
  paraphrase it into something stronger.
- The API reads committed CSVs through pure `dashboard/data_loading.py`
  functions, with tests. Invalid parameters return 422, never a silent
  default.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. API: confirm or add read-only endpoints for everything these views need
   (/api/scorecard, /api/preregistration, /api/independence,
   /api/contributions, /api/reliability, /api/useful-scale,
   /api/economic-value), each with tests (valid shape, 422 on bad
   params). Where a results CSV lacks something a view needs (e.g.
   reliability tables per method), add it to the producing script rather
   than computing it in JS.

2. Build the views as four new modules: scorecard.js, independence.js,
   reliability.js and decisionValue.js. Register them in main.js's VIEWS
   registry with the right needsLead / needsMetric / needsThreshold flags
   (add a threshold selector if needed, following renderLeadSelector's
   pattern), add sidebar nav items, and add script tags to index.html.
   Scorecard details:
   - colour-blind-safe up/down encoding (shape AND colour)
   - a legend explaining "coloured = 95% CI excludes 0"
   - the claims panel lists every claim from /api/preregistration, with
     failed claims shown equally prominently
   Independence details:
   - heat-map cells labelled with values
   - N_eff line with a reference line at N (the nominal model count)
   - Shapley as stacked bars per region
   - caption explains HRES's zero weight with the real correlation
     numbers from docs/independence-results.md

3. Skill trends: add raw single sources (from
   results/single_source_baseline.csv) as dashed lines with legend
   entries, and update the caption to the v2 finding from
   docs/v2-evidence-base-results.md.

4. Verify in a real browser. Start uvicorn on a fresh, never-used port,
   then for every new view:
   - every lead / threshold / metric combination renders
   - no console errors: check via direct JS, e.g. typeof ScorecardView !==
     'undefined', and query the rendered SVG node counts
   - captions match the docs' wording
   - layout holds at 375 px width (no horizontal page scroll)
   Screenshot one state per view for the PR description.

5. README: extend the "Frontend migration" section with the new views, and
   mention them in the docs of steps 04/09/15. One PR.
```

## Done when

- Four new evidence views and the updated skill-trends view are live.
- Each is verified in a real browser across all parameter combinations,
  with no console errors and mobile-width layout checked.
- Captions reproduce each results doc's real finding, including
  negatives.
