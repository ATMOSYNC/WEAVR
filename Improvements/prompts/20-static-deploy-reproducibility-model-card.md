# Step 20: Public URL (static build), one-command reproducibility and a model card

**Type**: engineering + publishing prompt. Publishing is outward-facing and
needs explicit user confirmation.
**Plan items**: D3 + D4.
**Depends on**: steps 16–19 (every view and export that should go into the
static build).

## Goal

1. **Static build.** Every API response is a pure function of committed
   files, so pre-render them all to static JSON/XML. The whole dashboard
   then runs from any static host: a URL and QR code on the slides that a
   judge can open on a phone. It also removes the "VS Code Show Preview →
   404" problem for good.
2. **Reproducibility.** One command rebuilds stores (skipping existing
   ones via manifests), runs every results script in dependency order,
   and exports all dashboard data. A `--check` mode confirms committed
   results reproduce. Add a Dockerfile.
3. **Model card** (`docs/model-card.md`): intended use, what it is *not*
   for, inputs, training periods, the evaluation summary with CIs, known
   limits, fallbacks, and licences/attributions.

## Things to respect

- **Licences in the published output:**
  - ECMWF open data is CC BY 4.0, so attribute it.
  - Attribute WeatherBench 2 and IMD.
  - HEPPI-derived content follows step 08's user decision.
  - District boundaries follow `DISTRICTS_LICENSE.md`.
  Put an "Attributions" page in the static site.
- **India map compliance** (step 14) must hold on the public site.
- **FastAPI stays** for development. Static mode is a build output, not a
  second codebase.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Add scripts/export_static_site.py:
   - Import dashboard.api's route functions (or use FastAPI's TestClient)
     and call every endpoint for every valid parameter combination
     (enumerate them from /api/meta/leads and each route's documented
     options). Write the results under site/ (gitignored build output),
     e.g. site/api/weight-map/lead-24.json, and CAP as .xml.
   - Copy dashboard-web/ into site/.
   - Generate site/js/config.js setting window.WEAVR_STATIC = true.
   - In dashboard-web/js/api.js, add a small static-mode path mapping in
     fetchJson (one place, not per call), so the same frontend works
     against uvicorn or static files.
   - Add a test that the static export covers every endpoint (compare the
     generated file list against the API's route table).

2. Verify locally: serve site/ with `python -m http.server` on a fresh
   port and browser-verify every view, as in steps 16-19 (all leads and
   parameters, no console errors, mobile width). Also confirm that opening
   index.html via a plain static server no longer 404s.

3. Hosting: route via AskUserQuestion:
   - GitHub Pages via an Actions workflow deploying site/ (free; public if
     the repo or Pages is public)
   - Netlify / Cloudflare Pages (needs an account the user creates)
   - don't publish yet; build the artifact only
   Also state whether the repo is public, what becomes public, and the
   licence checklist status (item "Things to respect").
   Only after an explicit yes in chat: add the deploy workflow, run it
   once, open the URL in the browser pane and verify it. Then generate a
   QR code PNG for the URL into ppt/ (outside the repo) using a
   well-known library, and say which.

4. Reproducibility:
   - Add scripts/reproduce_all.py (or a Makefile target) with ordered
     stages (stores -> results -> exports -> static site), --from-stage,
     --dry-run and --check. --check re-runs the results scripts to a temp
     dir and compares them with the committed CSVs within a stated
     tolerance; report any difference instead of hiding it.
   - Add a Dockerfile (python:3.12-slim; pip install -e ".[dev,dashboard-api]";
     cfgrib/eccodes via pip wheels, as docs/data-sources.md notes). Build
     it and run the tests inside, if Docker is available locally;
     otherwise say it was NOT tested.

5. Write docs/model-card.md, following the "Goal" item 3 list. Every
   number is pulled from docs/v2-evidence-base-results.md and
   results/preregistration_verdicts.csv, with failed claims included. Add
   the explicit statement: "Not an official IMD/NCMRWF product;
   demonstration warnings only." Link it from README and the static
   site's footer.

6. README: a "Reproduce everything" section and a "Live demo" link (if
   published). One PR (plus the deploy workflow, if approved).
```

## Done when

- The static build reproduces every view and passes browser checks.
- It is published only with explicit approval, and verified live.
- `reproduce_all.py --check` runs and reports honestly.
- The Dockerfile is built (or honestly marked untested).
- `docs/model-card.md` exists, with every claim traceable to a results
  file.
