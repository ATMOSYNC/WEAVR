# Phase 7 dashboard scope: data sources, IMD colour scheme, and stack

Issue #9 names 4 views (blended map, weight-map per model x lead time, skill
trend charts, extreme-probability maps in IMD colour codes) and a stack
("Streamlit/Plotly/Leaflet"). Before any view is built (steps 2-5), this
document locks: which real data each view reads, IMD's real, cited colour
convention, and the stack's exact packages.

## 1. Per-view data-source audit

Checked directly against every file under `results/*.csv` (headers read with
`head -2`) and against `.gitignore`:

| View | Data it needs | What this repo actually has |
|---|---|---|
| Weight-map (model x lead) | per-region, per-lead, per-source weight | `results/tier1_regional_weights.csv` — exact shape needed: `lead_hours, region, is_fallback, reason, n_train_points, weight_graphcast, weight_hres, weight_ifs_ens_mean`. **Fully supported, already committed, zero new computation.** |
| Skill trend charts | per-lead RMSE/CRPS/etc. across tiers | `results/tier0_baseline.csv`, `tier1_regional_baseline.csv`, `tier2_hierarchical_baseline.csv`, `tier3_regime_conditioned_baseline.csv` all carry per-lead scalar scores (RMSE, bias, ACC, SEEPS, CRPS depending on tier, FSS/POD/FAR/CSI/ETS per IMD threshold). Column names differ per tier (`tier1_`/`equal_` prefixes, `emos_graphcast_`/`emos_ifs_ens_`/`bma_` prefixes, `regime_` prefix) — confirmed by reading each header directly, not assumed identical. **Fully supported, already committed, zero new computation** (a tidying/reshaping step is needed, not new numbers). |
| Blended map | a real `(latitude, longitude)` blended-forecast grid | No committed CSV carries a spatial grid at all — every `results/*.csv` is scalar/aggregate per lead/bin/region. The only real spatial arrays exist in `data/*.zarr` (gitignored: `.gitignore` has `*.zarr` and `/data/`) and in `results/daily_pipeline_forecasts/*.npy` (also gitignored, and per PR #39 never actually populated — `run_daily_pipeline.py`'s `main()` was deliberately not run live given the documented ECMWF retry-storm risk). **Not supported by anything committed today.** |
| Extreme-probability map | a real `(latitude, longitude)` exceedance-probability grid | Same gap as the blended map — no committed spatial grid exists anywhere. **Not supported by anything committed today.** |

## 2. Spatial-data-source decision (routed to the user, not assumed)

`data/baseline_2020_jjas.zarr` exists locally on this machine right now
(confirmed: `ls data/` shows it, built via `scripts/build_baseline_store.py`
in an earlier phase). But it is gitignored, so it exists only on whichever
machine happened to run that build script — not on a fresh clone, and not
guaranteed on a judge's or teammate's machine.

This is a real tradeoff (always-live vs. always-renderable), routed to the
user via `AskUserQuestion` rather than picked by default. **Decision: bake a
small, committed example export.**

- A new script, `scripts/export_dashboard_example_grids.py`, will (in step 3
  / step 4, when those views are actually built) load
  `data/baseline_2020_jjas.zarr` once, real values for one representative
  lead time, and export:
  - `dashboard/data/example_blend_grid.npz` — Tier 1's real regional blend
    (via `run_tier1_regional_baseline.py`'s own `build_region_weight_grid`
    / `blend_with_region_weights`), on `weavr.grid.COMMON_LAT`/`COMMON_LON`.
  - `dashboard/data/example_probability_grid.npz` — P(rain > 204.5mm) from
    the real fitted EMOS-CSG or BMA distribution (step 4's own choice of
    which combiner, and why).
- Both files are small (one 2D float grid each on the real 129x135-point
  common grid, `weavr.grid.COMMON_LAT`/`COMMON_LON` — a few hundred KB, safe
  to commit), and are the concrete target
  steps 3 and 4 build against.
- Consequence, stated plainly for the dashboard's own UI text (per step 5):
  these two views show one real, static, already-computed example grid, not
  a live re-blend of the current local store. Anyone who *does* have the
  local Zarr store built can re-run the export script to refresh the
  committed example, but the dashboard itself does not require that store
  to exist to render.

## 3. IMD's real rainfall colour scheme (cited, not guessed)

Checked by grep: `docs/phase-plan.md:48` names "IMD colour codes" as a
requirement, but no colour values are defined anywhere in this codebase
before this document.

Two distinct, real IMD colour conventions exist publicly — checked directly
against IMD's own site and against independent reporting, not assumed to be
the same scheme:

1. **IMD's rainfall-*departure*-from-normal legend** (percentage vs. the
   long-period average, not absolute rainfall) — read directly from IMD's
   own site, `mausam.imd.gov.in/responsive/img/img-in/legendsRFPer.svg`:
   `#ffff00` Large Deficient (-99% to -60%), `#ff0012` Deficient (-59% to
   -20%), `#00ff3e` Normal (-19% to 19%), `#58cced` Excess (20% to 59%),
   `#3895de` Large Excess (60%+). **Not used here** — this project's views
   are about absolute rainfall amount/exceedance probability, not departure
   from a long-period average, so this scheme doesn't apply.
2. **IMD's official 4-colour weather-warning code** (green / yellow /
   orange / red), confirmed via IMD's own public communications and
   independent reporting (Business Standard, Outlook Traveller) to carry
   exact 24-hour rainfall thresholds:
   - Green: < 64.5mm
   - Yellow: 64.5mm - 115.5mm
   - Orange: 115.6mm - 204.4mm
   - Red: >= 204.5mm

   **These boundaries are exactly this project's own already-established
   `weavr.verify.IMD_RAIN_THRESHOLDS_MM = (7.5, 64.5, 115.6, 204.5)` and
   `weavr.rain_bins.RAIN_BIN_LABELS = ("dry", "light", "heavy",
   "very_heavy", "extremely_heavy")` above the 7.5mm split** (64.5 -> heavy,
   115.6 -> very_heavy, 204.5 -> extremely_heavy) — checked, and no
   reconciliation is needed; the project's own bin edges already are IMD's
   real published category boundaries.

   **Decision: use this warning-code scheme** for both the blended-map's
   rain-intensity breakpoints (step 3) and the extreme-probability map's
   colour scale (step 4), since it is IMD's own real, officially cited,
   rainfall-amount-based convention (not the unrelated departure-from-normal
   scheme above), and its 3 non-trivial breakpoints already match this
   project's own bins exactly. Mapping used by both views:

   | Bin (`weavr.rain_bins`) | Range | Colour | Approx. hex |
   |---|---|---|---|
   | dry | < 7.5mm | white / no fill | `#ffffff` |
   | light | 7.5 - 64.4mm | green | `#2ecc40` |
   | heavy | 64.5 - 115.5mm | yellow | `#ffdc00` |
   | very_heavy | 115.6 - 204.4mm | orange | `#ff8c00` |
   | extremely_heavy | >= 204.5mm | red | `#ff0000` |

   The 4 non-white hex values above are the commonly used renderings of
   IMD's named green/yellow/orange/red alert colours (IMD's own warning
   graphics use this convention; no single official hex value is published
   alongside the alert names themselves) — the mm boundaries are the real,
   cited, IMD-sourced fact; the exact hex shades are a reasonable, stated
   approximation of them, not an invented palette or an invented boundary.

## 4. Stack decision

Checked directly against `pyproject.toml`: none of Streamlit, Plotly, or
Leaflet's Python binding is an existing dependency (`dependencies` currently
lists only `numpy`, `xarray`, `dask`, `zarr`, `gcsfs`, `ecmwf-opendata`,
`imdlib`, `cfgrib`, `netcdf4`, `xskillscore`, `scipy`, `pyarrow`,
`requests`).

**Decision: Streamlit + Plotly**, not Leaflet.

- Streamlit is the app framework (issue #9's own wording lists it first);
  it needs no separate frontend build step and composes cleanly with this
  project's existing pandas/xarray-shaped data.
- Plotly renders every view: `plotly.express.choropleth`/`imshow` for the
  weight-map and the two spatial grids (a `(lat, lon)` array renders
  directly as a heatmap/contour with a discrete IMD-threshold colour scale,
  no separate raster-tiling step needed), and `plotly.express.line` for the
  skill-trend charts.
- Leaflet is not added: it is a JS mapping library whose common Python
  bindings (e.g. `streamlit-folium`) exist specifically to embed raster
  *tile* layers over a real basemap, which this project's data doesn't need
  — the spatial grids here are a fixed India-domain array on
  `weavr.grid.COMMON_LAT`/`COMMON_LON`, not tile-served imagery, so a
  Plotly choropleth/heatmap over India's outline is the simpler, already
  sufficient real fit, and avoids installing a second, largely redundant
  mapping stack that issue #9's own "or" phrasing ("Streamlit/Plotly/
  Leaflet") does not require picking all three of.

New optional-dependency group added to `pyproject.toml`:

```toml
[project.optional-dependencies]
dashboard = [
    "streamlit",
    "plotly",
]
```
