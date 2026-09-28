"""Thin, read-only JSON API for the dashboard frontend migration.

Step 2 of `workspace/frontend-prompts/` (outside this repo), executing
`workspace/frontendplan.md` per `docs/frontend-migration-scope.md`'s
locked decisions (FastAPI + uvicorn, plain nested JSON grid payloads).
Every handler below imports `dashboard.data_loading` / `dashboard.colors`
directly and unchanged -- this module only validates query params and
serializes their real return values to JSON; it does not reimplement any
loading, reshaping, or rain-bin classification logic a second time.

The original Streamlit prototype this API's views were ported from
(`dashboard/app.py`, `dashboard/views/*.py`) has been retired (step 7 of
`workspace/frontend-prompts/`, after this new frontend reached real,
checked parity with it) -- see README.md's "Streamlit's status" section
for the routed decision. `dashboard.data_loading` / `dashboard.colors`
remain, unchanged, as the single source of truth both the old and new
frontends read from.

Also serves the static `dashboard-web/` frontend (step 3 of
`workspace/frontend-prompts/`) from this same process, at `/` -- the
simplest option given this step's own FastAPI decision, and it avoids
needing CORS entirely since the frontend's `fetch()` calls stay
same-origin. The API routes above are registered first, so `/api/*`
always resolves to a real endpoint rather than the static mount's
catch-all.

Run with:
    pip install -e ".[dashboard-api]"
    uvicorn dashboard.api:app --reload
Then open http://127.0.0.1:8000/ for the frontend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from dashboard.colors import PROBABILITY_COLORSCALE, RAIN_BIN_COLORS, rain_bin_index
from dashboard.data_loading import (
    available_blend_grid_leads,
    available_probability_grid_leads,
    load_blend_grid,
    load_probability_grid,
    load_skill_trends_data,
    load_weight_map_data,
)
from weavr.rain_bins import RAIN_BIN_LABELS
from weavr.verify import IMD_RAIN_THRESHOLDS_MM

app = FastAPI(title="WEAVR dashboard API")

DASHBOARD_WEB_DIR = Path(__file__).resolve().parent.parent / "dashboard-web"


def _validate_lead(lead: int, available: list[int]) -> None:
    """A 422 with the real available leads, not a silent default."""
    if lead not in available:
        raise HTTPException(
            status_code=422,
            detail=f"lead={lead} not in this data's real available leads {available}",
        )


def _records(df: Any) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records, via pandas' own to_json (handles
    numpy int64/float64/bool_ scalars correctly, unlike DataFrame.to_dict,
    which would leave numpy scalar types in the response)."""
    result: list[dict[str, Any]] = json.loads(df.to_json(orient="records"))
    return result


@app.get("/api/weight-map")
def weight_map(lead: int = Query(..., description="Lead time in hours")) -> list[dict[str, Any]]:
    data = load_weight_map_data()
    available = sorted(int(lh) for lh in data["lead_hours"].unique())
    _validate_lead(lead, available)
    return _records(data[data["lead_hours"] == lead])


@app.get("/api/skill-trends")
def skill_trends(
    metric: str = Query(..., description="rmse_mm or crps_mm"),
) -> list[dict[str, Any]]:
    data = load_skill_trends_data()
    available_metrics = sorted(data["metric"].unique())
    if metric not in available_metrics:
        raise HTTPException(
            status_code=422,
            detail=(
                f"metric={metric!r} not one of this data's real available "
                f"metrics {available_metrics}"
            ),
        )
    return _records(data[data["metric"] == metric])


@app.get("/api/blended-map")
def blended_map(lead: int = Query(..., description="Lead time in hours")) -> dict[str, Any]:
    available = available_blend_grid_leads()
    _validate_lead(lead, available)
    grid = load_blend_grid(lead)
    bin_index = rain_bin_index(grid["values"])
    return {
        "latitude": grid["latitude"].tolist(),
        "longitude": grid["longitude"].tolist(),
        "bin_index": bin_index.tolist(),
        "sample_time": grid["sample_time"],
        "lead_hours": grid["lead_hours"],
    }


@app.get("/api/extreme-probability")
def extreme_probability(
    lead: int = Query(..., description="Lead time in hours"),
) -> dict[str, Any]:
    available = available_probability_grid_leads()
    _validate_lead(lead, available)
    grid = load_probability_grid(lead)
    return {
        "latitude": grid["latitude"].tolist(),
        "longitude": grid["longitude"].tolist(),
        "probability": grid["probability"].tolist(),
        "is_fallback": grid["is_fallback"].tolist(),
        "sample_time": grid["sample_time"],
        "lead_hours": grid["lead_hours"],
    }


@app.get("/api/colors")
def colors() -> dict[str, Any]:
    return {
        "rain_bin_colors": RAIN_BIN_COLORS,
        "rain_bin_labels": list(RAIN_BIN_LABELS),
        "probability_colorscale": PROBABILITY_COLORSCALE,
        "imd_rain_thresholds_mm": list(IMD_RAIN_THRESHOLDS_MM),
    }


@app.get("/api/meta/leads")
def meta_leads() -> dict[str, list[int]]:
    blend_leads = available_blend_grid_leads()
    probability_leads = available_probability_grid_leads()
    if blend_leads != probability_leads:
        # A real, checkable invariant -- both example grids were exported
        # together by scripts/export_dashboard_example_grids.py for the
        # same 5 lead times. A divergence here means the committed exports
        # are out of sync, not something callers should silently paper over.
        raise HTTPException(
            status_code=500,
            detail=(
                "blend-grid and probability-grid leads have diverged: "
                f"{blend_leads} vs {probability_leads}"
            ),
        )
    return {"leads": blend_leads}


# Registered last so it never shadows the /api/* routes above -- FastAPI
# resolves routes in registration order, and this mount's catch-all would
# otherwise intercept every path, including a typo'd /api/ call, and
# return a 404 from the static-files layer instead of routing to FastAPI.
if DASHBOARD_WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_WEB_DIR, html=True), name="dashboard-web")
