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
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
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
# Built by scripts/build_basemap.py; gitignored. Module-level so tests can point it
# at a small temp file.
BASEMAP_TILES_PATH = Path(__file__).resolve().parent.parent / "data" / "basemap" / "india.pmtiles"
# The official India boundary, supplied by the user (docs/basemap-scope.md). Kept
# next to the tiles under the gitignored data/ so a file whose licence is not
# confirmed is never committed or served by the static mount.
BASEMAP_BOUNDARY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "basemap" / "india-boundary.geojson"
)
DEFAULT_BOUNDARY_ATTRIBUTION = "Boundary: user-supplied official file"
_BOUNDARY_GEOMETRIES = {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}


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


ALLOWED_EXTREME_THRESHOLDS = (115.6, 204.5)


@app.get("/api/extreme-probability")
def extreme_probability(
    lead: int = Query(..., description="Lead time in hours"),
    threshold: float = Query(204.5, description="Threshold in mm (115.6 or 204.5)"),
) -> dict[str, Any]:
    if threshold not in ALLOWED_EXTREME_THRESHOLDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"threshold={threshold} not supported; must be one of "
                f"{list(ALLOWED_EXTREME_THRESHOLDS)}"
            ),
        )
    available = available_probability_grid_leads()
    _validate_lead(lead, available)
    try:
        grid = load_probability_grid(lead, threshold=threshold)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "latitude": grid["latitude"].tolist(),
        "longitude": grid["longitude"].tolist(),
        "probability": grid["probability"].tolist(),
        "is_fallback": grid["is_fallback"].tolist(),
        "method": (
            grid["method"].tolist()
            if hasattr(grid["method"], "tolist")
            else list(grid["method"])
        ),
        "sample_time": grid["sample_time"],
        "lead_hours": grid["lead_hours"],
        "threshold": threshold,
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


@app.get("/api/basemap/status")
def basemap_status() -> dict[str, Any]:
    """Whether the local tile file exists, so the frontend can fall back cleanly."""
    if BASEMAP_TILES_PATH.is_file():
        return {"available": True, "bytes": BASEMAP_TILES_PATH.stat().st_size}
    return {"available": False, "bytes": None}


def _boundary_geometries(document: Any) -> list[dict[str, Any]]:
    """Every geometry in a GeoJSON document, or raise ValueError if it is not
    an outline (line or polygon) GeoJSON."""
    if not isinstance(document, dict):
        raise ValueError("boundary file is not a GeoJSON object")
    kind = document.get("type")
    if kind == "FeatureCollection":
        geometries = [f.get("geometry") for f in document.get("features", [])]
    elif kind == "Feature":
        geometries = [document.get("geometry")]
    elif kind in _BOUNDARY_GEOMETRIES:
        geometries = [document]
    else:
        raise ValueError(f"unsupported GeoJSON type {kind!r}")
    if not geometries:
        raise ValueError("boundary file has no geometry")
    for geometry in geometries:
        if not isinstance(geometry, dict) or geometry.get("type") not in _BOUNDARY_GEOMETRIES:
            raise ValueError("boundary geometries must be lines or polygons")
    return [g for g in geometries if isinstance(g, dict)]


@app.get("/api/basemap/boundary")
def basemap_boundary() -> Response:
    """The user-supplied official boundary, or 404 {"configured": false}.

    A file that exists but is not valid outline GeoJSON is a 500 with the
    reason, not a silently ignored one: a boundary that quietly fails to draw
    would look exactly like "no boundary configured"."""
    if not BASEMAP_BOUNDARY_PATH.is_file():
        return JSONResponse({"configured": False}, status_code=404)
    try:
        document = json.loads(BASEMAP_BOUNDARY_PATH.read_text())
        _boundary_geometries(document)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"invalid boundary file: {exc}") from exc
    attribution = document.get("attribution") if isinstance(document, dict) else None
    return JSONResponse(
        {
            "configured": True,
            "attribution": attribution or DEFAULT_BOUNDARY_ATTRIBUTION,
            "geojson": document,
        }
    )


@app.get("/basemap/india.pmtiles", response_model=None)
def basemap_tiles() -> Response:
    """The local PMTiles archive. FileResponse answers HTTP Range requests
    (206 Partial Content), which is how the browser reads a PMTiles file."""
    if not BASEMAP_TILES_PATH.is_file():
        return PlainTextResponse(
            "Basemap tiles not built. Run: python scripts/build_basemap.py",
            status_code=404,
        )
    return FileResponse(BASEMAP_TILES_PATH, media_type="application/octet-stream")


# Registered last so it never shadows the /api/* routes above -- FastAPI
# resolves routes in registration order, and this mount's catch-all would
# otherwise intercept every path, including a typo'd /api/ call, and
# return a 404 from the static-files layer instead of routing to FastAPI.
if DASHBOARD_WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_WEB_DIR, html=True), name="dashboard-web")
