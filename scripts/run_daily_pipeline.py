#!/usr/bin/env python3
"""The real daily pipeline -- issue #8's actual operational ask, wiring
steps 1-3 (`docs/phase6-operational-scope.md`, `weavr.renormalize`,
`weavr.drift`/`scripts/run_daily_verification.py`) into one runnable job,
not just the individual pieces in isolation.

## Which combiner: Phase 3's regional weighting, not EMOS-CSG/BMA -- a real, checked reason

Phase 4/5's fitted combiners (`weavr.emos.fit_emos_csg`,
`weavr.bma.fit_hierarchical_bma`) fit *per-source statistical parameters*
-- CSGD regression coefficients, per-source spread-variance regressions --
to each named source's own real historical error characteristics
(`docs/tier2-hierarchical-baseline-results.md`: EMOS-graphcast wins 3 of 5
leads domain-wide, BMA wins 2, neither dominates). But
`docs/phase6-operational-scope.md` decided the daily pipeline fetches AIFS
in place of GraphCast/Pangu (no published live route exists for either)
and a single deterministic ECMWF Open Data pull for IFS/HRES (no ensemble
members) -- neither matches what those fits were actually built from.
Reusing GraphCast's fitted CSGD parameters against AIFS's real forecasts
would apply one model's bias-correction to a different model's output: a
real statistical mismatch, not an approximation of the same thing. Routed
to the user via `AskUserQuestion`; decided: **use Phase 3's
`weavr.weighting.fit_region_weights` instead** -- its fitted weights
(`results/tier1_regional_weights.csv`, already committed) are flat linear
OLS coefficients per source, not a per-source distributional fit, so
reusing the `graphcast`-slot weight under an AIFS substitution is a much
lighter, explicitly flagged approximation, stated here rather than hidden.
EMOS-CSG/BMA remain this project's best combiners for the *historical*
2020 JJAS evaluation; they are not used operationally by this daily
pipeline for the reason above.

## Live source substitutions, stated plainly (`SOURCE_LIVE_MODELS`)

- `graphcast` (Tier 1's fitted weight slot) <- **AIFS** (`aifs-single`),
  substituted per `docs/phase6-operational-scope.md` -- no published live
  GraphCast/Pangu route exists.
- `ifs_ens_mean` (Tier 1's fitted weight slot) <- **deterministic IFS**
  (`ifs`) -- the real 50-member ensemble mean this weight was fit against
  is not published near-real-time through this project's client; a single
  deterministic pull stands in, a second real, flagged approximation.
- `hres` <- **HRES** (`hres`) -- no substitution; same real model, live.

## Ground truth is not available same-day -- checked live, not assumed

Live-tested directly before designing the verification step:
`imdlib.get_data("rain", <current year>, <current year>, ...)` downloads a
file but fails to parse it ("Error in file reading, mismatch in size of
data-length") -- IMD's real published gridded product does not have valid
current-year data available through this project's client, consistent with
a genuine real-world publication lag. `fetch_today_obs` therefore returns
`None` as an expected, first-class case, exactly like a failed forecast
fetch -- callers need no separate handling path for it. This also means a
forecast issued *today* at lead N validates N hours in the future and can
never be scored today regardless of IMD's latency; verification here is
necessarily retrospective (see `compute_rolling_and_drift`'s own
docstring).

## Failure handling: one convention, not two

A source that fails to fetch (`fetch_live_source` returns `None`) is
handled by `build_daily_blend` exactly the way `weavr.renormalize`'s own
"never pulled at all" missing shape is designed for -- dropped and its
already-fit weight redistributed proportionally across whichever sources
actually arrived, per region, never re-fit, never a silent zero.

## Output: small committed artifacts, per step 1's real decision

`docs/phase6-operational-scope.md` decided GitHub Actions runners are
ephemeral, so outputs are small committed files, not a growing local Zarr
store. Two real artifacts, both intentionally small:
`results/daily_pipeline_history.csv` (one row per scored (date, lead) --
only once obs becomes available for that valid day) and
`results/daily_pipeline_forecasts/<issue_date>_<lead>h.npy` (one small,
~70KB flat array per (issue day, lead) on the fixed common grid -- kept
specifically so a forecast can be retrospectively scored once its valid
day's obs eventually arrives; without persisting the grid itself, no real
scoring would ever be possible given the same-day latency finding above).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tier1_regional_baseline import (  # noqa: E402
    blend_with_region_weights,
    build_region_weight_grid,
)

from weavr import verify as V  # noqa: E402
from weavr.data import imd_gridded  # noqa: E402
from weavr.data.ecmwf_open_data import fetch_latest_forecast, slice_india  # noqa: E402
from weavr.drift import (  # noqa: E402
    TRAILING_WINDOW_SAMPLES,
    DriftResult,
    compute_baseline_stats,
    detect_drift,
    trailing_window_value,
)
from weavr.grid import COMMON_LAT, COMMON_LON, regrid_to_common  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.renormalize import renormalize_weights  # noqa: E402

LEAD_HOURS = [24, 48, 72, 96, 120]
PRECIP_M_TO_MM = 1000.0
TIER1_SOURCE_NAMES = ["graphcast", "hres", "ifs_ens_mean"]

# See this module's own docstring ("Live source substitutions") for why
# each of these differs from the name it's filling.
SOURCE_LIVE_MODELS = {
    "graphcast": "aifs-single",
    "hres": "hres",
    "ifs_ens_mean": "ifs",
}

DEFAULT_WEIGHTS_CSV = "results/tier1_regional_weights.csv"
DEFAULT_HISTORY_CSV = "results/daily_pipeline_history.csv"
DEFAULT_FORECASTS_DIR = "results/daily_pipeline_forecasts"


def fetch_live_source(model: str, lead_hours: int) -> xr.DataArray | None:
    """One source's live precipitation forecast at `lead_hours`, regridded
    onto the common India grid -- or `None` if the fetch fails for any
    reason (network, upstream outage, an unpublished step for this model).
    Same "never pulled at all" missing shape `weavr.renormalize`'s own
    docstring documents; a failed fetch needs no separate handling path.

    **Known, real, unfixed risk** (found live during this project's Phase 6
    step 1 scoping, not fixed here): `ecmwf.opendata.Client.retrieve`'s own
    retry loop retries an AWS `503 Slow Down` response up to 500 times with
    a 120s backoff internally, *inside* the library call -- this function's
    `except Exception` only ever sees whatever exception eventually
    surfaces after that loop gives up (or never, in the worst case), not a
    quick failure. A real run of this script can therefore hang far longer
    than a normal network timeout if ECMWF's AWS mirror is rate-limiting.
    Not fixed in this step (the fix belongs in `ecmwf_open_data.py`'s own
    client construction, e.g. a request timeout or retry-count override,
    out of this step's scope); worth flagging for whoever runs this
    operationally.
    """
    try:
        ds = fetch_latest_forecast(param="tp", step=lead_hours, model=model)
        ds = slice_india(ds)
        ds = regrid_to_common(ds)
        return ds["tp"] * PRECIP_M_TO_MM
    except Exception:
        return None


def fetch_today_sources(lead_hours: int) -> dict[str, xr.DataArray | None]:
    return {
        slot: fetch_live_source(model, lead_hours) for slot, model in SOURCE_LIVE_MODELS.items()
    }


def fetch_today_obs() -> xr.DataArray | None:
    """See this module's own docstring ("Ground truth is not available
    same-day") -- returns `None`, checked live rather than assumed, when
    IMD's real product has no valid data for today.
    """
    try:
        today = date.today()
        ds = imd_gridded.fetch_year(today.year, var_type="rain")
        obs = ds["rain"].sel(time=today.isoformat())
        obs_ds = regrid_to_common(obs.to_dataset(name="rain"))
        return obs_ds["rain"]
    except Exception:
        return None


def load_tier1_weights(path: str = DEFAULT_WEIGHTS_CSV) -> dict[int, dict[str, dict[str, float]]]:
    """`lead_hours -> region -> {source: weight}`, from Tier 1's real,
    already-fit and committed weights. Never re-fit here -- per
    `weavr.renormalize`'s own "no re-fitting" principle, this step only
    loads what Phase 3 already fit and renormalizes it at prediction time.
    """
    weights: dict[int, dict[str, dict[str, float]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lead = int(row["lead_hours"])
            weights.setdefault(lead, {})[row["region"]] = {
                name: float(row[f"weight_{name}"]) for name in TIER1_SOURCE_NAMES
            }
    return weights


def build_daily_blend(
    tier1_weights_for_lead: dict[str, dict[str, float]],
    region_labels: xr.DataArray,
    sources: dict[str, xr.DataArray | None],
) -> tuple[xr.DataArray | None, dict]:
    """Renormalized regional blend for one lead's already-fetched sources.

    A missing source (`sources[name] is None`) is dropped and its
    already-fit weight redistributed proportionally across whichever
    sources actually showed up, per region, before blending -- see this
    module's own docstring ("Failure handling").
    """
    present_sources = tuple(name for name, da in sources.items() if da is not None)
    dropped_sources = tuple(name for name in sources if name not in present_sources)

    if not present_sources:
        return None, {
            "status": "no_sources_present",
            "present_sources": "",
            "dropped_sources": "+".join(dropped_sources),
            "any_region_fallback": True,
        }

    renormalized = {
        region: renormalize_weights(weights, present_sources)
        for region, weights in tier1_weights_for_lead.items()
    }
    any_fallback = any(r.is_fallback for r in renormalized.values())

    weight_grids = build_region_weight_grid(renormalized, region_labels, list(present_sources))
    present_forecasts = {name: sources[name] for name in present_sources}
    blend = blend_with_region_weights(present_forecasts, weight_grids)

    meta = {
        "status": "ok",
        "present_sources": "+".join(present_sources),
        "dropped_sources": "+".join(dropped_sources),
        "any_region_fallback": any_fallback,
    }
    return blend, meta


def compute_rolling_and_drift(
    history_scores: list[float],
    today_score: float,
    window: int = TRAILING_WINDOW_SAMPLES,
) -> DriftResult | None:
    """A properly held-out baseline/rolling split -- unlike
    `scripts/run_daily_verification.py`'s own demo, which necessarily
    reused the same ~18-sample series for both (that's all the real
    historical data that exists for the frozen 2020 JJAS store). Here,
    history genuinely grows one real day at a time, so the baseline is
    computed only from history entries *older* than the trailing window --
    never overlapping the window the way step 3's demo had to.

    Returns `None` (not a `DriftResult`) until at least `window + 2` real
    history entries exist -- too little held-out history for a trustworthy
    baseline (`weavr.drift.compute_baseline_stats`'s own >=2-sample floor)
    -- stated plainly rather than returning a spurious always-passing
    check.
    """
    all_scores = [*history_scores, today_score]
    baseline_scores = all_scores[:-window] if len(all_scores) > window else []
    if len(baseline_scores) < 2:
        return None
    baseline = compute_baseline_stats(baseline_scores, metric_name="rmse_mm")
    rolling_value = trailing_window_value(all_scores, window=window)
    return detect_drift(rolling_value, baseline)


def _forecast_path(forecasts_dir: str, issue_date: date, lead_hours: int) -> Path:
    return Path(forecasts_dir) / f"{issue_date.isoformat()}_{lead_hours}h.npy"


def _load_history_scores(history_path: str, lead_hours: int) -> list[float]:
    path = Path(history_path)
    if not path.exists():
        return []
    scores = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["lead_hours"]) == lead_hours:
                scores.append(float(row["rmse_mm"]))
    return scores


def _append_history_row(history_path: str, row: dict) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-csv", default=DEFAULT_WEIGHTS_CSV)
    parser.add_argument("--history-csv", default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--forecasts-dir", default=DEFAULT_FORECASTS_DIR)
    args = parser.parse_args()

    tier1_weights = load_tier1_weights(args.weights_csv)
    region_labels = assign_regions()
    today = date.today()
    obs = fetch_today_obs()
    print(f"Daily pipeline run for {today.isoformat()} -- obs available: {obs is not None}")

    Path(args.forecasts_dir).mkdir(parents=True, exist_ok=True)

    for lead_hours in LEAD_HOURS:
        sources = fetch_today_sources(lead_hours)
        blend, meta = build_daily_blend(tier1_weights[lead_hours], region_labels, sources)
        print(f"[lead {lead_hours:>3}h] {meta}")

        if blend is not None:
            np.save(_forecast_path(args.forecasts_dir, today, lead_hours), blend.values)

        # Retrospective scoring: a forecast issued `lead_hours` ago at this
        # lead validates today -- see this module's own docstring ("Ground
        # truth is not available same-day"). lead_hours is always a
        # multiple of 24 (LEAD_HOURS), so this is always a whole number of
        # days.
        issued_on = today - timedelta(days=lead_hours // 24)
        due_path = _forecast_path(args.forecasts_dir, issued_on, lead_hours)
        if obs is not None and due_path.exists():
            past_forecast = xr.DataArray(
                np.load(due_path), coords={"latitude": COMMON_LAT, "longitude": COMMON_LON},
                dims=["latitude", "longitude"],
            )
            today_score = float(V.rmse(past_forecast, obs))
            history_scores = _load_history_scores(args.history_csv, lead_hours)
            drift = compute_rolling_and_drift(history_scores, today_score)
            _append_history_row(
                args.history_csv,
                {
                    "date": today.isoformat(),
                    "lead_hours": lead_hours,
                    "rmse_mm": today_score,
                    "is_drift": drift.is_drift if drift else "",
                    "reason": drift.reason if drift else "insufficient held-out history",
                },
            )
            print(f"  scored: rmse={today_score:.2f}mm, drift={drift.is_drift if drift else 'n/a'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
