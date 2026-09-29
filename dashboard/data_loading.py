"""Pure data-loading/reshaping for the Phase 7 dashboard's views.

No Streamlit or Plotly imports here: these are plain pandas/pathlib
functions, tested directly against real and fixture CSVs (this project's
own established convention -- test pure logic, not rendered UI). Originally
imported by the Streamlit view modules under dashboard/views/ (retired --
see README.md's "Streamlit's status"); now imported directly by
dashboard/api.py, which adds only JSON serialization.

Named `data_loading.py`, not `data.py`, to leave `dashboard/data/` free for
step 3/4's small committed example export grids
(`docs/phase7-dashboard-scope.md`'s own decision) -- a Python module and a
package-relative directory of that same name would collide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DASHBOARD_DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_WEIGHTS_CSV = RESULTS_DIR / "tier1_regional_weights.csv"
DEFAULT_TIER0_CSV = RESULTS_DIR / "tier0_baseline.csv"
DEFAULT_TIER1_CSV = RESULTS_DIR / "tier1_regional_baseline.csv"
DEFAULT_TIER2_CSV = RESULTS_DIR / "tier2_hierarchical_baseline.csv"
DEFAULT_BLEND_GRID_NPZ = DASHBOARD_DATA_DIR / "example_blend_grid.npz"
DEFAULT_TIER3_CSV = RESULTS_DIR / "tier3_regime_conditioned_baseline.csv"
DEFAULT_PROBABILITY_GRID_NPZ = DASHBOARD_DATA_DIR / "example_probability_grid.npz"


def load_weight_map_data(csv_path: str | Path = DEFAULT_WEIGHTS_CSV) -> pd.DataFrame:
    """Tidy `results/tier1_regional_weights.csv` into one row per (lead, region, source).

    This project's real weighting scheme, per `docs/phase6-operational-scope.md`'s
    decision for the daily pipeline, is Phase 3's per-region OLS fit
    (`weavr.weighting.fit_region_weights`) -- not a softmax or GBM gate.
    This function only reshapes the already-fit `weight_<source>` columns;
    it does not re-fit anything.

    Returns columns: lead_hours, region, is_fallback, reason,
    n_train_points, source, weight.
    """
    df = pd.read_csv(csv_path)
    weight_cols = [c for c in df.columns if c.startswith("weight_")]
    tidy = df.melt(
        id_vars=["lead_hours", "region", "is_fallback", "reason", "n_train_points"],
        value_vars=weight_cols,
        var_name="source",
        value_name="weight",
    )
    tidy["source"] = tidy["source"].str.removeprefix("weight_")
    return tidy


def load_skill_trends_data(
    tier0_csv: str | Path = DEFAULT_TIER0_CSV,
    tier1_csv: str | Path = DEFAULT_TIER1_CSV,
    tier2_csv: str | Path = DEFAULT_TIER2_CSV,
    tier3_csv: str | Path = DEFAULT_TIER3_CSV,
) -> pd.DataFrame:
    """Tidy all 4 tiers' real results CSVs into one (tier, method, lead, metric, value) table.

    Each tier's CSV has a different real column layout, checked directly
    (`head -2` on every file) rather than assumed identical:

    - `tier0_baseline.csv`: one method, the equal-weight 3-source ensemble
      mean (`rmse_mm` only -- a deterministic blend has no CRPS).
    - `tier1_regional_baseline.csv`: two methods, `tier1_` (the fitted
      regional blend) and `equal_` (its own equal-weight baseline, RMSE
      only).
    - `tier2_hierarchical_baseline.csv`: three methods (`emos_graphcast`,
      `emos_ifs_ens`, `bma`), each with real `rmse_mm` AND `crps_mm` (real
      fitted predictive distributions, not just point forecasts).
    - `tier3_regime_conditioned_baseline.csv`: repeats tier2's 3 methods'
      columns verbatim -- checked directly (`results/tier2_*.csv` vs.
      `results/tier3_*.csv`, row by row): every `emos_graphcast_*`,
      `emos_ifs_ens_*`, `bma_*` value is bit-for-bit identical across both
      files (same held-out window, same numbers). This function does not
      re-plot that duplicate; tier3 contributes only its own new method,
      `regime_conditioned`.

    The real, checked outcome across tiers is mixed, not monotonically
    improving: tier1 beats tier0 domain-wide at every lead but not in every
    region x lead cell; tier2's EMOS-CSG and BMA trade off which wins (3 vs.
    2 of 5 leads); tier3's regime-conditioned reweighting does not beat
    tier2 at any lead (0 of 5) -- see `docs/tier1-regional-weights-results.md`,
    `docs/tier2-hierarchical-baseline-results.md`,
    `docs/phase5-regime-conditioned-results.md`. This function returns every
    real method's own numbers as-is; it does not smooth or hide the mixed
    picture.
    """
    rows: list[dict[str, object]] = []

    tier0 = pd.read_csv(tier0_csv)
    for _, row in tier0.iterrows():
        rows.append(
            {
                "tier": "tier0",
                "method": "tier0_ensemble_mean",
                "lead_hours": row["lead_hours"],
                "metric": "rmse_mm",
                "value": row["rmse_mm"],
            }
        )

    tier1 = pd.read_csv(tier1_csv)
    for _, row in tier1.iterrows():
        rows.append(
            {
                "tier": "tier1",
                "method": "tier1_regional_blend",
                "lead_hours": row["lead_hours"],
                "metric": "rmse_mm",
                "value": row["tier1_rmse_mm"],
            }
        )
        rows.append(
            {
                "tier": "tier1",
                "method": "tier1_equal_weight",
                "lead_hours": row["lead_hours"],
                "metric": "rmse_mm",
                "value": row["equal_rmse_mm"],
            }
        )

    tier2_methods = [
        ("tier2_emos_graphcast", "emos_graphcast"),
        ("tier2_emos_ifs_ens", "emos_ifs_ens"),
        ("tier2_bma", "bma"),
    ]
    tier2 = pd.read_csv(tier2_csv)
    for _, row in tier2.iterrows():
        for method, prefix in tier2_methods:
            for metric in ("rmse_mm", "crps_mm"):
                rows.append(
                    {
                        "tier": "tier2",
                        "method": method,
                        "lead_hours": row["lead_hours"],
                        "metric": metric,
                        "value": row[f"{prefix}_{metric}"],
                    }
                )

    tier3 = pd.read_csv(tier3_csv)
    for _, row in tier3.iterrows():
        for metric in ("rmse_mm", "crps_mm"):
            rows.append(
                {
                    "tier": "tier3",
                    "method": "tier3_regime_conditioned",
                    "lead_hours": row["lead_hours"],
                    "metric": metric,
                    "value": row[f"regime_{metric}"],
                }
            )

    return pd.DataFrame(rows)


def available_blend_grid_leads(npz_path: str | Path = DEFAULT_BLEND_GRID_NPZ) -> list[int]:
    """Lead times (hours) present in the committed example blend-grid export."""
    with np.load(npz_path) as data:
        return [int(lh) for lh in data["lead_hours"]]


def load_blend_grid(lead_hours: int, npz_path: str | Path = DEFAULT_BLEND_GRID_NPZ) -> dict:
    """Load one lead time's real example blended-forecast grid.

    Built by `scripts/export_dashboard_example_grids.py` from Tier 1's real
    regional blend (`docs/phase7-dashboard-scope.md`'s committed-example
    decision) -- this function only reads the already-exported `.npz`, it
    does not compute anything.

    Returns `{"latitude": ndarray, "longitude": ndarray, "values": ndarray
    (lat, lon), "sample_time": str, "lead_hours": int}`.
    """
    with np.load(npz_path) as data:
        available = [int(lh) for lh in data["lead_hours"]]
        if lead_hours not in available:
            raise ValueError(f"lead_hours={lead_hours} not in this export's leads {available}")
        return {
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "values": data[f"blend_lead_{lead_hours}"],
            "sample_time": str(data[f"sample_time_lead_{lead_hours}"]),
            "lead_hours": lead_hours,
        }


def available_probability_grid_leads(
    npz_path: str | Path | None = None,
) -> list[int]:
    """Lead times (hours) present in the committed example probability-grid export."""
    actual_path = DEFAULT_PROBABILITY_GRID_NPZ if npz_path is None else npz_path
    with np.load(actual_path) as data:
        return [int(lh) for lh in data["lead_hours"]]


def load_probability_grid(
    lead_hours: int,
    npz_path: str | Path | None = None,
    *,
    threshold: float = 204.5,
) -> dict:
    """Load one lead time's real example P(rain > threshold) grid.

    Built by `scripts/export_dashboard_example_grids.py` from EMOS-CSG's
    real fitted `ifs_ens` combiner with pooled extreme-value tail (Step 12).

    Returns `{"latitude": ndarray, "longitude": ndarray, "probability":
    ndarray (lat, lon), "is_fallback": bool ndarray (lat, lon), "method":
    ndarray (lat, lon), "sample_time": str, "lead_hours": int, "threshold": float}`.
    """
    actual_path = DEFAULT_PROBABILITY_GRID_NPZ if npz_path is None else npz_path
    with np.load(actual_path) as data:
        available = [int(lh) for lh in data["lead_hours"]]
        if lead_hours not in available:
            raise ValueError(f"lead_hours={lead_hours} not in this export's leads {available}")

        th_slug = "115p6" if abs(threshold - 115.6) < 0.1 else "204p5"
        key_prob = f"probability_{th_slug}_lead_{lead_hours}"
        if key_prob not in data:
            if abs(threshold - 204.5) < 0.1 and f"probability_lead_{lead_hours}" in data:
                key_prob = f"probability_lead_{lead_hours}"
            else:
                raise KeyError(
                    f"Threshold {threshold}mm not found in {actual_path} "
                    f"(missing key '{key_prob}')"
                )

        key_fallback = f"is_fallback_{th_slug}_lead_{lead_hours}"
        if key_fallback not in data:
            if abs(threshold - 204.5) < 0.1 and f"is_fallback_lead_{lead_hours}" in data:
                key_fallback = f"is_fallback_lead_{lead_hours}"
            else:
                raise KeyError(
                    f"Threshold {threshold}mm fallback mask not found in {actual_path} "
                    f"(missing key '{key_fallback}')"
                )

        key_method = f"method_{th_slug}_lead_{lead_hours}"
        if key_method in data:
            method = data[key_method]
        else:
            is_fb = data[key_fallback]
            method = np.where(is_fb, "fallback", "csgd")

        return {
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "probability": data[key_prob],
            "is_fallback": data[key_fallback],
            "method": method,
            "sample_time": str(data[f"sample_time_lead_{lead_hours}"]),
            "lead_hours": lead_hours,
            "threshold": threshold,
        }
