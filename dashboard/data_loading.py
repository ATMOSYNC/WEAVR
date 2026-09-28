"""Pure data-loading/reshaping for the Phase 7 dashboard's views.

No Streamlit or Plotly imports here: these are plain pandas/pathlib
functions, tested directly against real and fixture CSVs (this project's
own established convention -- test pure logic, not rendered UI). The view
modules under dashboard/views/ import from here and add only rendering.

Named `data_loading.py`, not `data.py`, to leave `dashboard/data/` free for
step 3/4's small committed example export grids
(`docs/phase7-dashboard-scope.md`'s own decision) -- a Python module and a
package-relative directory of that same name would collide.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_WEIGHTS_CSV = RESULTS_DIR / "tier1_regional_weights.csv"
DEFAULT_TIER0_CSV = RESULTS_DIR / "tier0_baseline.csv"
DEFAULT_TIER1_CSV = RESULTS_DIR / "tier1_regional_baseline.csv"
DEFAULT_TIER2_CSV = RESULTS_DIR / "tier2_hierarchical_baseline.csv"
DEFAULT_TIER3_CSV = RESULTS_DIR / "tier3_regime_conditioned_baseline.csv"


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
