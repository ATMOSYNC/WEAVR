#!/usr/bin/env python3
"""Run the Tier 3 regime-conditioned baseline: Phase 5's actual exit
criterion (issue #7's own go/no-go language, quoted in
`docs/phase5-regime-conditioned-results.md`).

Following Phase 4 step 6's own precedent
(`scripts/run_tier2_hierarchical_baseline.py`): recomputes Phase 4's real
EMOS-CSG and hierarchical BMA fits/scores *inside this script*, on the same
aligned samples and `seasonal_block_split` the regime-conditioned model
uses, rather than reading `docs/tier2-hierarchical-baseline-results.md`'s
own numbers directly (which used its own independently-computed
alignment) and assuming the splits line up -- the same "why the
equal-weight mean is recomputed here" reasoning
`scripts/run_tier1_regional_baseline.py` already documents, extended one
more tier.

**The regime-conditioned model itself (`src/weavr/regime_weighting.py`,
Phase 5 step 3) is a Tier-1-shaped model** -- a per-category OLS weight
blend of `graphcast`/`hres`/`ifs_ens_mean`, conditioned on
`weavr.regimes.classify_monsoon_active_break`'s per-day label instead of
`weavr.regions.assign_regions`'s per-gridpoint label -- not a per-bin/
per-source probabilistic model like EMOS-CSG or BMA. It therefore produces
one deterministic point forecast per cell, not a fitted predictive
distribution. Its CRPS is scored via a real, checked identity, not an ad
hoc substitute: **the CRPS of a degenerate (point-mass) predictive
distribution reduces exactly to absolute error**, `|forecast - obs|`
(`_score_point_forecast`) -- the same proper score EMOS-CSG/BMA are scored
with, evaluated at a distribution this model actually outputs, so the
comparison stays apples-to-apples rather than inventing a different
metric for one method.

## The pre-stated "beats Phase 4" criterion (decided before running this
script, not after seeing its numbers)

Issue #7's own words are domain-level ("must beat Phase 4's hierarchical
EMOS/BMA on held-out verification"), and every prior tier's own headline
comparison in this project has been domain-wide CRPS, counted by number of
leads won out of 5 (Phase 4 step 6's own "EMOS-graphcast wins 3, BMA wins
2, EMOS-ifs_ens wins 0" framing). This script adopts the identical
convention as the primary go/no-go criterion:

    The regime-conditioned model "beats Phase 4" if its domain-wide CRPS
    is the lowest of the four methods (itself, EMOS-graphcast,
    EMOS-ifs_ens, BMA) at a majority of leads (>= 3 of 5).

Per-bin CRPS (mirroring Phase 4 step 6's own per-bin table) is reported as
supplementary evidence for *where* any win or loss comes from, not as a
second, competing go/no-go criterion decided after the fact.
`docs/phase5-regime-conditioned-results.md` applies this exact criterion,
including the possibility that the honest answer is "no go."

Usage:
    python scripts/run_tier3_regime_conditioned_baseline.py
        [--baseline-store PATH] [--lagged-store PATH] [--ifs-ensemble-store PATH]
        [--climatology PATH] [--out-csv PATH] [--bin-out-csv PATH]
        [--test-fraction F] [--n-monte-carlo N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seeps_climatology import load_climatology  # noqa: E402
from run_tier2_hierarchical_baseline import (  # noqa: E402
    _domain_summary,
    align_all_sources,
    load_graphcast_ensemble,
    load_hres_forecast,
    load_ifs_ensemble,
    score_bma_cells,
    score_emos_source,
)

from weavr.bma import fit_hierarchical_bma  # noqa: E402
from weavr.emos import fit_emos_csg  # noqa: E402
from weavr.rain_bins import RAIN_BIN_LABELS, classify_rain_bin  # noqa: E402
from weavr.regime_weighting import (  # noqa: E402
    blend_with_regime_weights,
    build_regime_weight_series,
    fit_regime_weights,
)
from weavr.regimes import classify_monsoon_active_break  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.splits import InsufficientTimeBlocksError, seasonal_block_split  # noqa: E402

LEAD_HOURS = [24, 48, 72, 96, 120]
# The 3 sources the regime-conditioned blend fits over -- identical to
# Tier 0/Tier 1's own scope (docs/tier1-regional-weights-results.md), so
# region- and regime-conditioning are compared on the same underlying model.
REGIME_SOURCE_NAMES = ["graphcast", "hres", "ifs_ens_mean"]
N_MONTE_CARLO_SAMPLES = 500


def load_monsoon_phase_for_samples(
    obs: xr.Dataset, climatology: xr.Dataset, sample_values: np.ndarray
) -> xr.DataArray:
    """The real per-day monsoon active/break label
    (`weavr.regimes.classify_monsoon_active_break`), computed from the full
    real season (so the core-zone climatology normalization sees every
    JJAS day, not just this lead's aligned subset) and then restricted to
    exactly the samples this lead's aligned forecasts have, renamed onto
    the shared `sample` dimension `weavr.regime_weighting.fit_regime_weights`
    expects.
    """
    phase = classify_monsoon_active_break(obs["rain"], climatology["rain"])
    return phase.sel(time=sample_values).rename(time="sample")


def score_point_forecast(forecast: xr.DataArray, obs: xr.DataArray) -> dict:
    """CRPS/RMSE/bias of a deterministic point forecast -- see this
    module's docstring for why CRPS here is exactly `|forecast - obs|`
    (the CRPS of a degenerate, point-mass predictive distribution), not an
    approximation.
    """
    error = (forecast - obs).values
    abs_error = np.abs(error)
    valid = ~np.isnan(abs_error)
    n = int(valid.sum())
    if n == 0:
        return {
            "n_test_cells": 0,
            "crps_mm": float("nan"),
            "mse_mm2": float("nan"),
            "bias_mm": float("nan"),
        }
    return {
        "n_test_cells": n,
        "crps_mm": float(np.mean(abs_error[valid])),
        "mse_mm2": float(np.mean(error[valid] ** 2)),
        "bias_mm": float(np.mean(error[valid])),
    }


def score_point_forecast_by_bin(
    forecast: xr.DataArray, obs: xr.DataArray, rain_bin_labels: xr.DataArray
) -> dict[str, dict]:
    """Per-bin breakdown of `score_point_forecast` -- supplementary
    evidence for *where* the regime-conditioned model wins or loses, not a
    second go/no-go criterion (see this module's docstring).
    """
    per_bin: dict[str, dict] = {}
    for bin_label in RAIN_BIN_LABELS:
        cell_mask = rain_bin_labels == bin_label
        per_bin[bin_label] = score_point_forecast(
            forecast.where(cell_mask), obs.where(cell_mask)
        )
    return per_bin


def _domain_summary_from_stats(stats: dict, prefix: str) -> dict:
    mse = stats["mse_mm2"]
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")
    return {
        f"{prefix}_crps_mm": stats["crps_mm"],
        f"{prefix}_rmse_mm": rmse,
        f"{prefix}_bias_mm": stats["bias_mm"],
        f"{prefix}_n_test_cells": stats["n_test_cells"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-store", default="data/baseline_2020_jjas.zarr")
    parser.add_argument("--lagged-store", default="data/lagged_ensemble_inputs_2020_jjas.zarr")
    parser.add_argument("--ifs-ensemble-store", default="data/ifs_ens_2020_jjas.zarr")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--out-csv", default="results/tier3_regime_conditioned_baseline.csv")
    parser.add_argument(
        "--bin-out-csv", default="results/tier3_regime_conditioned_baseline_by_bin.csv"
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-monte-carlo", type=int, default=N_MONTE_CARLO_SAMPLES)
    args = parser.parse_args()
    n_samples = args.n_monte_carlo

    obs = xr.open_zarr(args.baseline_store, group="imd_observed", consolidated=True).load()
    climatology = load_climatology(args.climatology)
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    rng = np.random.default_rng(0)

    print("Tier 3 regime-conditioned baseline: issue #7's go/no-go vs Phase 4's real combiners")
    print(
        "Regime-conditioned model: OLS blend of graphcast/hres/ifs_ens_mean, "
        "conditioned on monsoon active/break"
    )

    domain_rows = []
    bin_rows = []

    for lead_hours in LEAD_HOURS:
        graphcast_ensemble = load_graphcast_ensemble(args.lagged_store, lead_hours)
        ifs_ensemble = load_ifs_ensemble(args.ifs_ensemble_store, lead_hours)
        hres_forecast = load_hres_forecast(args.baseline_store, lead_hours)
        forecasts, obs_aligned = align_all_sources(
            graphcast_ensemble, ifs_ensemble, hres_forecast, obs
        )

        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        try:
            train_mask, test_mask = seasonal_block_split(
                sample_times, test_fraction=args.test_fraction
            )
        except InsufficientTimeBlocksError as exc:
            print(f"[lead {lead_hours:>3}h] skipped -- {exc}")
            continue

        rain_bin_labels = classify_rain_bin(
            forecasts["graphcast"].mean(dim="member", skipna=True)
        )

        # -- Phase 4's own combiners, recomputed on this exact split --
        emos_results = {
            "graphcast": fit_emos_csg(
                forecasts["graphcast"], obs_aligned, rain_bin_labels, train_mask, source="graphcast"
            ),
            "ifs_ens": fit_emos_csg(
                forecasts["ifs_ens"], obs_aligned, rain_bin_labels, train_mask, source="ifs_ens"
            ),
        }
        bma_results = fit_hierarchical_bma(
            forecasts, obs_aligned, rain_bin_labels, region_labels, train_mask
        )

        # -- Tier 3: the regime-conditioned blend --
        regime_sources = {
            "graphcast": forecasts["graphcast"].mean(dim="member", skipna=True),
            "hres": forecasts["hres"],
            "ifs_ens_mean": forecasts["ifs_ens"].mean(dim="member", skipna=True),
        }
        regime_labels = load_monsoon_phase_for_samples(
            obs, climatology, obs_aligned["sample"].values
        )
        regime_weight_results = fit_regime_weights(
            regime_sources, obs_aligned, regime_labels, train_mask, sample_dim="sample"
        )
        regime_weight_series = build_regime_weight_series(
            regime_weight_results, regime_labels, REGIME_SOURCE_NAMES
        )
        regime_blend = blend_with_regime_weights(regime_sources, regime_weight_series)

        test_regime_blend = regime_blend.isel(sample=test_mask)
        test_obs = obs_aligned.isel(sample=test_mask)
        test_bins = rain_bin_labels.isel(sample=test_mask)

        regime_stats = score_point_forecast(test_regime_blend, test_obs)
        regime_bin_stats = score_point_forecast_by_bin(test_regime_blend, test_obs, test_bins)

        domain_row = {
            "lead_hours": lead_hours,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        domain_row.update(_domain_summary_from_stats(regime_stats, "regime"))

        emos_per_bin = {}
        for source_key, results in emos_results.items():
            per_bin = score_emos_source(
                results, forecasts[source_key], obs_aligned, rain_bin_labels, test_mask, rng,
                n_samples=n_samples,
            )
            emos_per_bin[source_key] = per_bin
            domain_row.update(_domain_summary(per_bin, f"emos_{source_key}"))
            for bin_label, stats in per_bin.items():
                bin_rows.append({
                    "lead_hours": lead_hours, "method": f"emos_{source_key}",
                    "bin": bin_label, **stats,
                })

        bma_per_cell = score_bma_cells(
            bma_results, forecasts, obs_aligned, rain_bin_labels, region_labels, test_mask, rng,
            n_samples=n_samples,
        )
        domain_row.update(_domain_summary(bma_per_cell, "bma"))

        for bin_label, stats in regime_bin_stats.items():
            bin_rows.append({
                "lead_hours": lead_hours, "method": "regime_conditioned",
                "bin": bin_label, **stats,
            })

        domain_rows.append(domain_row)
        print(
            f"[lead {lead_hours:>3}h] regime crps={domain_row['regime_crps_mm']:.2f}mm "
            f"emos_graphcast crps={domain_row.get('emos_graphcast_crps_mm', float('nan')):.2f}mm "
            f"emos_ifs_ens crps={domain_row.get('emos_ifs_ens_crps_mm', float('nan')):.2f}mm "
            f"bma crps={domain_row.get('bma_crps_mm', float('nan')):.2f}mm"
        )

    def _write_csv(path_str: str, rows: list[dict]) -> None:
        out_path = Path(path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_path}")

    _write_csv(args.out_csv, domain_rows)
    _write_csv(args.bin_out_csv, bin_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
