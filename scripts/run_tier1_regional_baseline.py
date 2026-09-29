#!/usr/bin/env python3
"""Run the Tier 1 regional-weights baseline: Phase 3's actual exit criterion.

The static/regional-skill-weighting counterpart to Phase 1's Tier 0 run
(scripts/run_tier0_baseline.py): fits a per-region, per-lead weight for each
precipitation-carrying forecast source (src/weavr/weighting.py, pooled via
src/weavr/regions.py's Sreekala & Babu 6-zone scheme), blends the sources
with those weights, and scores the blend with the same src/weavr/verify.py
metrics Tier 0 used -- against the *same held-out test split* as an
equal-weight mean computed here alongside it, so the comparison is a fair,
apples-to-apples one rather than a comparison against Tier 0's own recorded
numbers (which used an independently-computed split; see "Why the
equal-weight mean is recomputed here" below).

Scope, decided by what steps 1-3 already established, not re-litigated here:

- **Precipitation only, same 3 sources as Tier 0.** graphcast, hres, and
  ifs_ens_mean -- the only sources in the store that carry
  `total_precipitation_24hr` (pangu has none at all, per
  docs/baseline-store.md). Identical to run_tier0_baseline.py's own scope.
- **CV strategy: `seasonal_block_split` for every source, uniformly.** Per
  docs/phase3-cv-and-regional-scheme.md's step 1 decision -- GraphCast's
  WeatherBench 2 archive contains exactly one full JJAS season, ever, so
  true leave-one-year-out CV (Wanders & Wood's own recipe) is impossible
  for the one AI source/variable this project scores, and the user
  confirmed keeping one CV methodology across every source (rather than a
  mixed-CV path) to keep fitted weights comparable within the same blend.
- **Regional scheme: Sreekala & Babu's 6-zone pooling**
  (src/weavr/regions.py), per step 1's finding that Neal et al.'s
  30-pattern scheme is a temporal regime classification, not a spatial one,
  and doesn't fit this phase's need.
- **Weight fitting: src/weavr/weighting.py's `fit_region_weights`** --
  unconstrained OLS per Wanders & Wood (2016), negative weights clipped to
  zero (Wang et al. 2025) and renormalized to sum to 1, with a documented
  equal-weight fallback for regions with too few train samples or a
  rank-deficient design matrix.

## Why the equal-weight mean is recomputed here, not read from Tier 0's doc

docs/tier0-baseline-results.md's numbers are scored **domain-wide** (every
gridpoint pooled into one score) on a test split drawn from Tier 0's own
sample alignment. This script needs a **region-by-region** breakdown to
answer Phase 3's actual question ("does the weighted blend beat equal
weighting, region by region"), and no per-region equal-weight numbers exist
anywhere yet. Rather than reverse-engineering per-region scores out of
Tier 0's domain-wide CSV (not possible -- the underlying per-gridpoint
values aren't in that output) or assuming this run's train/test split lines
up exactly with Tier 0's own (same code, same inputs, so it should -- but
"should" is not "checked"), this script recomputes the equal-weight mean
itself, from the same aligned forecasts and the same split, so both blends
are scored against literally the same held-out samples. The domain-wide
Tier 1 number is cross-checked against Tier 0's own recorded number as a
sanity check (see docs/tier1-regional-weights-results.md).

## Per-region scoring: RMSE/bias only, not the full metric set

FSS (a spatial neighborhood filter) and the contingency scores (categorical
counts) are computed domain-wide only, exactly as Tier 0 did -- restricting
either to one region by masking other gridpoints to NaN would corrupt the
neighborhood-fraction convolution (a plain box filter has no NaN-aware
mode; masked cells would bias every neighborhood touching a region
boundary) or the contingency table's counts in ways not worth the added
complexity for what step 4 actually needs. RMSE, bias, ACC, and SEEPS are
genuinely pointwise-then-averaged (already `skipna=True` throughout
src/weavr/verify.py), so masking to one region before scoring correctly
restricts the average to that region's gridpoints -- verified by
`tests/test_run_tier1_regional_baseline.py`'s masking tests, not assumed.
ACC needs a per-gridpoint climatology already, so it is included per-region
too; SEEPS is included per-region for the same reason.

Usage:
    python scripts/run_tier1_regional_baseline.py
        [--store PATH] [--climatology PATH]
        [--out-csv PATH] [--region-out-csv PATH] [--weights-out-csv PATH]
        [--test-fraction F] [--neighborhood-size N]
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
from run_tier0_baseline import PRECIP_M_TO_MM, _align_to_imd_day  # noqa: E402

from weavr import verify as V  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.score_io import per_day_scores, write_per_day_scores  # noqa: E402
from weavr.splits import InsufficientTimeBlocksError, seasonal_block_split  # noqa: E402
from weavr.weighting import RegionWeightResult, fit_region_weights  # noqa: E402

PRECIP_VARIABLE = "total_precipitation_24hr"
LEAD_HOURS = [24, 48, 72, 96, 120]
# Identical to run_tier0_baseline.py's scope: the 3 of 4 baseline-store
# sources that actually carry total_precipitation_24hr (pangu has none).
FORECAST_SOURCE_NAMES = ["graphcast", "hres", "ifs_ens_mean"]
NEIGHBORHOOD_SIZE = 5  # grid cells (~1.25 deg edge) -- matches Tier 0.


def load_aligned_forecasts_and_obs(
    sources: dict[str, xr.Dataset],
    obs: xr.Dataset,
    variable: str,
    lead_hours: int,
) -> tuple[dict[str, xr.DataArray], xr.DataArray]:
    """Load, unit-convert, and IMD-day-align each source's forecast at one
    lead, then reindex all three onto a shared `sample` coordinate (an outer
    join -- a source missing a particular sample gets NaN there rather than
    being silently dropped) before restricting to samples with real IMD
    ground truth, exactly matching run_tier0_baseline.py's own alignment
    convention (`_align_to_imd_day`, `PRECIP_M_TO_MM`), reused rather than
    reimplemented.

    Any (sample, gridpoint) still NaN in one source after this (e.g. a
    source genuinely missing that week) is excluded per-region at the
    fitting step itself (src/weavr/weighting.py's own NaN-exclusion logic),
    not here.
    """
    raw_aligned = {
        name: _align_to_imd_day(
            sources[name][variable].sel(prediction_timedelta=lead_hours).load()
            * PRECIP_M_TO_MM,
            lead_hours,
        )
        for name in FORECAST_SOURCE_NAMES
    }
    aligned_arrays = xr.align(*raw_aligned.values(), join="outer")
    forecasts = dict(zip(raw_aligned.keys(), aligned_arrays, strict=True))

    sample_values = forecasts[FORECAST_SOURCE_NAMES[0]]["sample"].values
    obs_aligned = obs["rain"].reindex(time=sample_values).rename(time="sample")
    has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
    forecasts = {name: da.isel(sample=has_obs.values) for name, da in forecasts.items()}
    obs_aligned = obs_aligned.isel(sample=has_obs.values)
    return forecasts, obs_aligned


def build_region_weight_grid(
    weight_results: dict[str, RegionWeightResult],
    region_labels: xr.DataArray,
    source_names: list[str],
) -> dict[str, xr.DataArray]:
    """Turn per-region fitted weights into a per-gridpoint `(latitude,
    longitude)` weight field for each source, so blending is a single
    broadcasted multiply against `(sample, latitude, longitude)` forecasts
    rather than a per-region loop over the forecast data itself.
    """
    weight_grids = {}
    for source in source_names:
        values = np.zeros(region_labels.shape, dtype=float)
        for region, result in weight_results.items():
            values[region_labels.values == region] = result.weights[source]
        weight_grids[source] = xr.DataArray(
            values, coords=region_labels.coords, dims=region_labels.dims
        )
    return weight_grids


def blend_with_region_weights(
    forecasts: dict[str, xr.DataArray], weight_grids: dict[str, xr.DataArray]
) -> xr.DataArray:
    """Weighted sum of every source's forecast, weighted by its
    per-gridpoint (region-derived) weight field. `clip_and_renormalize`
    already guaranteed each region's weights sum to 1 (or fell back to an
    equal split that also sums to 1), so this is a plain weighted average,
    not a fresh renormalization.
    """
    terms = [forecasts[name] * weight_grids[name] for name in forecasts]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def equal_weight_blend(forecasts: dict[str, xr.DataArray]) -> xr.DataArray:
    """The Tier 0-equivalent unweighted mean, computed from this script's
    own aligned forecasts (not read from Tier 0's CSV) so it is scored
    against literally the same held-out test samples as the Tier 1 blend --
    see this module's docstring for why that matters.
    """
    stacked = xr.concat(list(forecasts.values()), dim="source")
    return stacked.mean(dim="source", skipna=True)


def score_blend(
    test_forecast: xr.DataArray,
    test_obs: xr.DataArray,
    climatology: xr.Dataset,
    thresholds: tuple[float, ...],
    neighborhood_size: int,
    *,
    include_spatial_and_categorical: bool = True,
) -> dict:
    """Score one already-split test forecast against IMD ground truth, using
    exactly run_tier0_baseline.py's metric set. `include_spatial_and_categorical`
    is False for the per-region breakdown -- see this module's docstring
    section on why FSS/contingency stay domain-wide only.
    """
    climatology_mean = climatology["rain"].mean(dim="time", skipna=True)
    result: dict[str, object] = {
        "rmse_mm": float(V.rmse(test_forecast, test_obs)),
        "bias_mm": float(V.bias(test_forecast, test_obs)),
        "acc": float(V.acc(test_forecast, test_obs, climatology_mean)),
        "seeps": float(
            V.seeps(test_forecast, test_obs, climatology["rain"], climatology_dim="time")
        ),
    }
    if include_spatial_and_categorical:
        result["fss"] = {
            t: float(
                V.fss(test_forecast, test_obs, threshold=t, neighborhood_size=neighborhood_size)
            )
            for t in thresholds
        }
        contingency = V.contingency_scores(test_forecast, test_obs, thresholds=thresholds)
        result["contingency"] = {
            t: {k: float(v) for k, v in s.items()} for t, s in contingency.items()
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/baseline_2020_jjas.zarr")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--out-csv", default="results/tier1_regional_baseline.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--region-out-csv", default="results/tier1_regional_baseline_by_region.csv")
    parser.add_argument("--weights-out-csv", default="results/tier1_regional_weights.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--neighborhood-size", type=int, default=NEIGHBORHOOD_SIZE)
    args = parser.parse_args()

    sources = {
        name: xr.open_zarr(args.store, group=name, consolidated=True)
        for name in FORECAST_SOURCE_NAMES
    }
    obs = xr.open_zarr(args.store, group="imd_observed", consolidated=True).load()
    climatology = load_climatology(args.climatology).load()
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)

    print(f"Tier 1 regional-weights baseline, scored against {args.store}'s imd_observed group")
    print(f"Regions: {sorted(np.unique(region_labels.values).tolist())}")
    print("CV strategy: seasonal_block_split for every source (docs/phase3-cv-and-regional-scheme)")

    domain_rows = []
    region_rows = []
    weight_rows = []

    for lead_hours in LEAD_HOURS:
        forecasts, obs_aligned = load_aligned_forecasts_and_obs(
            sources, obs, PRECIP_VARIABLE, lead_hours
        )
        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        try:
            train_mask, test_mask = seasonal_block_split(
                sample_times, test_fraction=args.test_fraction
            )
        except InsufficientTimeBlocksError as exc:
            print(f"[lead {lead_hours:>3}h] skipped -- {exc}")
            continue

        weight_results = fit_region_weights(
            forecasts, obs_aligned, region_labels, train_mask, sample_dim="sample"
        )
        weight_grids = build_region_weight_grid(
            weight_results, region_labels, FORECAST_SOURCE_NAMES
        )

        tier1_blend = blend_with_region_weights(forecasts, weight_grids)
        equal_blend = equal_weight_blend(forecasts)

        test_tier1 = tier1_blend.isel(sample=test_mask)
        test_equal = equal_blend.isel(sample=test_mask)
        test_obs = obs_aligned.isel(sample=test_mask)

        # Step 04 (additive): per-day scores on this same test split. The
        # equal-weight blend is Tier 0's, already written by
        # run_tier0_baseline.py, so only the Tier 1 blend is written here --
        # two files under one method name would let the scorecard silently
        # use whichever script ran last.
        write_per_day_scores(
            "tier1_regional",
            lead_hours,
            per_day_scores(test_tier1, test_obs, fold="test"),
            out_dir=args.results_dir,
        )

        tier1_result = score_blend(
            test_tier1, test_obs, climatology, V.IMD_RAIN_THRESHOLDS_MM, args.neighborhood_size
        )
        equal_result = score_blend(
            test_equal, test_obs, climatology, V.IMD_RAIN_THRESHOLDS_MM, args.neighborhood_size
        )
        domain_row = {
            "lead_hours": lead_hours,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "split": "seasonal_block_split",
        }
        def _flatten(prefix: str, result: dict) -> None:
            for key, value in result.items():
                if key == "contingency":
                    for t, metrics in value.items():
                        for metric_name, metric_value in metrics.items():
                            domain_row[f"{prefix}_{metric_name}_{t}"] = metric_value
                elif isinstance(value, dict):
                    for t, v in value.items():
                        domain_row[f"{prefix}_{key}_{t}"] = v
                else:
                    domain_row[f"{prefix}_{key}"] = value

        _flatten("tier1", tier1_result)
        _flatten("equal", equal_result)
        domain_rows.append(domain_row)
        print(f"[lead {lead_hours:>3}h] tier1 rmse={tier1_result['rmse_mm']:.2f}mm "
              f"equal rmse={equal_result['rmse_mm']:.2f}mm")

        for region, result in weight_results.items():
            weight_row = {
                "lead_hours": lead_hours,
                "region": region,
                "is_fallback": result.is_fallback,
                "reason": result.reason or "",
                "n_train_points": result.n_train_points,
            }
            for source in FORECAST_SOURCE_NAMES:
                weight_row[f"weight_{source}"] = result.weights[source]
            weight_rows.append(weight_row)

            region_mask = (region_labels == region).values
            region_tier1 = test_tier1.where(region_mask)
            region_equal = test_equal.where(region_mask)
            region_obs = test_obs.where(region_mask)
            tier1_region_result = score_blend(
                region_tier1, region_obs, climatology, V.IMD_RAIN_THRESHOLDS_MM,
                args.neighborhood_size, include_spatial_and_categorical=False,
            )
            equal_region_result = score_blend(
                region_equal, region_obs, climatology, V.IMD_RAIN_THRESHOLDS_MM,
                args.neighborhood_size, include_spatial_and_categorical=False,
            )
            region_rows.append({
                "lead_hours": lead_hours,
                "region": region,
                "is_fallback": result.is_fallback,
                "tier1_rmse_mm": tier1_region_result["rmse_mm"],
                "tier1_bias_mm": tier1_region_result["bias_mm"],
                "tier1_acc": tier1_region_result["acc"],
                "tier1_seeps": tier1_region_result["seeps"],
                "equal_rmse_mm": equal_region_result["rmse_mm"],
                "equal_bias_mm": equal_region_result["bias_mm"],
                "equal_acc": equal_region_result["acc"],
                "equal_seeps": equal_region_result["seeps"],
            })

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
    _write_csv(args.region_out_csv, region_rows)
    _write_csv(args.weights_out_csv, weight_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
