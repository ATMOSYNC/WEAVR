"""Phase 5's regime-stratified weighting (Tier 3, step 3) -- the simplest
defensible extension of Phase 3's static regional weighting
(`weavr.weighting.fit_region_weights`) to condition on a real regime
covariate (`weavr.regimes`) instead of (or alongside) spatial region.

Structurally the same problem `fit_region_weights` already solves --pool
points sharing a category, fit via OLS, clip negative weights and
renormalize, fall back to equal weights when a category is too sparse --
but the stratifying key is different in kind: a region label is a property
of the **gridpoint** (constant across every day), while a regime label is a
property of the **day** (constant across every gridpoint). This module
therefore pools every (train day, gridpoint) pair whose *day* falls in a
given regime category, rather than every (train day, gridpoint) pair whose
*gridpoint* falls in a given region -- everything downstream (the OLS fit,
clip-and-renormalize, and fallback thresholds) reuses
`weavr.weighting`'s own functions and `MIN_TRAIN_SAMPLES` constant directly,
not a re-derived threshold.

Per `docs/phase5-regime-conditioned-results.md`'s step 3.1 assessment, the
regime covariate used here is monsoon active/break
(`weavr.regimes.classify_monsoon_active_break`) -- the one covariate with
a real, checked (if modest, in this project's own small training set)
link to elevated heavy-rain cell counts, and the only one free to compute
from data already in this repo. MJO phase and monsoon-depression presence
(also built in step 2) showed no comparably clean pattern in that same
assessment and are not conditioned on here; either remains available to a
caller as a `regime_labels` array of its own (this module's fit function is
generic over any per-sample category label, not hardcoded to monsoon
phase), should a later step find better evidence for one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from weavr.weighting import (
    MIN_TRAIN_SAMPLES,
    clip_and_renormalize,
    fit_weights_least_squares,
)


@dataclass
class RegimeWeightResult:
    """One regime category's fitted weights, or its documented fallback --
    mirrors `weavr.weighting.RegionWeightResult` exactly (see that class's
    own docstring for the meaning of each field); `regime` replaces
    `region` as the stratifying key's name.
    """

    regime: str
    weights: dict[str, float]
    is_fallback: bool = False
    reason: str | None = None
    n_train_points: int = 0
    sources: tuple[str, ...] = field(default_factory=tuple)


def _equal_weights(sources: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(sources, 1.0 / len(sources))


def fit_regime_weights(
    forecasts: dict[str, xr.DataArray],
    obs: xr.DataArray,
    regime_labels: xr.DataArray,
    train_mask: np.ndarray,
    sample_dim: str = "sample",
    min_train_samples: int = MIN_TRAIN_SAMPLES,
) -> dict[str, RegimeWeightResult]:
    """Fit one weight per source, per regime category, at whatever single
    lead the caller's `forecasts`/`obs` already represent.

    `forecasts` maps source name to a `(sample_dim, latitude, longitude)`
    DataArray at one lead; `obs` is the matching IMD observation array;
    `regime_labels` is a `(sample_dim,)` array of per-day category labels
    (e.g. `weavr.regimes.classify_monsoon_active_break`'s own output --
    `"active"`/`"break"`/`""`); `train_mask` is a boolean array over
    `sample_dim` from whatever CV strategy the caller decided (per
    `docs/phase3-cv-and-regional-scheme.md`: `seasonal_block_split`).

    For each regime category, pools every `(train sample landing in that
    category, gridpoint)` pair -- across the *entire spatial domain*, since
    a regime label doesn't vary by gridpoint -- where every source and obs
    are simultaneously non-NaN into one design matrix, fits via
    `weavr.weighting.fit_weights_least_squares`, then
    `weavr.weighting.clip_and_renormalize`. Falls back to equal weights,
    flagged via `RegimeWeightResult.is_fallback`/`reason`, using the exact
    same three conditions `fit_region_weights` does (too few contributing
    train samples, a rank-deficient pooled design matrix, or every fitted
    weight clipping to zero) -- see that function's own docstring.
    """
    source_names = tuple(forecasts.keys())
    n_sources = len(source_names)

    train_sample_coords = obs[sample_dim].values[train_mask]
    regime_train = regime_labels.sel({sample_dim: train_sample_coords}).values
    category_values = np.unique(regime_labels.values)

    obs_train = obs.sel({sample_dim: train_sample_coords})
    forecast_train = {
        name: forecasts[name].sel({sample_dim: train_sample_coords}) for name in source_names
    }

    results: dict[str, RegimeWeightResult] = {}
    for category in category_values:
        sample_mask = regime_train == category  # (n_train_samples,)
        n_contributing_samples = int(sample_mask.sum())

        obs_category = obs_train.isel({sample_dim: sample_mask})
        forecast_stack = np.stack(
            [forecast_train[name].isel({sample_dim: sample_mask}).values for name in source_names],
            axis=-1,
        )  # (n_category_samples, latitude, longitude, n_sources)

        obs_flat = obs_category.values.reshape(-1)
        forecast_flat = forecast_stack.reshape(-1, n_sources)
        valid = ~np.isnan(obs_flat) & ~np.isnan(forecast_flat).any(axis=1)
        n_valid_points = int(valid.sum())

        if n_contributing_samples < min_train_samples or n_valid_points < n_sources:
            results[str(category)] = RegimeWeightResult(
                regime=str(category),
                weights=_equal_weights(source_names),
                is_fallback=True,
                reason=(
                    f"only {n_contributing_samples} train sample(s) with usable data "
                    f"in this regime category (need >= {min_train_samples})"
                ),
                n_train_points=n_valid_points,
                sources=source_names,
            )
            continue

        design = forecast_flat[valid]
        target = obs_flat[valid]

        if np.linalg.matrix_rank(design) < n_sources:
            results[str(category)] = RegimeWeightResult(
                regime=str(category),
                weights=_equal_weights(source_names),
                is_fallback=True,
                reason=(
                    "design matrix is rank-deficient (sources indistinguishable in this "
                    "regime category)"
                ),
                n_train_points=n_valid_points,
                sources=source_names,
            )
            continue

        raw_weights = fit_weights_least_squares(design, target)
        normalized = clip_and_renormalize(raw_weights)

        if normalized is None:
            results[str(category)] = RegimeWeightResult(
                regime=str(category),
                weights=_equal_weights(source_names),
                is_fallback=True,
                reason="every fitted weight clipped to zero (no source with positive skill)",
                n_train_points=n_valid_points,
                sources=source_names,
            )
            continue

        results[str(category)] = RegimeWeightResult(
            regime=str(category),
            weights=dict(zip(source_names, normalized.tolist(), strict=True)),
            is_fallback=False,
            reason=None,
            n_train_points=n_valid_points,
            sources=source_names,
        )

    return results


def build_regime_weight_series(
    weight_results: dict[str, RegimeWeightResult],
    regime_labels: xr.DataArray,
    source_names: list[str],
) -> dict[str, xr.DataArray]:
    """Turn per-regime-category fitted weights into a per-*sample* (day)
    weight series for each source -- the temporal analogue of
    `scripts/run_tier1_regional_baseline.py`'s `build_region_weight_grid`
    (which builds a per-*gridpoint* field instead, since region is spatial).
    Blending is then a single broadcasted multiply against `(sample,
    latitude, longitude)` forecasts, exactly as that script's own
    `blend_with_region_weights` does.
    """
    weight_series = {}
    for source in source_names:
        values = np.zeros(regime_labels.shape, dtype=float)
        for category, result in weight_results.items():
            values[regime_labels.values == category] = result.weights[source]
        weight_series[source] = xr.DataArray(
            values, coords=regime_labels.coords, dims=regime_labels.dims
        )
    return weight_series


def blend_with_regime_weights(
    forecasts: dict[str, xr.DataArray], weight_series: dict[str, xr.DataArray]
) -> xr.DataArray:
    """Weighted sum of every source's forecast, weighted by its per-sample
    (regime-derived) weight series -- `clip_and_renormalize` already
    guaranteed each category's weights sum to 1 (or fell back to an equal
    split that also sums to 1), so this is a plain weighted average, not a
    fresh renormalization. xarray broadcasts each `(sample,)` weight series
    against each `(sample, latitude, longitude)` forecast automatically.
    """
    terms = [forecasts[name] * weight_series[name] for name in forecasts]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total
