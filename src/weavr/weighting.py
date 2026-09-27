"""Phase 3's static/regional skill weighting (Tier 1) -- fits a per-source,
per-region weight and blends forecasts with it.

Recipe checked directly against the cited paper, not assumed from its name:
Wanders & Wood (2016, *Environmental Research Letters* 11 094007) fits
multi-model weights via **unconstrained multivariate linear regression** of
each model's forecast onto observations, separately per region and per
forecast lead/initialization time -- `docs/phase3-cv-and-regional-scheme.md`
already established that "season" collapses to a single value (JJAS 2020)
for this project, and lead is handled by the caller passing in one lead's
forecasts at a time, so this module's job is the per-region regression fit
itself.

The paper's own weights are **not** constrained to sum to 1 (they are
plain regression coefficients, no intercept, applied as
`blend = sum_i(w_i * model_i)`). Issue #5 separately cites Wang et al.
(2025) for clipping negative weights to zero -- a distinct, later step
layered on top of the Wanders & Wood fit, not part of it. This module
does both: fits via ordinary least squares (matching the paper), then
clips negative weights to zero, then **renormalizes the clipped weights to
sum to 1** -- a deliberate choice for this project (not the paper's own
convention) so the resulting blend stays interpretable as a weighted
average, comparable to Phase 1/2's equal-weight blends, rather than an
arbitrary-scale regression output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

# A region-lead cell needs at least this many distinct TRAIN samples (time
# steps) to attempt a fit at all -- matches src/weavr/splits.py's own
# minimum for a meaningful split (a single time step can't support any
# notion of "fit on this, generalize to that").
MIN_TRAIN_SAMPLES = 2


@dataclass
class RegionWeightResult:
    """One region's fitted weights, or its documented fallback.

    `is_fallback=True` means the fit was not attempted (or was attempted
    and judged unusable) and `weights` is instead an equal split across
    every source -- visible and inspectable, not silently indistinguishable
    from a well-supported fit. `reason` is set only when `is_fallback` is
    True, naming exactly why (too few train samples, or a rank-deficient
    design matrix -- e.g. every source has near-identical values in this
    region, giving the regression nothing to distinguish).
    """

    region: str
    weights: dict[str, float]
    is_fallback: bool = False
    reason: str | None = None
    n_train_points: int = 0
    sources: tuple[str, ...] = field(default_factory=tuple)


def _equal_weights(sources: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(sources, 1.0 / len(sources))


def fit_weights_least_squares(
    forecast_matrix: np.ndarray, obs_vector: np.ndarray
) -> np.ndarray:
    """Unconstrained multivariate linear regression, no intercept -- the
    Wanders & Wood (2016) recipe: solve `forecast_matrix @ w ~= obs_vector`
    for `w` via ordinary least squares (`numpy.linalg.lstsq`), matching the
    paper's own no-intercept formulation (`blend = sum_i(w_i * model_i)`).

    `forecast_matrix` is `(n_points, n_sources)`, `obs_vector` is
    `(n_points,)`. Returns the raw, unclipped, unnormalized weights --
    clipping and renormalization are separate, deliberate steps (see
    `clip_and_renormalize`), not folded into the fit itself.
    """
    weights, _residuals, _rank, _singular_values = np.linalg.lstsq(
        forecast_matrix, obs_vector, rcond=None
    )
    return weights


def clip_and_renormalize(weights: np.ndarray) -> np.ndarray | None:
    """Clip negative weights to zero (Wang et al. 2025), then renormalize
    the clipped weights to sum to 1 so the blend stays a proper weighted
    average -- a deliberate choice for this project, not the source
    regression's own convention (see this module's docstring).

    Returns `None` if every weight clips to zero (nothing left to
    renormalize) -- a degenerate case the caller must handle explicitly
    (see `fit_region_weights`'s fallback), not divide-by-zero into NaN.
    """
    clipped = np.clip(weights, a_min=0.0, a_max=None)
    total = clipped.sum()
    if total <= 0.0:
        return None
    return clipped / total


def fit_region_weights(
    forecasts: dict[str, xr.DataArray],
    obs: xr.DataArray,
    region_labels: xr.DataArray,
    train_mask: np.ndarray,
    sample_dim: str = "sample",
    min_train_samples: int = MIN_TRAIN_SAMPLES,
) -> dict[str, RegionWeightResult]:
    """Fit one weight per source, per region, at whatever single lead the
    caller's `forecasts`/`obs` already represent.

    `forecasts` maps source name to a `(sample_dim, latitude, longitude)`
    DataArray at one lead; `obs` is the matching IMD observation array;
    `region_labels` is `src/weavr/regions.py`'s `(latitude, longitude)`
    region-label array; `train_mask` is a boolean array over `sample_dim`
    from whatever CV strategy the caller decided (per
    `docs/phase3-cv-and-regional-scheme.md`: `seasonal_block_split`,
    applied uniformly to every source). "Season" is not a separate
    parameter here -- the caller passes in already-season-sliced data,
    matching that same doc's finding that season is currently a single
    value throughout.

    For each region, pools every `(train sample, gridpoint)` pair in that
    region where every source and obs are simultaneously non-NaN into one
    design matrix, fits via `fit_weights_least_squares`, then
    `clip_and_renormalize`. Falls back to equal weights, flagged via
    `RegionWeightResult.is_fallback`/`reason`, when the region has fewer
    than `min_train_samples` distinct train samples with any usable data,
    when the pooled design matrix is rank-deficient (fewer independent
    equations than sources -- e.g. every source is ~identical in this
    region), or when every fitted weight clips to zero.
    """
    source_names = tuple(forecasts.keys())
    n_sources = len(source_names)
    region_values = np.unique(region_labels.values)

    train_sample_coords = obs[sample_dim].values[train_mask]

    results: dict[str, RegionWeightResult] = {}
    for region in region_values:
        region_mask = (region_labels == region).values  # (latitude, longitude)

        obs_train = obs.sel({sample_dim: train_sample_coords})
        obs_region = obs_train.where(region_mask)

        forecast_stack = np.stack(
            [
                forecasts[name].sel({sample_dim: train_sample_coords}).where(region_mask).values
                for name in source_names
            ],
            axis=-1,
        )  # (n_train_samples, latitude, longitude, n_sources)

        obs_flat = obs_region.values.reshape(-1)
        forecast_flat = forecast_stack.reshape(-1, n_sources)

        valid = ~np.isnan(obs_flat) & ~np.isnan(forecast_flat).any(axis=1)
        n_valid_points = int(valid.sum())

        # Distinct train samples actually contributing a usable point --
        # not just a raw point count, since one sample can contribute many
        # (correlated) spatial points but the CV split only ever varied at
        # the sample level.
        n_time = obs_region.sizes[sample_dim]
        n_space = obs_flat.size // max(n_time, 1)
        valid_by_sample = valid.reshape(n_time, n_space).any(axis=1)
        n_contributing_samples = int(valid_by_sample.sum())

        if n_contributing_samples < min_train_samples or n_valid_points < n_sources:
            results[str(region)] = RegionWeightResult(
                region=str(region),
                weights=_equal_weights(source_names),
                is_fallback=True,
                reason=(
                    f"only {n_contributing_samples} train sample(s) with usable data "
                    f"in this region (need >= {min_train_samples})"
                ),
                n_train_points=n_valid_points,
                sources=source_names,
            )
            continue

        design = forecast_flat[valid]
        target = obs_flat[valid]

        if np.linalg.matrix_rank(design) < n_sources:
            results[str(region)] = RegionWeightResult(
                region=str(region),
                weights=_equal_weights(source_names),
                is_fallback=True,
                reason="design matrix is rank-deficient (sources indistinguishable in this region)",
                n_train_points=n_valid_points,
                sources=source_names,
            )
            continue

        raw_weights = fit_weights_least_squares(design, target)
        normalized = clip_and_renormalize(raw_weights)

        if normalized is None:
            results[str(region)] = RegionWeightResult(
                region=str(region),
                weights=_equal_weights(source_names),
                is_fallback=True,
                reason="every fitted weight clipped to zero (no source with positive skill)",
                n_train_points=n_valid_points,
                sources=source_names,
            )
            continue

        results[str(region)] = RegionWeightResult(
            region=str(region),
            weights=dict(zip(source_names, normalized.tolist(), strict=True)),
            is_fallback=False,
            reason=None,
            n_train_points=n_valid_points,
            sources=source_names,
        )

    return results
