"""Hierarchical Bayesian Model Averaging (BMA) -- Phase 4's named comparison
combiner (issue #6), fit per (rain-intensity bin, region) cell so it can be
honestly compared against `weavr.emos`'s EMOS-CSG rather than defaulting to
EMOS-CSG untested.

Recipe checked directly against the two BMA papers step 1 (`docs/phase4-
data-and-combiner-scope.md`) already distinguished, not the "1-size-fits-
all" one:

- **Raftery, Gneiting, Balabdaoui & Polakowski (2005)**, *Monthly Weather
  Review* 133:1155-1174 -- the *general* BMA framework this module does
  **not** implement: a weighted mixture of Gaussian component predictive
  distributions, built for temperature/pressure, wrong for precipitation's
  mixed discrete/continuous structure (a positive probability of exactly
  zero, a right-skewed continuous part).
- **Sloughter, Raftery, Gneiting & Fraley (2007)**, *Monthly Weather
  Review* 135:3209-3220 -- the precipitation-specific extension this
  module implements: each component source's own predictive PDF is a
  mixture of a point mass at zero and a power-transformed (cube root)
  gamma distribution, with the mixture (BMA) weights across sources still
  fit by EM to maximize training-period log-likelihood.

"Hierarchical" is issue #6's own phrase for fitting per (bin x region)
stratum -- not a Bayesian hierarchical-model sense -- reusing
`weavr.regions.assign_regions`'s existing India zones (the same ones
`weavr.weighting.fit_region_weights` pools) rather than a second,
inconsistent regional scheme, and `weavr.rain_bins.RAIN_BIN_LABELS`'s
existing bins, matching `weavr.emos`'s own per-bin granularity so step 6
can compare the two combiners on identical strata.

Per-source component route, checked explicitly rather than assumed
uniform (per `docs/phase4-data-and-combiner-scope.md`'s decision that
GraphCast's lagged pseudo-ensemble and the real IFS 50-member ensemble are
fit independently as real ensembles, while HRES stays deterministic in
every store):

- **Ensemble sources** (GraphCast, IFS -- any forecast carrying a member
  dimension): the component's gamma variance (on the cube-root scale) is
  regressed on that source's own per-cell ensemble spread, a real,
  flow-dependent uncertainty signal.
- **Deterministic sources** (HRES): Sloughter et al.'s own "kernel
  dressing" route for models with no ensemble -- the component's gamma
  variance is instead a single fixed value, fit from the pooled training
  residuals (the empirical dressing kernel's bandwidth), since there is no
  per-cell spread statistic to regress against.

Scope simplification, documented rather than silently assumed: Sloughter
et al.'s own zero-probability logistic regression includes a
"fraction of ensemble members forecasting exactly zero" predictor for
ensemble sources; this module uses only the (cube-root transformed)
ensemble-mean forecast as the logistic predictor, uniformly across
sources, to keep one regression form shared by every route -- the
route-specific difference this module actually implements is the
variance term above, the one step 1/5 identifies as the real per-source
distinction (a deterministic model literally has no per-cell ensemble
statistic to draw a flow-dependent variance from).

Mixture weights are fit by the standard two-stage BMA procedure (Raftery
et al. 2005; Sloughter et al. 2007): each component's own regression
parameters are fit first (independently, from that source's own training
data), then EM iterates only over the mixture weights, holding those
per-component parameters fixed -- not a joint optimization over both at
once.

No closed-form CRPS exists for a general finite mixture of point-mass-
plus-power-transformed-gamma components (unlike `weavr.emos.csgd_crps`'s
single-component closed form), so `score_bma` estimates it via Monte Carlo
sampling from the fitted mixture, reusing `weavr.verify.crps`'s own
ensemble-CRPS machinery on the samples as a synthetic ensemble -- exactly
the sanctioned fallback ("closed form if one exists, else a numerical
substitute") this project's other combiner already documents.

Sparse/degenerate-cell fallback mirrors `weavr.emos`'s own pattern
(`is_fallback`/`reason`, `MIN_TRAIN_DAYS_PER_BIN = 5`, the same threshold
`docs/phase4-data-and-combiner-scope.md` found necessary) -- a
(bin, region) cell with too few distinct contributing train days falls
back to an equal-weight mixture of point masses at zero rather than a
silently over- or under-converged EM result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy import optimize
from scipy.stats import gamma as gamma_dist

from weavr.rain_bins import RAIN_BIN_LABELS
from weavr.verify import crps as ensemble_crps

# Same threshold, same reasoning, as weavr.emos.MIN_TRAIN_DAYS_PER_BIN --
# a mixture (or component regression) fit from fewer independent train
# days is fitting temporal noise, not a regime.
MIN_TRAIN_DAYS_PER_BIN = 5

_TINY = 1e-6
_CUBE_ROOT_POWER = 1.0 / 3.0


def _cube_root(x: np.ndarray) -> np.ndarray:
    """Sloughter et al. (2007)'s power transformation (cube root),
    applied to nonnegative precipitation values/forecasts only."""
    return np.clip(x, 0.0, None) ** _CUBE_ROOT_POWER


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class BmaComponentFit:
    """One source's own pre-fit point-mass-plus-gamma predictive model
    (Sloughter et al. 2007), fit independently of the mixture weights.

    `route` is `"ensemble_dressing"` (variance regressed on this source's
    own per-cell ensemble spread) or `"kernel_dressing"` (a single fixed
    variance, fit from pooled training residuals -- Sloughter et al.'s own
    route for a deterministic model with no ensemble to draw spread from).
    All regression coefficients operate on the cube-root-transformed
    scale, per Sloughter et al.'s own power transformation.
    """

    source: str
    route: str
    zero_intercept: float
    zero_slope: float
    gamma_mean_intercept: float
    gamma_mean_slope: float
    gamma_variance_intercept: float
    gamma_variance_slope: float


@dataclass
class BmaFitResult:
    """One (bin, region) cell's fitted BMA mixture, or its documented
    fallback -- mirrors `weavr.emos.CensoredShiftedGammaResult` /
    `weavr.weighting.RegionWeightResult`.

    `is_fallback=True` means too few train days contributed to this cell;
    `weights` is then an equal split across every source and `components`
    is empty, visible via `is_fallback`/`reason` rather than a silently
    meaningless EM result.
    """

    bin_label: str
    region: str
    weights: dict[str, float]
    components: dict[str, BmaComponentFit] = field(default_factory=dict)
    is_fallback: bool = False
    reason: str | None = None
    n_train_days: int = 0


def _equal_weights(sources: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(sources, 1.0 / len(sources))


def _fit_logistic(predictor: np.ndarray, is_zero: np.ndarray) -> tuple[float, float]:
    """MLE logistic regression of `is_zero` on `predictor` (two
    parameters: intercept, slope) -- Sloughter et al.'s zero-probability
    model, minimized here via `scipy.optimize` rather than a hand-rolled
    gradient (a two-parameter negative log-likelihood is cheap either way,
    and this avoids a separate iteratively-reweighted-least-squares
    implementation for a model this small).
    """

    def negative_log_likelihood(params: np.ndarray) -> float:
        intercept, slope = params
        p = np.clip(_sigmoid(intercept + slope * predictor), 1e-8, 1 - 1e-8)
        return -float(np.mean(is_zero * np.log(p) + (1 - is_zero) * np.log(1 - p)))

    result = optimize.minimize(negative_log_likelihood, x0=[0.0, 0.0], method="Nelder-Mead")
    intercept, slope = result.x
    return float(intercept), float(slope)


def _fit_least_squares_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _fit_component(
    source: str,
    forecast_mean: np.ndarray,
    forecast_spread: np.ndarray | None,
    obs: np.ndarray,
) -> BmaComponentFit:
    route = "ensemble_dressing" if forecast_spread is not None else "kernel_dressing"
    forecast_ct = _cube_root(forecast_mean)
    is_zero = (obs <= 0.0).astype(float)
    zero_intercept, zero_slope = _fit_logistic(forecast_ct, is_zero)

    wet = obs > 0.0
    if int(wet.sum()) < 2:
        # Too few wet training cells for this source to regress a real
        # gamma mean/variance from -- a flat, small fallback keeps this
        # component's density well-defined without crashing the mixture
        # EM; the overall cell's is_fallback (set by the caller when the
        # whole cell is sparse) is the real, user-visible flag.
        gamma_mean_intercept = float(np.mean(_cube_root(obs))) if obs.size else 0.0
        return BmaComponentFit(
            source, route, zero_intercept, zero_slope, gamma_mean_intercept, 0.0, _TINY, 0.0
        )

    obs_ct_wet = _cube_root(obs[wet])
    forecast_ct_wet = forecast_ct[wet]
    gamma_mean_intercept, gamma_mean_slope = _fit_least_squares_line(
        forecast_ct_wet, obs_ct_wet
    )
    predicted_mean = gamma_mean_intercept + gamma_mean_slope * forecast_ct_wet
    residual_sq = (obs_ct_wet - predicted_mean) ** 2

    if route == "ensemble_dressing":
        spread_ct_wet = _cube_root(forecast_spread[wet])  # type: ignore[index]
        gamma_variance_intercept, gamma_variance_slope = _fit_least_squares_line(
            spread_ct_wet, residual_sq
        )
        gamma_variance_intercept = max(gamma_variance_intercept, _TINY)
    else:
        gamma_variance_intercept = float(max(np.mean(residual_sq), _TINY))
        gamma_variance_slope = 0.0

    return BmaComponentFit(
        source,
        route,
        zero_intercept,
        zero_slope,
        gamma_mean_intercept,
        gamma_mean_slope,
        gamma_variance_intercept,
        gamma_variance_slope,
    )


def _component_predictive_params(
    component: BmaComponentFit,
    forecast_mean: np.ndarray,
    forecast_spread: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (p0, mean_ct, variance_ct) -- this component's zero-
    probability and cube-root-scale gamma mean/variance at new forecasts.
    """
    forecast_ct = _cube_root(forecast_mean)
    zero_logit = component.zero_intercept + component.zero_slope * forecast_ct
    p0 = np.clip(_sigmoid(zero_logit), 1e-8, 1 - 1e-8)
    mean_ct = component.gamma_mean_intercept + component.gamma_mean_slope * forecast_ct

    if component.route == "ensemble_dressing" and forecast_spread is not None:
        spread_ct = _cube_root(forecast_spread)
        variance_ct = np.clip(
            component.gamma_variance_intercept + component.gamma_variance_slope * spread_ct,
            _TINY,
            None,
        )
    else:
        variance_ct = np.full_like(forecast_ct, max(component.gamma_variance_intercept, _TINY))

    return p0, mean_ct, variance_ct


def _component_density(
    component: BmaComponentFit,
    forecast_mean: np.ndarray,
    forecast_spread: np.ndarray | None,
    obs: np.ndarray,
) -> np.ndarray:
    """This component's predictive PDF/PMF evaluated at real `obs` -- the
    zero-probability mass at `obs == 0`, else the power-transformed gamma
    density (with the cube-root transform's Jacobian) at `obs > 0`. Mixed
    discrete/continuous, per Sloughter et al. (2007)'s own likelihood.
    """
    p0, mean_ct, variance_ct = _component_predictive_params(
        component, forecast_mean, forecast_spread
    )

    obs_ct = _cube_root(obs)
    mean_ct_safe = np.clip(mean_ct, _TINY, None)
    kappa = mean_ct_safe**2 / variance_ct
    theta = variance_ct / mean_ct_safe
    gamma_pdf = gamma_dist.pdf(
        obs_ct, a=np.clip(kappa, _TINY, None), scale=np.clip(theta, _TINY, None)
    )
    jacobian = _CUBE_ROOT_POWER * np.clip(obs, _TINY, None) ** (_CUBE_ROOT_POWER - 1)

    continuous_density = np.clip((1 - p0) * gamma_pdf * jacobian, _TINY, None)
    return np.where(obs <= 0.0, p0, continuous_density)


def _em_fit_weights(
    component_densities: dict[str, np.ndarray], n_iterations: int = 50
) -> dict[str, float]:
    """Standard EM for a finite mixture's weights, holding every
    component's own density fixed (the two-stage BMA procedure -- see
    this module's docstring): E-step computes each training point's
    responsibility per source, M-step sets each weight to its mean
    responsibility, renormalized to sum to 1.
    """
    sources = list(component_densities.keys())
    density_matrix = np.clip(
        np.stack([component_densities[s] for s in sources], axis=1), _TINY, None
    )
    weights = np.full(len(sources), 1.0 / len(sources))
    for _ in range(n_iterations):
        weighted = density_matrix * weights[None, :]
        total = np.clip(weighted.sum(axis=1, keepdims=True), _TINY, None)
        responsibilities = weighted / total
        weights = responsibilities.mean(axis=0)
        weights = weights / weights.sum()
    return dict(zip(sources, weights.tolist(), strict=True))


def fit_hierarchical_bma(
    forecasts: dict[str, xr.DataArray],
    obs: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    region_labels: xr.DataArray,
    train_mask: np.ndarray,
    sample_dim: str = "sample",
    member_dim: str = "member",
    min_train_days: int = MIN_TRAIN_DAYS_PER_BIN,
    em_iterations: int = 50,
) -> dict[tuple[str, str], BmaFitResult]:
    """Fits one BMA mixture per (rain bin, region) cell.

    `forecasts` maps source name to a DataArray at one lead -- with a
    `member_dim` dimension for an ensemble source (GraphCast, IFS: fit via
    the "ensemble_dressing" route) or without it for a deterministic
    source (HRES: "kernel_dressing"), per this module's docstring.
    `rain_bin_labels` is `weavr.rain_bins.classify_rain_bin`'s label array;
    `region_labels` is `weavr.regions.assign_regions`'s `(latitude,
    longitude)` array; `train_mask` is a boolean array over `sample_dim`
    from `seasonal_block_split` (per `docs/phase4-data-and-combiner-scope.md`).

    For each (bin, region) pair, pools every (train sample, gridpoint)
    cell where every source and obs are simultaneously non-NaN, fits each
    source's own component (`_fit_component`) independently, then fits the
    mixture weights across sources via EM. Falls back to an equal-weight
    mixture, flagged via `BmaFitResult.is_fallback`/`reason`, when the cell
    has fewer than `min_train_days` distinct contributing train days.
    """
    source_names = tuple(forecasts.keys())
    train_sample_coords = obs[sample_dim].values[train_mask]

    obs_train = obs.sel({sample_dim: train_sample_coords}).values
    bin_train = rain_bin_labels.sel({sample_dim: train_sample_coords}).values

    mean_train: dict[str, np.ndarray] = {}
    spread_train: dict[str, np.ndarray | None] = {}
    for source in source_names:
        da = forecasts[source]
        if member_dim in da.dims:
            mean_train[source] = (
                da.mean(dim=member_dim, skipna=True).sel({sample_dim: train_sample_coords}).values
            )
            spread_train[source] = (
                da.std(dim=member_dim, ddof=1, skipna=True)
                .sel({sample_dim: train_sample_coords})
                .values
            )
        else:
            mean_train[source] = da.sel({sample_dim: train_sample_coords}).values
            spread_train[source] = None

    region_values = np.unique(region_labels.values)
    n_time = bin_train.shape[0]
    n_space = bin_train.size // max(n_time, 1)

    base_valid = ~np.isnan(obs_train)
    for source in source_names:
        base_valid = base_valid & ~np.isnan(mean_train[source])

    results: dict[tuple[str, str], BmaFitResult] = {}
    for bin_label in RAIN_BIN_LABELS:
        for region in region_values:
            region_mask = (region_labels == region).values  # (latitude, longitude)
            cell_mask = (bin_train == bin_label) & region_mask[None, :, :] & base_valid

            n_contributing_days = int(cell_mask.reshape(n_time, n_space).any(axis=1).sum())
            key = (bin_label, str(region))

            if n_contributing_days < min_train_days:
                results[key] = BmaFitResult(
                    bin_label=bin_label,
                    region=str(region),
                    weights=_equal_weights(source_names),
                    is_fallback=True,
                    reason=(
                        f"only {n_contributing_days} train day(s) with usable data in this "
                        f"cell (need >= {min_train_days})"
                    ),
                    n_train_days=n_contributing_days,
                )
                continue

            obs_cells = obs_train[cell_mask]
            components: dict[str, BmaComponentFit] = {}
            component_densities: dict[str, np.ndarray] = {}
            for source in source_names:
                mean_cells = mean_train[source][cell_mask]
                source_spread = spread_train[source]
                spread_cells = source_spread[cell_mask] if source_spread is not None else None
                component = _fit_component(source, mean_cells, spread_cells, obs_cells)
                components[source] = component
                component_densities[source] = _component_density(
                    component, mean_cells, spread_cells, obs_cells
                )

            weights = _em_fit_weights(component_densities, n_iterations=em_iterations)
            results[key] = BmaFitResult(
                bin_label=bin_label,
                region=str(region),
                weights=weights,
                components=components,
                is_fallback=False,
                reason=None,
                n_train_days=n_contributing_days,
            )

    return results


def sample_bma_mixture(
    result: BmaFitResult,
    forecast_mean: dict[str, np.ndarray],
    forecast_spread: dict[str, np.ndarray | None],
    rng: np.random.Generator,
    n_samples: int = 500,
) -> np.ndarray:
    """Draws `n_samples` per prediction cell from the fitted (or fallback)
    mixture -- the sampling step `score_bma` turns into a Monte Carlo CRPS
    estimate (see this module's docstring for why no closed form exists).

    Returns an array shaped `forecast's own shape + (n_samples,)`. A
    fallback result draws from a point mass at zero (matching
    `weavr.emos.predict_csgd_params`'s own fallback behavior).
    """
    any_mean = next(iter(forecast_mean.values()))
    shape = np.asarray(any_mean).shape

    if result.is_fallback:
        return np.zeros(shape + (n_samples,))

    sources = list(result.weights.keys())
    weight_arr = np.array([result.weights[s] for s in sources])
    weight_arr = weight_arr / weight_arr.sum()

    component_choice = rng.choice(len(sources), size=shape + (n_samples,), p=weight_arr)
    samples = np.zeros(shape + (n_samples,))

    for index, source in enumerate(sources):
        component = result.components[source]
        p0, mean_ct, variance_ct = _component_predictive_params(
            component, np.asarray(forecast_mean[source]), forecast_spread.get(source)
        )
        mean_ct_safe = np.clip(mean_ct, _TINY, None)
        kappa = mean_ct_safe**2 / variance_ct
        theta = variance_ct / mean_ct_safe

        broadcast_shape = shape + (n_samples,)
        p0_b = np.broadcast_to(p0[..., None], broadcast_shape)
        kappa_b = np.broadcast_to(kappa[..., None], broadcast_shape)
        theta_b = np.broadcast_to(theta[..., None], broadcast_shape)

        is_zero_draw = rng.uniform(size=broadcast_shape) < p0_b
        gamma_draw = rng.gamma(np.clip(kappa_b, _TINY, None), np.clip(theta_b, _TINY, None))
        continuous_draw = gamma_draw**3  # invert the cube-root transform

        drawn = np.where(is_zero_draw, 0.0, continuous_draw)
        samples = np.where(component_choice == index, drawn, samples)

    return samples


def score_bma(
    result: BmaFitResult,
    forecast_mean: dict[str, xr.DataArray],
    forecast_spread: dict[str, xr.DataArray | None],
    obs: xr.DataArray,
    rng: np.random.Generator | None = None,
    n_samples: int = 500,
    member_dim: str = "member",
) -> xr.DataArray:
    """CRPS of the fitted (or fallback) BMA mixture against real `obs` --
    the same interface shape as `weavr.emos.score_csgd` (an unreduced
    `xr.DataArray` of per-cell scores), so step 6 can call both
    combiners identically. Estimated via Monte Carlo sampling (see this
    module's docstring) and `weavr.verify.crps`'s ensemble-CRPS machinery.
    """
    rng = rng if rng is not None else np.random.default_rng()

    mean_arrays = {source: da.values for source, da in forecast_mean.items()}
    spread_arrays = {
        source: (da.values if da is not None else None) for source, da in forecast_spread.items()
    }

    samples = sample_bma_mixture(result, mean_arrays, spread_arrays, rng, n_samples=n_samples)
    sample_da = xr.DataArray(
        samples,
        dims=(*obs.dims, member_dim),
        coords=obs.coords,
    )
    return ensemble_crps(sample_da, obs, member_dim=member_dim)
