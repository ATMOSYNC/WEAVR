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
from scipy.special import gammaincc
from scipy.stats import gamma as gamma_dist

from weavr.rain_bins import RAIN_BIN_LABELS
from weavr.renormalize import renormalize_weights
from weavr.verify import crps as ensemble_crps

# Same threshold, same reasoning, as weavr.emos.MIN_TRAIN_DAYS_PER_BIN --
# a mixture (or component regression) fit from fewer independent train
# days is fitting temporal noise, not a regime.
MIN_TRAIN_DAYS_PER_BIN = 5

# Cells drawn per Monte Carlo block by `sample_bma_mixture`. Each live
# `(cells, n_samples)` float64 temporary is ~8 * cells * n_samples bytes and
# several are alive at once, so this caps the transient cost of a block at a
# few hundred MB for the default n_samples=500 regardless of how many cells a
# (rain bin, region) group covers.
MAX_CELLS_PER_CHUNK = 20_000

# `weavr.verify.crps` delegates to `xskillscore.crps_ensemble`, whose working
# set grows with cells * n_samples**2 -- it forms a pairwise member-difference
# tensor. For one (rain bin, region) group of a real 129x135 grid that is
# gigabytes at the default n_samples=500, and it was what the OOM killer was
# terminating Tier 2 on. Scoring CRPS in cell blocks, with the block chosen so
# the pairwise tensor stays under this budget, makes the default sample count
# usable without changing any number: the members are independent per cell, so
# the score of a block does not depend on which other cells are in it.
MAX_CRPS_PAIRWISE_BYTES = 256 << 20


def crps_chunk_cells(n_samples: int, budget: int = MAX_CRPS_PAIRWISE_BYTES) -> int:
    """Cells per CRPS block that keeps the pairwise tensor inside `budget`."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    per_cell = 8 * n_samples * n_samples
    return max(1, int(budget // per_cell))


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

    #: Upper bound on this component's predicted cube-root mean, set from the
    #: largest *training* observation in cube-root space.
    #:
    #: The mean regressor is the only unbounded quantity in this fit. Every
    #: other one is guarded -- `p0` is clipped into (0, 1), the variance into
    #: `[max(intercept, _TINY), inf)` -- because `gamma_mean_intercept` and
    #: `gamma_mean_slope` come straight from `np.linalg.lstsq` with no
    #: constraint. When `forecast_ct` has near-zero spread within a single
    #: (bin, region) training cell, that regression is ill-conditioned and
    #: returns a large slope, so a test forecast slightly above the training
    #: range predicts an enormous `mean_ct`. Since the density uses `mean_ct**3`
    #: as the gamma scale, the predictive mean explodes.
    #:
    #: Measured on the two-season daily base: one (bin, region) per fold
    #: produced this -- `heavy`xSI on the 2018 fold at CRPS 145mm with bias
    #: +277mm, and `heavy`xNE1 on 2020 at CRPS 154mm with bias +197mm -- while
    #: the other five regions in the same bin and lead were fine (17-40mm).
    #: Observed daily accumulations top out near 400mm, so an RMSE of 667-748mm
    #: in those rows is not a forecast error at all. Bounding the prediction at
    #: the training range is the standard remedy and leaves well-posed fits
    #: untouched, because a correctly conditioned regression already predicts
    #: inside it.
    max_mean_ct: float = float("inf")


@dataclass
class BmaFitResult:
    """One (bin, region) cell's fitted BMA mixture, or its documented
    fallback -- mirrors `weavr.emos.CensoredShiftedGammaResult` /
    `weavr.weighting.RegionWeightResult`.

    `is_fallback=True` means too few train days contributed to this cell;
    `weights` is then an equal split across every source and `components`
    is empty, visible via `is_fallback`/`reason` rather than a silently
    meaningless EM result. `renormalize_bma_for_present_sources` also sets
    `is_fallback=True` for a *different* reason -- only one source survived
    at prediction time -- but leaves that one source's real fitted
    component in place; `components` (empty vs. non-empty) is what actually
    distinguishes "no real fit exists" from "a real fit exists but
    degenerated to one source for this day", not `is_fallback` alone (see
    `sample_bma_mixture`).
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
    # The cap the prediction will be held to. Derived from this cell's own
    # training observations, so it is as loose as the data allows and as tight
    # as the data requires -- see `BmaComponentFit.max_mean_ct`.
    max_mean_ct = float(obs_ct_wet.max())
    predicted_mean = gamma_mean_intercept + gamma_mean_slope * forecast_ct_wet
    residual_sq = (obs_ct_wet - predicted_mean) ** 2

    if route == "ensemble_dressing":
        spread_ct_wet = _cube_root(forecast_spread[wet])  # type: ignore[index]
        gamma_variance_intercept, gamma_variance_slope = _fit_least_squares_line(
            spread_ct_wet, residual_sq
        )
        gamma_variance_intercept = max(gamma_variance_intercept, _TINY)
    else:
        gamma_variance_intercept = max(float(np.mean(residual_sq)), _TINY)
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
        max_mean_ct=max_mean_ct,
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
    # Held inside the training range and non-negative. See
    # `BmaComponentFit.max_mean_ct` for the measurement that motivated it.
    mean_ct = np.clip(mean_ct, _TINY, component.max_mean_ct)

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


def renormalize_bma_for_present_sources(
    result: BmaFitResult, present_sources: tuple[str, ...] | list[str]
) -> BmaFitResult:
    """Adapts a fitted BMA mixture to a prediction day where one or more of
    its fit-time sources didn't arrive -- BMA's weights are already the
    flat `{source: mixture_weight}` shape `weavr.renormalize.
    renormalize_weights` operates on, but unlike `weavr.weighting`/
    `weavr.regime_weighting` (a plain weighted sum with no other per-source
    state), a missing source's fitted `BmaComponentFit` -- its own
    predictive distribution -- has to be dropped too, not just its mixture
    share: `sample_bma_mixture`/`score_bma` both iterate `result.weights`
    and index `result.components` by the same source names, so leaving a
    missing source's component behind after renormalizing its weight away
    would still try to sample from a distribution with no real data behind
    it for that day.

    A cell already `is_fallback` at fit time is returned unchanged --
    prediction-time renormalization only has meaning for a mixture that
    was actually fit; an equal-weight-mixture-of-point-masses-at-zero
    fallback has no real per-source components to drop from in the first
    place.
    """
    if result.is_fallback:
        return result

    renormalized = renormalize_weights(result.weights, present_sources)
    components = {s: result.components[s] for s in renormalized.present_sources}

    return BmaFitResult(
        bin_label=result.bin_label,
        region=result.region,
        weights=renormalized.weights,
        components=components,
        is_fallback=renormalized.is_fallback,
        reason=renormalized.reason,
        n_train_days=result.n_train_days,
    )


def _sample_bma_block(
    result: BmaFitResult,
    forecast_mean: dict[str, np.ndarray],
    forecast_spread: dict[str, np.ndarray | None],
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Draws from the fitted mixture for one flat block of cells.

    `forecast_mean` / `forecast_spread` must be 1-D and aligned. Kept separate
    from `sample_bma_mixture` so that the chunking there is the only place
    that decides block boundaries.
    """
    sources = list(result.weights.keys())
    weight_arr = np.array([result.weights[s] for s in sources])
    weight_arr = weight_arr / weight_arr.sum()

    shape = (len(forecast_mean[sources[0]]),)
    component_choice = rng.choice(len(sources), size=shape + (n_samples,), p=weight_arr)
    samples = np.zeros(shape + (n_samples,))

    for index, source in enumerate(sources):
        component = result.components[source]
        p0, mean_ct, variance_ct = _component_predictive_params(
            component, forecast_mean[source], forecast_spread.get(source)
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


def sample_bma_mixture(
    result: BmaFitResult,
    forecast_mean: dict[str, np.ndarray],
    forecast_spread: dict[str, np.ndarray | None],
    rng: np.random.Generator,
    n_samples: int = 500,
    max_cells_per_chunk: int = MAX_CELLS_PER_CHUNK,
) -> np.ndarray:
    """Draws `n_samples` per prediction cell from the fitted (or fallback)
    mixture -- the sampling step `score_bma` turns into a Monte Carlo CRPS
    estimate (see this module's docstring for why no closed form exists).

    Returns an array shaped `forecast's own shape + (n_samples,)`. A
    result with no fitted components at all draws from a point mass at zero
    (matching `weavr.emos.predict_csgd_params`'s own fallback behavior) --
    checked via `not result.components` rather than `result.is_fallback`
    directly, since `renormalize_bma_for_present_sources` can flag
    `is_fallback=True` for a real, non-empty single-component mixture (only
    one source survived a missing/late source at prediction time) that must
    still be sampled from, not zeroed out.

    Cells are drawn in chunks of at most `max_cells_per_chunk` so that peak
    memory stays bounded by the chunk size instead of growing with the
    (bin, region) group. A single unchunked draw of `C` cells keeps ~7
    full-size `(C, n_samples)` temporaries alive at once -- component choice,
    samples, the zero mask, the gamma draw, the cube, the masked draw and the
    `where` output -- which is ~57 bytes per cell-draw. Over a real
    129x135 grid that is gigabytes per group, and the step 07/11/12/13
    evaluation runners were being killed by the OOM killer part-way through
    Tier 2. Chunking changes the exact draw values (each chunk is seeded from
    a child sequence) but not the distribution, and stays deterministic for a
    given `max_cells_per_chunk`.
    """
    any_mean = next(iter(forecast_mean.values()))
    shape = np.asarray(any_mean).shape

    if not result.components:
        return np.zeros(shape + (n_samples,))

    if max_cells_per_chunk is None or max_cells_per_chunk <= 0:
        max_cells_per_chunk = int(np.prod(shape)) or 1

    n_cells = int(np.prod(shape)) if shape else 1

    # `_sample_bma_block` works on flat 1-D cell axes, so both the chunked and
    # the single-block path go through the same layout and only differ in how
    # many cells a block holds.
    flat_mean = {s: np.asarray(v).reshape(-1) for s, v in forecast_mean.items()}
    flat_spread = {
        s: (None if v is None else np.asarray(v).reshape(-1)) for s, v in forecast_spread.items()
    }

    if n_cells <= max_cells_per_chunk:
        return _sample_bma_block(result, flat_mean, flat_spread, rng, n_samples).reshape(
            shape + (n_samples,)
        )

    # Seed every chunk up front so the result depends only on the parent
    # generator and the chunk layout, not on the order the work happens in.
    n_chunks = -(-n_cells // max_cells_per_chunk)
    chunk_seeds = rng.integers(0, 2**63 - 1, size=n_chunks)

    out = np.empty((n_cells, n_samples), dtype=float)
    for chunk_index, start in enumerate(range(0, n_cells, max_cells_per_chunk)):
        stop = min(start + max_cells_per_chunk, n_cells)
        block_mean = {s: v[start:stop] for s, v in flat_mean.items()}
        block_spread = {
            s: (None if v is None else v[start:stop]) for s, v in flat_spread.items()
        }
        out[start:stop] = _sample_bma_block(
            result,
            block_mean,
            block_spread,
            np.random.default_rng(int(chunk_seeds[chunk_index])),
            n_samples,
        )
    return out.reshape(shape + (n_samples,))


def _bma_sources(result: BmaFitResult) -> tuple[str, ...]:
    """Sources in a stable order: the fitted components, else the weight keys.

    A fallback cell has weights but no components, so the weight keys are the
    only list that exists for every result. Iterating a dict's keys keeps the
    order stable within a run, which matters because the mixture sums in that
    order and the weights are floats.
    """
    if result.components:
        return tuple(result.components)
    return tuple(result.weights)


#: Value grid the analytic CRPS is integrated on: 1 mm to 150 mm, then 5 mm to
#: 1000 mm. Beyond the top of the grid the CDF is 1 while the step term is 0, so
#: the integrand is already ~0 and truncation contributes nothing measurable.
BMA_CRPS_GRID_MM = np.concatenate(
    [np.arange(0.0, 150.0 + 1.0, 1.0), np.arange(155.0, 1000.0 + 5.0, 5.0)]
)


def bma_mixture_cdf(
    result: BmaFitResult,
    forecast_mean: dict[str, np.ndarray],
    forecast_spread: dict[str, np.ndarray | None],
    values: np.ndarray,
) -> np.ndarray:
    """Exact predictive CDF of the mixture, evaluated at `values`.

    Each component's predictive variable is `X**3` with `X ~ Gamma(kappa,
    theta)`, and cubing is strictly increasing, so
    `P(X**3 <= v) = P(X <= v**(1/3)) = gamma.cdf(v**(1/3))` for `v >= 0`. The
    component's own point mass at zero adds to that for every `v >= 0`, giving
    `F(v) = p0 + (1 - p0) * gamma.cdf(v**(1/3))`, and the mixture CDF is the
    weighted sum of those.

    `values` is either a shared 1-D grid -- giving the shape `(*cells, G)` -- or
    one value per cell with a trailing singleton axis (`obs[..., None]`), giving
    the cells' own shape. `gamma_dist.cdf` broadcasts the two against each
    other, so no reshaping is needed and neither calling convention can
    double-count a rank.
    """
    values = np.asarray(values, dtype=float)
    cut = np.cbrt(np.maximum(values, 0.0))
    values_ndim = values.ndim
    cell_shape = np.asarray(forecast_mean[_bma_sources(result)[0]], dtype=float).shape
    cell_ndim = len(cell_shape)
    shape = cell_shape
    pad = (1,) * max(0, cell_ndim + 1 - values_ndim)
    padded = cut.reshape(pad + cut.shape)
    if not result.components:
        # A fallback cell is a point mass at zero, so its CDF is 1 at every
        # non-negative value.
        # Same two calling conventions as the fitted path: a shared grid gives
        # `(*cells, G)`, a per-cell value gives the cells' own shape.
        out_shape = (
            shape + values.shape if values_ndim <= cell_ndim
            else np.broadcast_shapes(shape, values.shape)
        )
        return np.ones(out_shape, dtype=float)

    total = None
    for source in _bma_sources(result):
        p0, mean_ct, variance_ct = _component_predictive_params(
            result.components[source], forecast_mean[source], forecast_spread.get(source)
        )
        mean_ct_safe = np.clip(mean_ct, _TINY, None)
        # Both operands need to end up at rank `ndim + 1`: the per-cell
        # parameters get a *trailing* singleton so their cell axes stay leading,
        # and the evaluation points get *leading* singletons so theirs stay
        # trailing. NumPy aligns from the right, so a (ndim, ndim) parameter
        # block cannot broadcast against a (G,) grid without this.
        kappa = np.clip(mean_ct_safe**2 / variance_ct, _TINY, None).reshape(shape + (1,))
        theta = np.clip(variance_ct / mean_ct_safe, _TINY, None).reshape(shape + (1,))
        # `p0 +` is the point mass at zero, and it is not optional. A dry cell has
        # p0 ~ 1, so dropping it leaves the CDF near 0 across the whole grid and
        # CRPS integrates to ~294 mm instead of ~2.7 mm.
        contribution = result.weights[source] * (
            p0.reshape(shape + (1,))
            + (1.0 - p0).reshape(shape + (1,))
            * gamma_dist.cdf(padded, kappa, scale=theta)
        )
        total = contribution if total is None else total + contribution
    return total


def _cumulative_trapezoid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Running integral of `values` over `grid`; the output starts at 0."""
    widths = np.diff(grid)
    increments = 0.5 * (values[..., :-1] + values[..., 1:]) * widths
    return np.concatenate(
        [np.zeros(values.shape[:-1] + (1,), dtype=float), np.cumsum(increments, axis=-1)],
        axis=-1,
    )


def bma_crps(
    result: BmaFitResult,
    forecast_mean: dict[str, np.ndarray],
    forecast_spread: dict[str, np.ndarray | None],
    obs: np.ndarray,
    grid: np.ndarray = BMA_CRPS_GRID_MM,
) -> np.ndarray:
    """Exact per-cell CRPS, integrated from the mixture CDF rather than sampled.

    Uses the split form `CRPS = int_0^y F^2 + int_y^inf (1 - F)^2`, in which
    both integrands are continuous, and handles the single interval that
    straddles `y` separately using the CDF evaluated exactly at `y`.

    The one-sided form `int (F - 1{x >= y})^2` has a jump at `x = y`, and a plain
    trapezoid across that interval is biased by about half a grid step -- enough
    to score a point mass at zero against `obs = 3` as 2.5 instead of 3.0.
    Splitting at `y` removes that bias entirely.

    This replaces the sample-based CRPS for the same reason the predictive mean
    was replaced: `Y = X**3` is heavy-tailed enough that 500 draws cannot
    represent it. On `heavy` x SI the sampled CRPS reads 146.9 mm where this
    value is 26.8 mm, a factor of 5.5 -- large enough that the Step 07 output
    gate raised four advisories describing a degenerate fit that does not exist.
    Every other region measured agrees with the exact value to 1.00x.

    It is also **per-cell**, which the sampled version was not: CRPS used to be
    reduced to a scalar before `score_bma_cells` assigned it into a per-cell
    grid, so every cell in a (bin, region) group received the group average.
    """
    obs = np.asarray(obs, dtype=float)
    cdf = bma_mixture_cdf(result, forecast_mean, forecast_spread, grid)
    cdf_at_obs = bma_mixture_cdf(result, forecast_mean, forecast_spread, obs[..., None])[..., 0]

    cum_f2 = _cumulative_trapezoid(cdf**2, grid)
    cum_g2 = _cumulative_trapezoid((1.0 - cdf) ** 2, grid)

    last = grid.size - 1
    j = np.clip(np.searchsorted(grid, obs, side="right") - 1, 0, last)
    j_up = np.minimum(j + 1, last)

    j3 = j[..., None]
    below_f2 = np.take_along_axis(cum_f2, j3, axis=-1)[..., 0]
    below_g2 = np.take_along_axis(cum_g2, j3, axis=-1)[..., 0]
    f_lo = np.take_along_axis(cdf, j3, axis=-1)[..., 0]
    f_hi = np.take_along_axis(cdf, j_up[..., None], axis=-1)[..., 0]

    g_lo = grid[j]
    g_hi = grid[j_up]
    left = below_f2 + 0.5 * (f_lo**2 + cdf_at_obs**2) * np.clip(obs - g_lo, 0.0, None)
    right = (cum_g2[..., last] - below_g2) + 0.5 * (
        (1.0 - cdf_at_obs) ** 2 + (1.0 - f_hi) ** 2
    ) * np.clip(g_hi - obs, 0.0, None)
    # An observation above the grid is entirely in the saturated region, where
    # F is 1 and the second integral is zero, so the first integral grows by the
    # full excess.
    right = right + np.clip(obs - grid[last], 0.0, None)
    return np.clip(left + right, 0.0, None)


def bma_analytic_predictive_mean(
    result: BmaFitResult,
    forecast_mean: dict[str, np.ndarray],
    forecast_spread: dict[str, np.ndarray | None],
) -> np.ndarray:
    """Exact predictive mean of the mixture, with no Monte Carlo.

    The mixture draws `X ~ Gamma(kappa, theta)` on the cube-root scale and then
    cubes it, so the predictive variable is `Y = X**3` and its mean is
    `E[X**3] = theta**3 * kappa * (kappa + 1) * (kappa + 2)` -- **not**
    `(E[X])**3`. Jensen guarantees the two differ, and for the fat-tailed
    components fitted here they differ a lot.

    This exists because the mean was previously estimated as the mean of
    `n_samples` draws. That estimator is unusable here: `Var(X**3)` is enormous,
    so a 500-draw sample mean is dominated by a handful of draws and reported
    327.6 mm where the true mean is 62.2 mm. On the `heavy` x SI cell at lead24
    that turned a genuine +13.5 mm bias into a reported +277.4 mm, which then
    looked like a fitted-component defect and sent two earlier root-cause
    theories after the wrong culprit.

    The parameters are exactly the ones `sample_bma_mixture` uses, so this is
    the same distribution -- just not sampled.
    """
    if not result.components:
        # A fallback cell's density is a point mass at zero, so its mean is zero
        # -- `sample_bma_mixture` documents the same fallback.
        return np.zeros(np.asarray(forecast_mean[_bma_sources(result)[0]], dtype=float).shape)
    total = None
    for index, source in enumerate(_bma_sources(result)):
        component = result.components[source]
        p0, mean_ct, variance_ct = _component_predictive_params(
            component, forecast_mean[source], forecast_spread.get(source)
        )
        mean_ct_safe = np.clip(mean_ct, _TINY, None)
        kappa = mean_ct_safe**2 / variance_ct
        theta = variance_ct / mean_ct_safe
        moment3 = theta**3 * kappa * (kappa + 1.0) * (kappa + 2.0)
        weight = result.weights[source]
        contribution = weight * (1.0 - p0) * moment3
        total = contribution if total is None else total + contribution
    if total is None:
        return np.zeros((), dtype=float)
    return total


def bma_exceedance_probability(
    result: BmaFitResult,
    forecast_mean: dict[str, np.ndarray],
    forecast_spread: dict[str, np.ndarray | None],
    threshold: float,
) -> np.ndarray:
    """Exact `P(Y > threshold)` for the mixture, with no Monte Carlo.

    `Y = X**3` is strictly increasing in `X`, so the exceedance event maps to
    `X > threshold ** (1/3)` and the answer is the gamma survival function of
    the cube-root threshold, summed over the mixture with the point mass at
    zero folded in.

    This replaces `(samples > threshold).mean(axis=-1)`. With 500 draws that
    estimator quantises every probability to multiples of 1/500 and, in the
    upper tail where the pre-registered H3 thresholds live, is dominated by
    sampling noise -- so the heavy-rain probabilities that H3 is decided on
    carried roughly +/-0.002 of Monte Carlo error per cell before any bootstrap.
    """
    sources = _bma_sources(result)
    shape = np.asarray(forecast_mean[sources[0]], dtype=float).shape
    if not result.components:
        return np.zeros(shape, dtype=float)
    total = np.zeros(shape, dtype=float)
    cut = float(threshold) ** (1.0 / 3.0)
    for source in _bma_sources(result):
        component = result.components[source]
        p0, mean_ct, variance_ct = _component_predictive_params(
            component, forecast_mean[source], forecast_spread.get(source)
        )
        mean_ct_safe = np.clip(mean_ct, _TINY, None)
        kappa = np.clip(mean_ct_safe**2 / variance_ct, _TINY, None)
        theta = np.clip(variance_ct / mean_ct_safe, _TINY, None)
        survival = gammaincc(kappa, cut / theta)
        total = total + result.weights[source] * ((1.0 - p0) * survival)
    return np.clip(total, 0.0, 1.0)


def score_bma_and_mean(
    result: BmaFitResult,
    forecast_mean: dict[str, xr.DataArray],
    forecast_spread: dict[str, xr.DataArray | None],
    obs: xr.DataArray,
    rng: np.random.Generator | None = None,
    n_samples: int = 500,
    member_dim: str = "member",
    probability_grids: dict[float, np.ndarray] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """CRPS *and* the predictive mean of the fitted (or fallback) BMA mixture
    against real `obs`, from a **single** Monte Carlo draw set.

    Both numbers have to come from the same draws: `crps_mm`, `rmse_mm` and
    `bias_mm` are all derived from one realisation of the fitted mixture, so
    estimating them from two independent draw sets (by calling `score_bma` and
    then `sample_bma_mixture` again) makes them mutually inconsistent, injects
    extra Monte Carlo noise into the mean-based metrics, and doubles both the
    runtime and the peak memory of the `(n_cells, n_samples)` sample array --
    which is what actually exhausts RAM on a 17 GB machine.

    Returns `(crps, predictive_mean)`, both unreduced `xr.DataArray`s shaped
    like `obs`.
    """
    rng = rng if rng is not None else np.random.default_rng()

    mean_arrays = {source: da.values for source, da in forecast_mean.items()}
    spread_arrays = {
        source: (da.values if da is not None else None) for source, da in forecast_spread.items()
    }

    # Every summary this function returns is exact, per-cell, and seed-free:
    # CRPS from the CDF integral, the mean from E[X**3], and any exceedance
    # probability from the gamma survival function. None needs draws.
    #
    # That is not an optimisation. `Y = X**3` is heavy-tailed enough that 500
    # draws could not represent it -- the sampled mean read 327.6 mm against a
    # true 62.2 mm, and the sampled CRPS read 146.9 mm against a true 26.8 mm,
    # the latter enough for the Step 07 gate to report four advisories
    # describing a degenerate fit that does not exist.
    obs_values = obs.values if hasattr(obs, "values") else np.asarray(obs)
    crps_values = bma_crps(result, mean_arrays, spread_arrays, obs_values)

    if probability_grids is not None:
        for threshold in probability_grids:
            probability_grids[threshold][:] = bma_exceedance_probability(
                result, mean_arrays, spread_arrays, float(threshold)
            )

    analytic_mean = bma_analytic_predictive_mean(result, mean_arrays, spread_arrays)
    predictive_mean = xr.DataArray(
        np.broadcast_to(analytic_mean, crps_values.shape).copy(),
        dims=obs.dims, coords=obs.coords,
    )
    return xr.DataArray(crps_values, dims=obs.dims, coords=obs.coords), predictive_mean


def _ensemble_crps_chunked(  # noqa: D401
    samples: np.ndarray, obs: xr.DataArray, member_dim: str
) -> xr.DataArray:
    """Retained only for the sampled path `sample_bma_mixture` users build.

    `score_bma_and_mean` no longer calls this: it integrates the CDF instead.
    Anything still scoring from draws can use it, but note it reduces to a
    scalar and so cannot fill a per-cell grid.
    """
    """`weavr.verify.crps` over `samples`, blocked over the leading (cell)
    axes so `xskillscore`'s pairwise member-difference tensor stays bounded.

    `samples` is `obs`'s shape plus a trailing ensemble axis. Members are
    independent per cell, so a cell's score does not depend on which other
    cells share its block; the per-cell scores are then reduced exactly as
    `ensemble_crps(dim=None)` reduces them, so the returned number is
    unchanged -- this only caps peak memory.
    """
    n_samples = samples.shape[-1]
    block = crps_chunk_cells(n_samples)
    n_cells = int(np.prod(samples.shape[:-1])) if samples.ndim > 1 else 1

    if n_cells <= block:
        sample_da = xr.DataArray(samples, dims=(*obs.dims, member_dim), coords=obs.coords)
        return ensemble_crps(sample_da, obs, member_dim=member_dim)

    out = np.empty(samples.shape[:-1], dtype=float)
    obs_flat = np.asarray(obs.values).reshape(-1)
    flat = samples.reshape(-1, n_samples)
    for start in range(0, n_cells, block):
        stop = min(start + block, n_cells)
        chunk_obs = xr.DataArray(obs_flat[start:stop], dims=("cell",))
        chunk_da = xr.DataArray(flat[start:stop], dims=("cell", member_dim))
        out.reshape(-1)[start:stop] = np.asarray(
            ensemble_crps(chunk_da, chunk_obs, member_dim=member_dim).values
        ).reshape(-1)
    # `ensemble_crps(dim=None)` averages over every non-member axis; match it
    # so callers see the same scalar they always have.
    return xr.DataArray(float(out.mean()))


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

    Callers that also need the predictive mean should use
    `score_bma_and_mean`, which shares one draw set between the two instead of
    sampling twice.
    """
    crps, _predictive_mean = score_bma_and_mean(
        result,
        forecast_mean,
        forecast_spread,
        obs,
        rng=rng,
        n_samples=n_samples,
        member_dim=member_dim,
    )
    return crps
