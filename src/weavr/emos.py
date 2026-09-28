"""EMOS-CSG (Ensemble Model Output Statistics, censored-shifted-gamma) --
Phase 4's named default combiner (issue #6), fit separately per rain-
intensity bin (`weavr.rain_bins.classify_rain_bin`) and per ensemble source.

Recipe checked directly against the two papers issue #6 and
`docs/phase4-data-and-combiner-scope.md` cite, not a remembered "some gamma
thing":

- **Scheuerer & Hamill (2015)**, *Monthly Weather Review* 143:4578-4596 --
  the censored, shifted gamma distribution (CSGD) itself: a gamma
  distribution with shape kappa and scale theta (related to its own
  mean/std mu/sigma via kappa=mu^2/sigma^2, theta=sigma^2/mu, their eq. 1),
  shifted left by delta<=0 and left-censored at zero (their eq. 2), fit by
  **minimizing mean CRPS** (not maximum likelihood). The paper states a
  closed-form CRPS exists (their eq. 10) but its derivation is in an
  online-only supplemental appendix this project doesn't have access to,
  and the paper's own equation, recovered via OCR from the published PDF,
  did not reproduce the correct value when checked against brute-force
  numerical integration of CRPS's raw definition (their eq. 9) -- caught
  empirically, not assumed correct from a single transcription. Rather
  than ship an unverified formula, `csgd_crps` below instead uses a
  from-scratch derivation, decomposing the censored CRPS into the
  well-established closed-form CRPS of the *uncensored* gamma distribution
  (matching the R `scoringRules` package's `crps_gamma`, itself checked
  against numerical integration first) minus a censoring correction term
  (a bounded integral with a known probabilistic meaning --
  `E[(c - max(U, U'))+]` for two iid unit-scale Gamma(kappa) variables --
  evaluated via fixed-node Gauss-Legendre quadrature rather than a second
  unverified closed form). The final result is verified in
  `tests/test_emos.py` against the same brute-force numerical integration,
  matching to floating-point precision across a range of parameters.
- **Baran & Nemoda (2016)**, *Environmetrics* 27:280-292 -- the companion
  paper confirming the same CSGD structure (kappa/theta/shift, left-censored
  at zero) and reporting it beats both the raw ensemble and a BMA model on
  precipitation test cases; the same model `HEPPI/example_EMOS_fit.R`
  itself fits (`model = "csg0"`), per
  `docs/phase4-data-and-combiner-scope.md`.

Scope simplification, documented rather than silently assumed: Scheuerer &
Hamill's own full regression model (their eq. 14-15) uses a
neighborhood-weighted, quantile-mapped ensemble mean, a "probability of
precipitation" predictor, and a precipitable-water predictor -- none of
which this project computes (no neighborhood pooling exists anywhere in
weavr, and no precipitable-water variable is in any baseline store). This
module instead fits their **own stated "most basic" regression** (eq.
11-12: location linear in the ensemble mean, scale proportional to
sqrt(location), plus a real flow-dependent spread term per eq. 15) using
predictors weavr actually has: the per-cell ensemble mean and per-cell
ensemble standard deviation across members (`ensemble_forecast.std(dim=
member_dim)`) -- a direct substitute for the paper's neighborhood-pooled
mean-absolute-difference statistic, both being ensemble-dispersion
measures. The paper's own eq. 11/12 divide by climatological references
purely "to normalize the regression parameters" and states this "does not
change the actual model" -- so this module drops that normalization and
fits the mathematically equivalent, unnormalized form directly:

    location   = a1 + a2 * ensemble_mean
    scale      = a3 * sqrt(max(location, eps)) + a4 * ensemble_spread
    shift      = delta_cl (fixed; see below)

per Scheuerer & Hamill section 4c's own choice ("We fix delta_s = delta_
cl,s and model the conditional CSGDs as deviations from the climatological
CSGD"), fit by minimizing mean CRPS over the bin's train cells.

The climatological shift/location/scale (delta_cl, mu_cl, sigma_cl) --
their "unconditional precipitation accumulations" step (section 4b) -- is
itself a CRPS-minimizing CSGD fit, with no predictors, to the bin's
training observations, reusing the exact same `csgd_crps` machinery (an
intercept-only special case of the same regression), not a separately
invented moment estimator.

Fit is per (ensemble source, rain bin) pair -- `fit_emos_csg` is source-
agnostic (the caller passes one source's `ensemble_forecast` at a time),
matching `docs/phase4-data-and-combiner-scope.md`'s decision to fit
GraphCast's lagged pseudo-ensemble and the real IFS 50-member ensemble
independently rather than pooling their members (their spread
characteristics genuinely differ -- GraphCast's lagged ensemble is
measured, in `docs/phase2-ensemble-baseline-results.md`, to be strongly
under-dispersive).

Degenerate/sparse-bin fallback mirrors `src/weavr/weighting.py`'s
`RegionWeightResult` pattern (`is_fallback`/`reason`, never a bare
exception or a silently overfit result): a bin with fewer than
`MIN_TRAIN_DAYS_PER_BIN` distinct contributing train days (the same
threshold `docs/phase4-data-and-combiner-scope.md` found necessary -- e.g.
the "extreme" bin, never fittable at any lead), or whose climatological fit
finds essentially no wet days at all (probability of precipitation below
0.005, the same threshold Scheuerer & Hamill use for their own "extremely
dry grid point" fallback), falls back to a plain point-mass-at-zero
predictive distribution instead.

No `weavr.renormalize` adaptation applies to this module, checked rather
than assumed: `fit_emos_csg` fits one source's own `CensoredShiftedGammaResult`
at a time (see this module's docstring on why sources are never pooled) and
produces no cross-source weight of any kind -- there is nothing here for a
missing source to redistribute onto another source, unlike
`weavr.weighting`/`weavr.regime_weighting`'s flat weight dict or
`weavr.bma`'s mixture weights. A missing/late source at prediction time
just means the caller (`scripts/run_tier2_hierarchical_baseline.py`'s
`score_emos_source`, one call per source) skips that source's own call for
the day it's missing; the other source's independently-fit EMOS-CSG result
is entirely unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy import optimize
from scipy.special import beta as beta_fn
from scipy.stats import gamma as gamma_dist

from weavr.rain_bins import RAIN_BIN_LABELS

# A bin needs at least this many distinct TRAIN days with usable data to fit
# a real combiner from -- the same threshold, for the same reason (fitting
# temporal noise, not a regime, from too few independent days), that
# docs/phase4-data-and-combiner-scope.md already measured per bin/lead.
MIN_TRAIN_DAYS_PER_BIN = 5

# Scheuerer & Hamill (2015), section 4b: below this probability-of-
# precipitation, even their simple preliminary climatology estimates are
# judged unreliable, and a fallback is used instead of a real fit.
MIN_PROBABILITY_OF_PRECIPITATION = 0.005

# Numerical floor for shape/scale/std parameters -- keeps the gamma
# distribution and the closed-form CRPS well-defined for a
# near-zero-variance bin (e.g. "dry") instead of dividing by zero.
_TINY = 1e-6


@dataclass
class CensoredShiftedGammaResult:
    """One (source, bin)'s fitted EMOS-CSG regression, or its documented
    fallback -- mirrors `src/weavr/weighting.py`'s `RegionWeightResult`.

    `is_fallback=True` means no real fit was attempted (too few train days)
    or the attempted climatological fit was judged degenerate (probability
    of precipitation below `MIN_PROBABILITY_OF_PRECIPITATION`); the
    predictive distribution is then a plain point mass at zero, not a
    fitted CSGD. `coefficients` holds `a1`-`a4` (see this module's
    docstring) and is empty when `is_fallback` is True. `shift` is the
    fixed climatological delta_cl (0.0 when falling back).
    """

    bin_label: str
    source: str
    shift: float
    climatological_mean: float
    climatological_std: float
    coefficients: dict[str, float] = field(default_factory=dict)
    is_fallback: bool = False
    reason: str | None = None
    n_train_days: int = 0


def _gamma_cdf(x: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """F_kappa(x): CDF of a unit-scale gamma distribution with shape
    `shape`, clipped to a tiny positive floor -- `scipy.stats.gamma` is
    undefined for shape<=0, which a degenerate (near-zero-variance) fit
    could otherwise reach during optimization. Zero for x<0 (`gamma.cdf`
    already returns 0 there, since the gamma's support starts at 0).
    """
    return gamma_dist.cdf(x, a=np.clip(shape, _TINY, None))


def _crps_gamma_std(shape: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of an *uncensored*, unit-scale Gamma(`shape`)
    distribution against `y` -- matches the R `scoringRules` package's
    `crps_gamma(y, shape, scale=1)`, itself checked against numerical
    integration before being trusted here (see this module's docstring)."""
    p1 = _gamma_cdf(y, shape)
    p2 = _gamma_cdf(y, shape + 1)
    return y * (2 * p1 - 1) - (shape * (2 * p2 - 1) + 1 / beta_fn(0.5, np.clip(shape, _TINY, None)))


# Fixed-node Gauss-Legendre quadrature for the censoring-correction integral
# below -- a bounded integral (over [0, c], c typically O(1-10)), so a
# modest fixed node count is accurate to floating-point precision and,
# unlike `scipy.integrate.quad`, vectorizes across an entire training set
# in one call (needed since this runs inside every CRPS-minimization step).
_QUAD_NODES, _QUAD_WEIGHTS = np.polynomial.legendre.leggauss(48)


def _censoring_correction(c: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """`integral_0^c F_kappa(u)^2 du`, i.e. `E[(c - max(U, U'))+]` for two
    iid unit-scale Gamma(`shape`) variables `U`, `U'` -- the amount by
    which left-censoring at zero reduces CRPS relative to the uncensored
    gamma, derived from the raw CRPS definition (see this module's
    docstring) rather than transcribed from the paper's own (unverified)
    closed form.
    """
    c = np.clip(c, 0.0, None)
    # Map Gauss-Legendre's [-1, 1] nodes onto each point's own [0, c]
    # interval; broadcasts over every (node, data point) pair at once.
    u = (_QUAD_NODES[:, None] + 1) / 2 * c[None, :]
    shape_broadcast = np.broadcast_to(shape[None, :], u.shape)
    f_shape_u = _gamma_cdf(u, shape_broadcast)
    return (c / 2) * np.sum(_QUAD_WEIGHTS[:, None] * f_shape_u**2, axis=0)


def csgd_crps(
    mean: np.ndarray | float,
    std: np.ndarray | float,
    shift: np.ndarray | float,
    y: np.ndarray | float,
) -> np.ndarray | float:
    """CRPS of the censored, shifted gamma distribution (CSGD) against a
    real observation `y` (Scheuerer & Hamill 2015's distribution, section
    3) -- see this module's docstring for why this is a from-scratch,
    numerically-verified derivation rather than a transcription of the
    paper's own closed-form equation.

    `mean`/`std` are the *underlying* (unshifted, uncensored) gamma
    distribution's own mean and standard deviation (kappa = mean^2/std^2,
    theta = std^2/mean); `shift` is delta (<=0), which moves that gamma's
    support to include negative values, all of which then censor to an
    exact zero. `y` must be >= 0.

    Verified in `tests/test_emos.py` against brute-force numerical
    integration of CRPS's raw definition, over a grid of parameters, to
    floating-point precision.
    """
    mean_arr = np.asarray(mean, dtype=float)
    std_arr = np.asarray(std, dtype=float)
    shift_arr = np.asarray(shift, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    broadcast_shape = np.broadcast_shapes(
        mean_arr.shape, std_arr.shape, shift_arr.shape, y_arr.shape
    )
    # Clipped once, consistently, before use in both kappa and theta below
    # -- `mean` is the underlying gamma's own mean, intrinsically positive;
    # a caller passing a stray non-positive value (e.g. a fitted
    # regression extrapolating against real, slightly-negative
    # numerical-noise forecast data) must not see kappa computed from the
    # raw value while theta uses a clipped one, which produced a
    # near-zero-shape, huge-scale gamma and a wildly wrong CRPS when this
    # was caught running Phase 4 step 6 against the real stores.
    mean_b = np.clip(np.broadcast_to(mean_arr, broadcast_shape).astype(float), _TINY, None)
    std_b = np.broadcast_to(std_arr, broadcast_shape).astype(float)
    shift_b = np.broadcast_to(shift_arr, broadcast_shape).astype(float)
    y_b = np.broadcast_to(y_arr, broadcast_shape).astype(float)

    kappa = mean_b**2 / np.clip(std_b**2, _TINY, None)
    theta = np.clip(std_b**2, _TINY, None) / mean_b

    c = -shift_b / theta
    y_tilde = (y_b - shift_b) / theta

    correction = _censoring_correction(c.reshape(-1), kappa.reshape(-1)).reshape(broadcast_shape)

    # `theta` grows unboundedly as `mean` shrinks toward zero (theta =
    # std^2/mean), while `_crps_gamma_std(kappa, y_tilde) - correction`
    # shrinks to compensate -- multiplying a huge theta by the difference
    # of two nearly-equal small floating-point terms can amplify ordinary
    # floating-point error into an O(1) result for a pathologically tiny
    # `mean` (checked directly: still bounded and small, not silently
    # wrong, for any `mean` a real fitted regression's own bounds and
    # nonnegative-precipitation input data actually produce). CRPS is
    # mathematically never negative, so a negative result here is
    # unambiguous floating-point noise, clamped rather than returned.
    result = np.clip(theta * _crps_gamma_std(kappa, y_tilde) - theta * correction, 0.0, None)
    return result if broadcast_shape != () else float(result)


def _fit_climatology(obs: np.ndarray) -> tuple[float, float, float, bool, str | None]:
    """The "unconditional precipitation accumulations" step (Scheuerer &
    Hamill section 4b): a CRPS-minimizing, predictor-free CSGD fit to
    `obs` -- i.e. an intercept-only special case of the same regression
    `_fit_conditional` performs, not a separately invented moment
    estimator. Returns (mu_cl, sigma_cl, delta_cl, is_fallback, reason).
    """
    n = obs.size
    if n == 0:
        return 0.0, _TINY, 0.0, True, "no training observations in this bin"

    p_pop = float(np.mean(obs > 0))
    if p_pop < MIN_PROBABILITY_OF_PRECIPITATION:
        return 0.0, _TINY, 0.0, True, f"probability of precipitation {p_pop:.4f} too low to fit"

    wet = obs[obs > 0]
    # Starting values per Scheuerer & Hamill section 4b: assume kappa=1
    # (exponential) so mean(wet) estimates both mu and sigma, and delta
    # follows from p_pop under that same assumption.
    mu0 = float(np.mean(wet))
    sigma0 = mu0
    delta0 = mu0 * np.log(p_pop)

    def objective(params: np.ndarray) -> float:
        mu, sigma, delta = params
        return float(np.mean(csgd_crps(mu, sigma, delta, obs)))

    result = optimize.minimize(
        objective,
        x0=[mu0, sigma0, delta0],
        method="L-BFGS-B",
        bounds=[(_TINY, None), (_TINY, None), (None, 0.0)],
    )
    mu_cl, sigma_cl, delta_cl = result.x
    return float(mu_cl), float(sigma_cl), float(delta_cl), False, None


def _fit_conditional(
    ensemble_mean: np.ndarray,
    ensemble_spread: np.ndarray,
    obs: np.ndarray,
    shift: float,
    mu_cl: float,
    sigma_cl: float,
) -> dict[str, float]:
    """Fits a1-a4 of `location = a1 + a2*ensemble_mean`, `scale = a3*sqrt(
    max(location, eps)) + a4*ensemble_spread`, minimizing mean CRPS against
    `shift` held fixed -- Scheuerer & Hamill eq. 11-12/15, unnormalized (see
    this module's docstring)."""

    def objective(params: np.ndarray) -> float:
        a1, a2, a3, a4 = params
        location = a1 + a2 * ensemble_mean
        scale = a3 * np.sqrt(np.clip(location, _TINY, None)) + a4 * ensemble_spread
        return float(np.mean(csgd_crps(location, scale, shift, obs)))

    x0 = [max(mu_cl, _TINY), 0.5, max(sigma_cl / np.sqrt(max(mu_cl, _TINY)), _TINY), 0.1]
    result = optimize.minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=[(_TINY, None), (0.0, None), (_TINY, None), (0.0, None)],
    )
    a1, a2, a3, a4 = result.x
    return {"a1": float(a1), "a2": float(a2), "a3": float(a3), "a4": float(a4)}


def fit_emos_csg(
    ensemble_forecast: xr.DataArray,
    obs: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    train_mask: np.ndarray,
    source: str,
    sample_dim: str = "sample",
    member_dim: str = "member",
    min_train_days: int = MIN_TRAIN_DAYS_PER_BIN,
) -> dict[str, CensoredShiftedGammaResult]:
    """Fits one EMOS-CSG regression per rain-intensity bin for one ensemble
    `source` (see this module's docstring for why sources are fit
    independently, never pooled).

    `ensemble_forecast` is `(sample_dim, latitude, longitude, member_dim)`;
    `obs` is the matching IMD observation array; `rain_bin_labels` is
    `weavr.rain_bins.classify_rain_bin`'s `(sample_dim, latitude,
    longitude)` label array (already classified from this same forecast,
    per the caller); `train_mask` is a boolean array over `sample_dim` from
    whatever CV strategy the caller decided
    (`docs/phase4-data-and-combiner-scope.md`: `seasonal_block_split`).

    For each of `weavr.rain_bins.RAIN_BIN_LABELS`, pools every (train
    sample, gridpoint) cell landing in that bin (across every gridpoint, not
    just one) where the forecast and obs are simultaneously non-NaN, and
    fits the climatology-then-conditional CSGD regression described in this
    module's docstring. Falls back to a point-mass-at-zero result, flagged
    via `CensoredShiftedGammaResult.is_fallback`/`reason`, when the bin has
    fewer than `min_train_days` distinct contributing train days, or when
    its climatological fit itself falls back (see `_fit_climatology`).
    """
    ensemble_mean = ensemble_forecast.mean(dim=member_dim, skipna=True)
    ensemble_spread = ensemble_forecast.std(dim=member_dim, ddof=1, skipna=True)

    train_sample_coords = obs[sample_dim].values[train_mask]

    bin_train = rain_bin_labels.sel({sample_dim: train_sample_coords}).values
    obs_train = obs.sel({sample_dim: train_sample_coords}).values
    mean_train = ensemble_mean.sel({sample_dim: train_sample_coords}).values
    spread_train = ensemble_spread.sel({sample_dim: train_sample_coords}).values

    n_time = bin_train.shape[0]
    n_space = bin_train.size // max(n_time, 1)
    valid = ~np.isnan(obs_train) & ~np.isnan(mean_train) & ~np.isnan(spread_train)

    results: dict[str, CensoredShiftedGammaResult] = {}
    for bin_label in RAIN_BIN_LABELS:
        cell_mask = (bin_train == bin_label) & valid

        n_contributing_days = int(cell_mask.reshape(n_time, n_space).any(axis=1).sum())

        if n_contributing_days < min_train_days:
            results[bin_label] = CensoredShiftedGammaResult(
                bin_label=bin_label,
                source=source,
                shift=0.0,
                climatological_mean=0.0,
                climatological_std=_TINY,
                is_fallback=True,
                reason=(
                    f"only {n_contributing_days} train day(s) with usable data in this bin "
                    f"(need >= {min_train_days})"
                ),
                n_train_days=n_contributing_days,
            )
            continue

        obs_cells = obs_train[cell_mask]
        mean_cells = mean_train[cell_mask]
        spread_cells = spread_train[cell_mask]

        mu_cl, sigma_cl, delta_cl, clim_fallback, clim_reason = _fit_climatology(obs_cells)
        if clim_fallback:
            results[bin_label] = CensoredShiftedGammaResult(
                bin_label=bin_label,
                source=source,
                shift=0.0,
                climatological_mean=mu_cl,
                climatological_std=sigma_cl,
                is_fallback=True,
                reason=clim_reason,
                n_train_days=n_contributing_days,
            )
            continue

        coefficients = _fit_conditional(
            mean_cells, spread_cells, obs_cells, delta_cl, mu_cl, sigma_cl
        )
        results[bin_label] = CensoredShiftedGammaResult(
            bin_label=bin_label,
            source=source,
            shift=delta_cl,
            climatological_mean=mu_cl,
            climatological_std=sigma_cl,
            coefficients=coefficients,
            is_fallback=False,
            reason=None,
            n_train_days=n_contributing_days,
        )

    return results


def predict_csgd_params(
    result: CensoredShiftedGammaResult,
    ensemble_mean: np.ndarray,
    ensemble_spread: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a fitted (or fallback) result and a new forecast's per-cell
    ensemble mean/spread, returns the predictive distribution's (mean, std,
    shift) arrays -- a fallback result predicts a degenerate CSGD collapsed
    at zero (mean=0, shift=0, a tiny positive std for numerical validity),
    equivalent to a point mass at zero.
    """
    ensemble_mean = np.asarray(ensemble_mean, dtype=float)
    ensemble_spread = np.asarray(ensemble_spread, dtype=float)

    if result.is_fallback:
        zeros = np.zeros_like(ensemble_mean)
        return zeros, np.full_like(ensemble_mean, _TINY), zeros

    a1 = result.coefficients["a1"]
    a2 = result.coefficients["a2"]
    a3 = result.coefficients["a3"]
    a4 = result.coefficients["a4"]

    # Clipped once, consistently, before either use below -- a fitted
    # regression's own coefficients keep `location` positive for any
    # nonnegative `ensemble_mean` (a1, a2 >= 0 by `_fit_conditional`'s own
    # optimizer bounds), but real forecast data can carry tiny negative
    # numerical-noise artifacts near zero precipitation (checked directly:
    # ~1% of GraphCast's own real cells are slightly negative, a known
    # ML-weather-model artifact, not a bug in this project's own pipeline).
    # An unclipped negative `location` reaching `csgd_crps` produces a
    # near-zero-shape, huge-scale gamma that blows up numerically -- caught
    # empirically running Phase 4 step 6 against the real stores, not
    # assumed safe.
    location = np.clip(a1 + a2 * ensemble_mean, _TINY, None)
    scale = a3 * np.sqrt(location) + a4 * ensemble_spread
    shift = np.full_like(ensemble_mean, result.shift)
    return location, scale, shift


def score_csgd(
    result: CensoredShiftedGammaResult,
    ensemble_mean: xr.DataArray,
    ensemble_spread: xr.DataArray,
    obs: xr.DataArray,
) -> xr.DataArray:
    """CRPS of the fitted (or fallback) predictive CSGD against real `obs`,
    at new forecasts -- the predict/score path `weavr.verify`'s existing
    scoring conventions expect (an `xr.DataArray` of per-cell scores, not
    reduced over any dimension, matching `weavr.verify.crps`'s own
    unreduced-by-default shape).
    """
    mean, std, shift = predict_csgd_params(result, ensemble_mean.values, ensemble_spread.values)
    scores = csgd_crps(mean, std, shift, obs.values)
    return xr.DataArray(scores, dims=obs.dims, coords=obs.coords, name="csgd_crps")
