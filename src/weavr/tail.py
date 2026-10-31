"""Extreme-value tail repair for unfittable rain bins (Step 12).

Anchors extreme rainfall exceedance probabilities on a pooled Generalized
Pareto distribution (GPD) fitted across IMD climatological extremes:
- Exceedances above threshold u share a single shape parameter xi across India,
  with region-specific scale parameters sigma_r (weavr.regions zones).
- Spliced with the nearest fittable EMOS-CSG bin at threshold u:
    P(Y > y) = P_CSGD(Y > u) * (1 + xi * (y - u) / sigma_r)^(-1 / xi)   for y > u.
- For y <= u, P(Y > y) = P_CSGD(Y > y), guaranteeing exact continuity at u.
  `spliced_exceedance_probability` only knows P(Y > u), so it refuses y < u;
  `exceedance_probability_with_tail` evaluates the CSGD itself below u.
- `spliced_twcrps`: threshold-weighted CRPS of the spliced distribution for
  t >= u, in closed form.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy import optimize

from weavr.emos import (
    CensoredShiftedGammaResult,
    exceedance_probability_csgd,
    predict_csgd_params,
)
from weavr.rain_bins import RAIN_BIN_LABELS, classify_rain_bin
from weavr.regions import SREEKALA_BABU_ZONES

DEFAULT_TAIL_THRESHOLD_U = 64.5  # IMD heavy rainfall threshold (mm)
_MIN_EXCEEDANCES_PER_REGION = 10
_EPS = 1e-7


@dataclass(frozen=True)
class TailFit:
    """Fitted pooled GPD parameters across regions."""

    threshold_u: float
    shape_xi: float
    scale_by_region: dict[str, float]
    xi_se: float | None = None
    xi_ci: tuple[float, float] | None = None
    n_samples_by_region: dict[str, int] = field(default_factory=dict)


def _inverse_hessian_diagonal(
    f: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-4
) -> np.ndarray:
    """Diagonal of the inverse Hessian of `f` at `x`, by central differences.

    Returns NaNs when the Hessian is not positive definite -- an optimum on a
    bound, or a step that crosses the GPD support boundary into the penalty
    region -- because inverting it would report a meaningless variance.
    """
    n = x.size
    hessian = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            corners = []
            for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = x.astype(float).copy()
                p[i] += di * step
                p[j] += dj * step
                corners.append(f(p))
            pp, pm, mp, mm = corners
            hessian[i, j] = hessian[j, i] = (pp - pm - mp + mm) / (4.0 * step**2)
    if not np.all(np.isfinite(hessian)) or np.any(np.linalg.eigvalsh(hessian) <= 0):
        return np.full(n, np.nan)
    return np.diag(np.linalg.inv(hessian))


def fit_pooled_gpd(
    exceedances_by_region: dict[str, np.ndarray],
    threshold_u: float = DEFAULT_TAIL_THRESHOLD_U,
    regions: tuple[str, ...] = SREEKALA_BABU_ZONES,
) -> TailFit:
    """Fit a pooled GPD model with one shared shape xi and region-specific scales sigma_r.

    Parameters
    ----------
    exceedances_by_region:
        Mapping from region code (e.g. 'WC', 'CI') to 1D array of rainfall
        observations that exceed threshold_u (or excess values z = y - u).
        If raw observations y are provided, excess values z = y - u are computed.
    threshold_u:
        Threshold u (mm) above which the GPD applies.
    regions:
        Region names to include in the fit.

    Returns
    -------
    TailFit holding the shared shape xi, scales sigma_r per region, and standard error.
    """
    clean_excesses: dict[str, np.ndarray] = {}
    n_samples: dict[str, int] = {}

    for r in regions:
        vals = np.asarray(exceedances_by_region.get(r, []), dtype=float)
        vals = vals[~np.isnan(vals)]
        # If values appear to be raw rainfall (min >= threshold_u), subtract u
        if len(vals) > 0 and np.all(vals >= threshold_u - 1e-5):
            excesses = vals - threshold_u
        else:
            excesses = vals
        excesses = excesses[excesses > 0]
        clean_excesses[r] = excesses
        n_samples[r] = len(excesses)

    active_regions = [r for r in regions if n_samples[r] >= _MIN_EXCEEDANCES_PER_REGION]
    if not active_regions:
        raise ValueError(
            f"No region has >= {_MIN_EXCEEDANCES_PER_REGION} exceedances above {threshold_u} mm."
        )

    # Initial estimates: method of moments on pooled data
    all_excesses = np.concatenate([clean_excesses[r] for r in active_regions])
    mean_excess = float(np.mean(all_excesses))
    var_excess = float(np.var(all_excesses))
    if var_excess > mean_excess**2:
        init_xi = 0.5 * (mean_excess**2 / var_excess - 1.0)
    else:
        init_xi = 0.1
    init_xi = float(np.clip(init_xi, -0.5, 0.8))

    init_log_scales: list[float] = []
    for r in active_regions:
        reg_mean = float(np.mean(clean_excesses[r]))
        scale = max(reg_mean * (1.0 - init_xi), 1.0)
        init_log_scales.append(np.log(scale))

    init_params = np.array([init_xi] + init_log_scales)

    def neg_log_likelihood(params: np.ndarray) -> float:
        xi = params[0]
        log_scales = params[1:]
        total_nll = 0.0

        for r, log_sig in zip(active_regions, log_scales, strict=True):
            sig = np.exp(log_sig)
            z = clean_excesses[r]
            n = len(z)

            if abs(xi) < _EPS:
                # Exponential limit
                total_nll += n * log_sig + np.sum(z) / sig
            else:
                arg = 1.0 + xi * z / sig
                if np.any(arg <= 0):
                    # Violation of support condition
                    penalty = 1e8 + 1e4 * np.sum(np.maximum(0.0, -arg))
                    return penalty
                total_nll += n * log_sig + (1.0 + 1.0 / xi) * np.sum(np.log(arg))

        return float(total_nll)

    res = optimize.minimize(
        neg_log_likelihood,
        init_params,
        method="L-BFGS-B",
        bounds=[(-0.8, 1.5)] + [(np.log(0.1), np.log(500.0))] * len(active_regions),
    )

    opt_xi = float(res.x[0])
    scale_dict: dict[str, float] = {}
    for r, log_sig in zip(active_regions, res.x[1:], strict=True):
        scale_dict[r] = float(np.exp(log_sig))

    # For regions with insufficient samples, borrow pooled mean scale
    avg_scale = float(np.mean(list(scale_dict.values()))) if scale_dict else 10.0
    for r in regions:
        if r not in scale_dict:
            scale_dict[r] = avg_scale

    # Standard error of xi from the full observed information matrix. Only
    # `(H^-1)[0, 0]` accounts for xi being estimated jointly with every regional
    # scale; `1 / H[0, 0]` alone is the variance with the scales held fixed, and
    # because xi and sigma are strongly correlated in a GPD that understates
    # the SE -- by 26% on an i.i.d. sample, where the nominal 95% interval
    # covered the true xi only 85% of the time.
    xi_se: float | None = None
    xi_ci: tuple[float, float] | None = None
    try:
        variance = _inverse_hessian_diagonal(neg_log_likelihood, res.x)[0]
        if np.isfinite(variance) and variance > 0:
            xi_se = float(np.sqrt(variance))
            xi_ci = (opt_xi - 1.96 * xi_se, opt_xi + 1.96 * xi_se)
    except np.linalg.LinAlgError as exc:
        warnings.warn(f"Standard error calculation failed: {exc}", UserWarning, stacklevel=2)

    return TailFit(
        threshold_u=threshold_u,
        shape_xi=opt_xi,
        scale_by_region=scale_dict,
        xi_se=xi_se,
        xi_ci=xi_ci,
        n_samples_by_region=n_samples,
    )


def spliced_exceedance_probability(
    p_exceed_u: float | np.ndarray,
    tail_fit: TailFit,
    region: str | np.ndarray,
    y: float | np.ndarray,
) -> float | np.ndarray:
    """Compute spliced exceedance probability P(Y > y).

    Guarantees:
    - Exactly continuous at y = threshold_u: P(Y > u) = p_exceed_u.
    - Strictly non-increasing (monotone) for y >= u.
    - Always bounded in [0, 1].

    Defined for y >= u only. Below u the spliced distribution is the CSGD
    itself, which this function is not given; it used to return P(Y > u) there,
    a step where the distribution has none, so it now raises instead
    (`exceedance_probability_with_tail` evaluates the CSGD below u).

    Parameters
    ----------
    p_exceed_u:
        Exceedance probability at threshold u, P(Y > u), from EMOS-CSG.
    tail_fit:
        Fitted TailFit model with shape xi and region scales sigma_r.
    region:
        Region code string or array of region strings matching y shape.
    y:
        Rainfall threshold y (mm) to evaluate.
    """
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p_exceed_u, dtype=float)
    is_scalar = (y_arr.ndim == 0) and (p_arr.ndim == 0) and not isinstance(region, np.ndarray)

    broadcast_shape = np.broadcast_shapes(y_arr.shape, p_arr.shape)
    y_b = np.broadcast_to(y_arr, broadcast_shape)
    p_b = np.clip(np.broadcast_to(p_arr, broadcast_shape), 0.0, 1.0)

    u = tail_fit.threshold_u
    xi = tail_fit.shape_xi
    default_scale = np.median(list(tail_fit.scale_by_region.values()))
    if np.any(y_b < u):
        raise ValueError(
            f"spliced_exceedance_probability is defined for y >= u = {u} mm only; "
            "below u the spliced distribution is the CSGD itself"
        )

    if isinstance(region, str):
        sig = float(tail_fit.scale_by_region.get(region, default_scale))
        sig_b = np.full(broadcast_shape, sig, dtype=float)
    else:
        reg_arr = np.broadcast_to(np.asarray(region), broadcast_shape)
        lookup = np.vectorize(
            lambda r: tail_fit.scale_by_region.get(str(r), default_scale),
            otypes=[float],
        )
        sig_b = lookup(reg_arr)

    # For y <= u, probability is p_exceed_u
    z = np.maximum(0.0, y_b - u)

    if abs(xi) < _EPS:
        gpd_surv = np.exp(-z / sig_b)
    else:
        arg = 1.0 + xi * z / sig_b
        # When xi < 0, GPD has finite upper support boundary: u - sig / xi
        gpd_surv = np.where(arg <= 0, 0.0, arg ** (-1.0 / xi))

    result = np.where(y_b <= u, p_b, p_b * gpd_surv)
    result = np.clip(result, 0.0, 1.0)

    return float(result) if is_scalar else result


def nearest_fittable_bin(
    target_bin: str,
    emos_results_by_bin: dict[str, CensoredShiftedGammaResult],
) -> CensoredShiftedGammaResult | None:
    """Find the nearest fittable EMOS-CSG bin relative to `target_bin`."""
    if target_bin in emos_results_by_bin and not emos_results_by_bin[target_bin].is_fallback:
        return emos_results_by_bin[target_bin]

    all_labels = list(RAIN_BIN_LABELS)
    if target_bin not in all_labels:
        # Fallback to any valid fittable bin
        valid = [res for res in emos_results_by_bin.values() if not res.is_fallback]
        return valid[0] if valid else None

    target_idx = all_labels.index(target_bin)
    candidate_indices = sorted(
        range(len(all_labels)),
        key=lambda idx: (abs(idx - target_idx), idx > target_idx),
    )

    for idx in candidate_indices:
        b_name = all_labels[idx]
        if b_name in emos_results_by_bin and not emos_results_by_bin[b_name].is_fallback:
            return emos_results_by_bin[b_name]

    return None


def _select_region(
    region: str | np.ndarray, mask: np.ndarray, shape: tuple[int, ...]
) -> str | np.ndarray:
    """Pull the region labels belonging to `mask`, broadcasting a spatial grid.

    `region` is normally a `(latitude, longitude)` grid while `mask` comes from
    `f_arr`, which for a time series is `(sample, latitude, longitude)` -- two
    orders of magnitude more entries. Masking the grid directly with that mask
    raised `IndexError` on the first real multi-sample call, so the grid is
    broadcast over sample first. A caller-supplied region that already matches
    `mask` is passed through untouched.
    """
    if not isinstance(region, np.ndarray):
        return region
    reg = region if region.shape == shape else np.broadcast_to(region, shape)
    return reg[mask]


def exceedance_probability_with_tail(
    forecast_values: xr.DataArray | np.ndarray,
    ensemble_mean: xr.DataArray | np.ndarray,
    ensemble_spread: xr.DataArray | np.ndarray,
    emos_results_by_bin: dict[str, CensoredShiftedGammaResult],
    tail_fit: TailFit,
    region: str | np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exceedance probability with extreme-value tail for unfittable bins.

    Returns:
    --------
    probabilities: np.ndarray in [0, 1]
    methods: np.ndarray of strings ("csgd" | "csgd+gpd_tail" | "fallback")
    """
    f_arr = (
        forecast_values.values
        if isinstance(forecast_values, xr.DataArray)
        else np.asarray(forecast_values)
    )
    m_arr = (
        ensemble_mean.values
        if isinstance(ensemble_mean, xr.DataArray)
        else np.asarray(ensemble_mean)
    )
    s_arr = (
        ensemble_spread.values
        if isinstance(ensemble_spread, xr.DataArray)
        else np.asarray(ensemble_spread)
    )

    # Classify rain bin
    f_da = xr.DataArray(f_arr) if not isinstance(forecast_values, xr.DataArray) else forecast_values
    bins_da = classify_rain_bin(f_da)
    bin_names = bins_da.values

    probs = np.zeros_like(f_arr, dtype=float)
    methods = np.full(f_arr.shape, "fallback", dtype=object)

    u = tail_fit.threshold_u

    for bin_name in RAIN_BIN_LABELS:
        mask = (bin_names == bin_name) & (~np.isnan(f_arr))
        if not np.any(mask):
            continue

        res = emos_results_by_bin.get(bin_name)
        is_fittable = res is not None and not res.is_fallback

        if threshold < u and not is_fittable:
            # Below u the spliced distribution of an unfittable bin *is* the
            # nearest fittable bin's CSGD; the GPD only describes the excess
            # above u. (This branch used to fall through to the splice, which
            # returned P(Y > u) for any threshold below u.)
            nearest_res = nearest_fittable_bin(bin_name, emos_results_by_bin)
            if nearest_res is None or nearest_res.is_fallback:
                probs[mask] = 0.0
                methods[mask] = "fallback"
            else:
                pred_m, pred_s, pred_shift = predict_csgd_params(
                    nearest_res, m_arr[mask], s_arr[mask]
                )
                probs[mask] = np.asarray(
                    exceedance_probability_csgd(pred_m, pred_s, pred_shift, threshold), dtype=float
                )
                methods[mask] = "csgd+gpd_tail"
        elif is_fittable and threshold <= u and res is not None:
            # Within useful fittable range: use CSGD directly
            mean_b = m_arr[mask]
            std_b = s_arr[mask]
            pred_m, pred_s, pred_shift = predict_csgd_params(res, mean_b, std_b)
            p_val = exceedance_probability_csgd(pred_m, pred_s, pred_shift, threshold)
            probs[mask] = np.asarray(p_val, dtype=float)
            methods[mask] = "csgd"
        else:
            # Unfittable bin or extreme threshold > u: use nearest fittable bin + GPD tail
            nearest_res = nearest_fittable_bin(bin_name, emos_results_by_bin)
            if nearest_res is None or nearest_res.is_fallback:
                probs[mask] = 0.0
                methods[mask] = "fallback"
            else:
                mean_b = m_arr[mask]
                std_b = s_arr[mask]
                pred_m, pred_s, pred_shift = predict_csgd_params(nearest_res, mean_b, std_b)
                p_u = exceedance_probability_csgd(pred_m, pred_s, pred_shift, u)
                reg_sub = _select_region(region, mask, f_arr.shape)
                p_tail = spliced_exceedance_probability(
                    p_exceed_u=np.asarray(p_u, dtype=float),
                    tail_fit=tail_fit,
                    region=reg_sub,
                    y=threshold,
                )
                probs[mask] = np.asarray(p_tail, dtype=float)
                methods[mask] = "csgd+gpd_tail"

    return probs, methods


def _gpd_survival_integrals(
    c: np.ndarray, xi: float, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`int_c^inf S(z) dz` and `int_c^inf S(z)^2 dz` for the GPD survival S.

    With `S(z) = (1 + xi z / sigma)^(-1/xi)`: the first is
    `sigma / (1 - xi) * S(c)^(1 - xi)` and the second
    `sigma / (2 - xi) * S(c)^(2 - xi)` (exponential limits at xi = 0). For
    xi < 0 S is 0 beyond the support end, and so are both integrals. The first
    is finite only for xi < 1 -- a tail whose mean does not exist has no CRPS.
    """
    if xi >= 1.0:
        raise ValueError(f"tw-CRPS needs a finite tail mean (xi < 1), got xi = {xi}")
    if abs(xi) < _EPS:
        s = np.exp(-c / sigma)
    else:
        arg = 1.0 + xi * c / sigma
        s = np.where(arg <= 0, 0.0, np.maximum(arg, 0.0) ** (-1.0 / xi))
    return sigma / (1.0 - xi) * s ** (1.0 - xi), sigma / (2.0 - xi) * s ** (2.0 - xi)


def spliced_twcrps(
    p_exceed_u: float | np.ndarray,
    tail_fit: TailFit,
    region: str | np.ndarray,
    y: float | np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Threshold-weighted CRPS, weight `1{z >= threshold}`, of the spliced
    distribution, for `threshold >= u`.

    Above u the spliced CDF is `F(x) = 1 - p S(x - u)` with `p = P(Y > u)` and
    S the region's GPD survival, so only p and the GPD enter. With
    `a = threshold - u` and `b = max(y, threshold) - u`,

        twCRPS = int_a^b (1 - p S)^2 dz + int_b^inf (p S)^2 dz
               = (b - a) - 2 p (I1(a) - I1(b)) + p^2 I2(a),

    where `I1(c) = int_c^inf S` and `I2(c) = int_c^inf S^2` are closed form
    (`_gpd_survival_integrals`). A cell with p = 0 (a true fallback) is a
    point mass below the threshold and scores `b - a`.

    Step 12 asks for this "by numerical integration, tested"; the closed form
    is exact and is what `tests/test_tail.py` checks against quadrature of the
    weighted definition and against Monte Carlo draws of the spliced
    distribution.
    """
    u = tail_fit.threshold_u
    if threshold < u:
        raise ValueError(
            f"spliced_twcrps needs threshold >= u = {u} mm: below u the distribution "
            "is the CSGD's, and csgd_twcrps scores it"
        )
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p_exceed_u, dtype=float), 0.0, 1.0)
    shape = np.broadcast_shapes(y_arr.shape, p_arr.shape)
    default_scale = np.median(list(tail_fit.scale_by_region.values()))
    if isinstance(region, str):
        sigma = np.full(shape, float(tail_fit.scale_by_region.get(region, default_scale)))
    else:
        lookup = np.vectorize(
            lambda r: tail_fit.scale_by_region.get(str(r), default_scale), otypes=[float]
        )
        sigma = lookup(np.broadcast_to(np.asarray(region), shape))
    a = np.full(shape, threshold - u, dtype=float)
    b = np.maximum(np.broadcast_to(y_arr, shape), threshold) - u
    i1_a, i2_a = _gpd_survival_integrals(a, tail_fit.shape_xi, sigma)
    i1_b, _ = _gpd_survival_integrals(b, tail_fit.shape_xi, sigma)
    p = np.broadcast_to(p_arr, shape)
    return (b - a) - 2.0 * p * (i1_a - i1_b) + p**2 * i2_a
