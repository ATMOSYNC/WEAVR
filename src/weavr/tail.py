"""Extreme-value tail repair for unfittable rain bins (Step 12).

Anchors extreme rainfall exceedance probabilities on a pooled Generalized
Pareto distribution (GPD) fitted across IMD climatological extremes:
- Exceedances above threshold u share a single shape parameter xi across India,
  with region-specific scale parameters sigma_r (weavr.regions zones).
- Spliced with the nearest fittable EMOS-CSG bin at threshold u:
    P(Y > y) = P_CSGD(Y > u) * (1 + xi * (y - u) / sigma_r)^(-1 / xi)   for y > u.
- For y <= u, P(Y > y) = P_CSGD(Y > y), guaranteeing exact continuity at u.
"""

from __future__ import annotations

import warnings
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

    # Compute standard error of xi using numerical Hessian
    xi_se: float | None = None
    xi_ci: tuple[float, float] | None = None
    try:
        eps = 1e-4
        f0 = neg_log_likelihood(res.x)
        p_plus = res.x.copy()
        p_plus[0] += eps
        f_plus = neg_log_likelihood(p_plus)
        p_minus = res.x.copy()
        p_minus[0] -= eps
        f_minus = neg_log_likelihood(p_minus)
        d2 = (f_plus - 2.0 * f0 + f_minus) / (eps**2)
        if d2 > 0:
            xi_se = float(1.0 / np.sqrt(d2))
            xi_ci = (opt_xi - 1.96 * xi_se, opt_xi + 1.96 * xi_se)
    except Exception as exc:
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

        if is_fittable and threshold <= u and res is not None:
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
                reg_sub = region[mask] if isinstance(region, np.ndarray) else region
                p_tail = spliced_exceedance_probability(
                    p_exceed_u=np.asarray(p_u, dtype=float),
                    tail_fit=tail_fit,
                    region=reg_sub,
                    y=threshold,
                )
                probs[mask] = np.asarray(p_tail, dtype=float)
                methods[mask] = "csgd+gpd_tail"

    return probs, methods
