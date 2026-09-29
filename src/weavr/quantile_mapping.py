"""Quantile mapping for rainfall forecast tail repair -- Step 11.

AI weather models (GraphCast, AIFS, GenCast) systematically smooth heavy rain away
over India (research brief §3.4; F7 shows POD at >=115.6 mm drops to ~0 beyond day 1).

Quantile mapping (QM) maps each source's forecast cumulative distribution function (CDF)
onto IMD's observed distribution per region and lead before blending or EMOS:
x_corr = F_obs^{-1}(F_fcst(x))

Key considerations:
1. Drizzle inflation prevention: maps the forecast dry fraction to the observed dry fraction.
   Forecast values below the quantile corresponding to the observed dry fraction are set to 0.
2. Upper tail extrapolation beyond the maximum fitted quantile: uses a constant additive
   anomaly correction x_corr = x + (q_obs_max - q_fcst_max) to preserve extremes without
   unstable multiplicative inflation.
3. Strict non-negativity: clipped at 0.0.
4. Monotonicity: rank preservation guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr

ExtrapolationRule = Literal["constant_additive", "ratio"]


@dataclass
class QuantileMap:
    """Fitted quantile mapping transfer function."""

    quantiles: np.ndarray
    forecast_quantiles: np.ndarray
    obs_quantiles: np.ndarray
    forecast_dry_fraction: float
    obs_dry_fraction: float
    wet_threshold: float = 0.1
    extrapolation_rule: ExtrapolationRule = "constant_additive"
    source: str | None = None
    region: str | None = None
    lead: int | None = None
    metadata: dict[str, float | str] = field(default_factory=dict)


def fit_quantile_map(
    forecast_train: np.ndarray | xr.DataArray,
    obs_reference: np.ndarray | xr.DataArray,
    quantiles: np.ndarray | list[float] | None = None,
    wet_threshold: float = 0.1,
    extrapolation_rule: ExtrapolationRule = "constant_additive",
    source: str | None = None,
    region: str | None = None,
    lead: int | None = None,
) -> QuantileMap:
    """Fit an empirical quantile map from training forecasts to reference observations.

    Parameters
    ----------
    forecast_train : array_like
        Training forecast values (mm).
    obs_reference : array_like
        Reference observations (mm) -- from matching training period or climatology.
    quantiles : array_like, optional
        Quantile evaluation levels in [0, 1]. Defaults to 101 points from 0 to 1.
    wet_threshold : float, default 0.1
        Threshold (mm) below which precipitation is considered dry.
    extrapolation_rule : {'constant_additive', 'ratio'}, default 'constant_additive'
        Rule applied for forecast values exceeding the maximum fitted quantile.
    """
    f_vals = np.asarray(forecast_train, dtype=float).ravel()
    o_vals = np.asarray(obs_reference, dtype=float).ravel()

    # Drop NaNs
    f_vals = f_vals[np.isfinite(f_vals)]
    o_vals = o_vals[np.isfinite(o_vals)]

    if len(f_vals) == 0 or len(o_vals) == 0:
        # Fallback identity map
        q_dummy = np.linspace(0.0, 1.0, 101)
        return QuantileMap(
            quantiles=q_dummy,
            forecast_quantiles=q_dummy,
            obs_quantiles=q_dummy,
            forecast_dry_fraction=0.0,
            obs_dry_fraction=0.0,
            wet_threshold=wet_threshold,
            extrapolation_rule=extrapolation_rule,
            source=source,
            region=region,
            lead=lead,
        )

    # Compute dry fraction
    f_dry = float(np.mean(f_vals < wet_threshold))
    o_dry = float(np.mean(o_vals < wet_threshold))

    if quantiles is None:
        # Use high resolution quantiles with extra density in the upper tail
        q_lower = np.linspace(0.0, 0.90, 50)
        q_upper = np.linspace(0.91, 1.0, 51)
        q_levels = np.unique(np.concatenate([q_lower, q_upper]))
    else:
        q_levels = np.sort(np.asarray(quantiles, dtype=float))

    f_q = np.quantile(f_vals, q_levels)
    o_q = np.quantile(o_vals, q_levels)

    # Ensure monotonicity
    f_q = np.maximum.accumulate(f_q)
    o_q = np.maximum.accumulate(o_q)

    return QuantileMap(
        quantiles=q_levels,
        forecast_quantiles=f_q,
        obs_quantiles=o_q,
        forecast_dry_fraction=f_dry,
        obs_dry_fraction=o_dry,
        wet_threshold=wet_threshold,
        extrapolation_rule=extrapolation_rule,
        source=source,
        region=region,
        lead=lead,
    )


def apply_quantile_map(
    qm: QuantileMap,
    forecast: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """Apply a fitted quantile map to a forecast array or xarray DataArray.

    Corrects intensity distribution while preserving rank order and preventing
    drizzle inflation. Strictly non-negative.
    """
    is_scalar = np.isscalar(forecast) or (isinstance(forecast, np.ndarray) and forecast.ndim == 0)
    is_xarray = isinstance(forecast, xr.DataArray)
    fcst_arr = forecast.values if is_xarray else np.asarray(forecast, dtype=float)
    orig_shape = fcst_arr.shape

    flat = fcst_arr.ravel()
    corrected = np.empty_like(flat)

    # Threshold below which forecast is mapped to zero to match observed dry fraction
    if qm.obs_dry_fraction > 0.0:
        drizzle_cutoff = float(np.interp(qm.obs_dry_fraction, qm.quantiles, qm.forecast_quantiles))
    else:
        drizzle_cutoff = qm.wet_threshold

    # Piecewise interpolation for values within the empirical range
    f_q_max = qm.forecast_quantiles[-1]
    o_q_max = qm.obs_quantiles[-1]

    # Standard interpolation
    unique_f_q, indices = np.unique(qm.forecast_quantiles, return_index=True)
    unique_o_q = qm.obs_quantiles[indices]

    if len(unique_f_q) >= 2:
        interp_vals = np.interp(flat, unique_f_q, unique_o_q)
    elif len(unique_f_q) == 1:
        # Constant forecast maps directly to reference observation quantile
        offset = unique_o_q[0] - unique_f_q[0]
        interp_vals = flat + offset
    else:
        interp_vals = flat

    # Handle upper extrapolation beyond maximum training forecast quantile
    above_max = flat > f_q_max
    if np.any(above_max):
        if qm.extrapolation_rule == "ratio" and f_q_max > 0.0:
            ratio = o_q_max / f_q_max
            interp_vals[above_max] = flat[above_max] * ratio
        else:
            # Constant additive correction
            interp_vals[above_max] = flat[above_max] + (o_q_max - f_q_max)

    # Handle lower boundary (below dry cutoff)
    below_cutoff = flat <= drizzle_cutoff
    interp_vals[below_cutoff] = 0.0

    # Ensure strictly non-negative
    corrected = np.clip(interp_vals, 0.0, None)
    corrected_reshaped = corrected.reshape(orig_shape)

    if is_scalar:
        return float(corrected.item()) if corrected.size == 1 else corrected_reshaped
    if is_xarray:
        return xr.DataArray(
            corrected_reshaped,
            dims=forecast.dims,
            coords=forecast.coords,
            attrs=forecast.attrs,
            name=forecast.name,
        )
    return corrected_reshaped


def fit_regional_quantile_maps(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    regions: xr.DataArray,
    sample_dim: str = "sample",
    member_dim: str | None = None,
    wet_threshold: float = 0.1,
    extrapolation_rule: ExtrapolationRule = "constant_additive",
    source: str | None = None,
    lead: int | None = None,
) -> dict[str, QuantileMap]:
    """Fit per-region quantile maps across an entire geographic domain.

    Parameters
    ----------
    forecast : xr.DataArray
        Forecast grid (sample, [member], latitude, longitude).
    obs : xr.DataArray
        Observations grid (sample, latitude, longitude).
    regions : xr.DataArray
        Region labels grid (latitude, longitude).
    """
    unique_regions = np.unique(regions.values)
    maps: dict[str, QuantileMap] = {}

    for r in unique_regions:
        reg_str = str(r)
        mask = regions.values == r

        # Select cells in this region
        # obs: (sample, lat, lon) -> cells in region
        # Broadcast mask over sample dimension
        obs_reg = obs.values[..., mask]

        if member_dim and member_dim in forecast.dims:
            # Pool all ensemble members for empirical distribution fitting
            fcst_reg = forecast.values[..., mask]
        else:
            fcst_reg = forecast.values[..., mask]

        qm = fit_quantile_map(
            fcst_reg,
            obs_reg,
            wet_threshold=wet_threshold,
            extrapolation_rule=extrapolation_rule,
            source=source,
            region=reg_str,
            lead=lead,
        )
        maps[reg_str] = qm

    return maps


def apply_regional_quantile_maps(
    regional_maps: dict[str, QuantileMap],
    forecast: xr.DataArray,
    regions: xr.DataArray,
) -> xr.DataArray:
    """Apply region-specific quantile maps across a domain."""
    out_vals = np.empty_like(forecast.values, dtype=float)

    for reg_str, qm in regional_maps.items():
        mask = regions.values == reg_str
        if not np.any(mask):
            continue

        if forecast.ndim == 3:  # (sample, lat, lon)
            sub_fcst = forecast.values[:, mask]
            out_vals[:, mask] = apply_quantile_map(qm, sub_fcst)
        elif forecast.ndim == 4:  # (sample, member, lat, lon) or (member, sample, lat, lon)
            # Find which axes correspond to lat, lon (last two)
            sub_fcst = forecast.values[..., mask]
            out_vals[..., mask] = apply_quantile_map(qm, sub_fcst)
        else:
            out_vals[..., mask] = apply_quantile_map(qm, forecast.values[..., mask])

    return xr.DataArray(
        out_vals,
        dims=forecast.dims,
        coords=forecast.coords,
        attrs=forecast.attrs,
        name=forecast.name,
    )
