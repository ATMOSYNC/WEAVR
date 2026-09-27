"""The Phase 1 verification protocol -- implemented before any model is trained.

Every metric named in the Phase 1 issue (see docs/phase-plan.md): RMSE/bias/
ACC, CRPS/Brier, SEEPS, FSS, and POD/FAR/CSI/ETS at IMD's own rain
thresholds. Nothing in Phase 2 onward should be judged "better" than the
Tier 0 baseline without going through these exact functions -- a metric
subtly wrong (a sign flip, wrong axis reduced) would invalidate every
downstream comparison, so getting it right once here matters more than
getting it fast.

RMSE, bias (mean error), ACC (correlation) and the categorical contingency
scores (POD/FAR/CSI/ETS) are computed via `xskillscore`, not hand-rolled --
it's a maintained, citeable, xarray-native implementation of exactly these
well-known scores, and reimplementing them from scratch would only
reintroduce the risk this module exists to remove. SEEPS and FSS have no
off-the-shelf xarray implementation available, so they're implemented here
directly, each documented with the formula/derivation used so a reviewer
can check it against the cited source.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr
import xskillscore as xs
from scipy.ndimage import uniform_filter

Dim = str | Sequence[str] | None

# IMD's own daily rainfall categories (mm/day), used throughout weavr for
# every categorical/threshold-based score -- see docs/grid-and-time-convention.md.
IMD_RAIN_THRESHOLDS_MM = (7.5, 64.5, 115.6, 204.5)


# --- Deterministic scores ----------------------------------------------------


def rmse(forecast: xr.DataArray, obs: xr.DataArray, dim: Dim = None) -> xr.DataArray:
    """Root-mean-squared error. 0 for a perfect forecast, no upper bound."""
    return xs.rmse(forecast, obs, dim=dim, skipna=True)


def bias(forecast: xr.DataArray, obs: xr.DataArray, dim: Dim = None) -> xr.DataArray:
    """Mean error, forecast minus obs. Positive means the forecast over-predicts.

    `xskillscore.me(a, b)` computes `mean(a - b)`; called here as
    `me(forecast, obs)` so the sign matches the stated convention -- verified
    directly (`me(a, b)` on `a=[1,2,3]`, `b=[3,3,3]` gives -1, i.e. `mean(a-b)`,
    not `mean(b-a)`), not assumed from the argument names alone.
    """
    return xs.me(forecast, obs, dim=dim, skipna=True)


def acc(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    climatology: xr.DataArray,
    dim: Dim = None,
) -> xr.DataArray:
    """Anomaly correlation coefficient: Pearson correlation of forecast and obs anomalies.

    Anomaly = value - climatology (per the WMO standard ACC definition).
    1.0 for a perfect forecast, 0 for no skill relative to climatology,
    negative for anti-correlated.

    `climatology` must already be broadcastable against `forecast`/`obs`
    (e.g. a per-gridpoint, per-day-of-year climatological mean) -- this
    function does not compute one. See docs/phase-1-data-requirements.md
    for why a native-resolution temperature climatology isn't available yet
    in this project; ACC for temperature is deferred until one exists.
    """
    forecast_anomaly = forecast - climatology
    obs_anomaly = obs - climatology
    return xs.pearson_r(forecast_anomaly, obs_anomaly, dim=dim, skipna=True)


# --- Probabilistic scores -----------------------------------------------------


def crps(
    ensemble_forecast: xr.DataArray,
    obs: xr.DataArray,
    dim: Dim = None,
    member_dim: str = "member",
) -> xr.DataArray:
    """Continuous Ranked Probability Score against an ensemble forecast.

    0 for a perfect deterministic forecast; lower is better. Delegates to
    `xskillscore.crps_ensemble`, which wraps `properscoring.crps_ensemble`.
    `ensemble_forecast` must carry a `member_dim` dimension (default
    "member") -- there is no ensemble source with this shape in the current
    baseline store (see docs/phase-1-data-requirements.md: CRPS/Brier are
    deferred to Phase 2's lagged-ensemble AI forecasts), so this function is
    tested against synthetic ensembles only for now.
    """
    return xs.crps_ensemble(obs, ensemble_forecast, member_dim=member_dim, dim=dim)


def brier_score(
    probability_forecast: xr.DataArray,
    obs_binary: xr.DataArray,
    dim: Dim = None,
) -> xr.DataArray:
    """Brier score for a binary event. 0 for a perfect probabilistic forecast.

    `probability_forecast` is the forecast probability of the event (values
    in [0, 1], no member dimension); `obs_binary` is 1 where the event
    occurred, 0 otherwise. Delegates to `xskillscore.brier_score`.
    """
    return xs.brier_score(obs_binary, probability_forecast, dim=dim)


# --- Categorical scores -------------------------------------------------------


def seeps(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    climatology: xr.DataArray,
    dim: Dim = None,
    climatology_dim: str = "time",
    dry_threshold_mm: float = 1.0,
) -> xr.DataArray:
    """Stable Equitable Error in Probability Space (Rodwell et al. 2010).

    Categorizes both forecast and obs into dry / light / heavy per IMD
    gridpoint, using climatological terciles computed from `climatology`
    (a multi-year historical record along `climatology_dim`, e.g. the
    archive built by scripts/build_seeps_climatology.py): "dry" is below
    `dry_threshold_mm`; among the remaining ("wet") days, the bottom 2/3 are
    "light" and the top 1/3 are "heavy" -- i.e. thresholds are local and
    climatological, not fixed global values, per the whole point of SEEPS.

    Scores each (forecast_category, obs_category) pair via the zero-diagonal
    3x3 matrix S used by Rodwell et al. (2010) / ECMWF:

        S(o=dry,  f=dry)=0        S(o=dry,  f=light)=1/(1-p1)   S(o=dry,  f=heavy)=1/(1-p1)+1/p3
        S(o=light,f=dry)=1/p1     S(o=light,f=light)=0          S(o=light,f=heavy)=1/p3
        S(o=heavy,f=dry)=1/p1+1/(1-p3)  S(o=heavy,f=light)=1/(1-p3)   S(o=heavy,f=heavy)=0

    where p1 is the local climatological dry-day fraction and p3 the
    local heavy-day fraction. This is the "equitable" score: a forecast
    drawn at random from the climatological category frequencies has the
    same expected score (=2) regardless of the true category -- verified
    directly by `tests/test_verify.py::TestSeeps::test_matrix_is_equitable`,
    which checks that property numerically for arbitrary p1/p3 rather than
    only trusting a remembered formula. The two-category jumps (dry<->heavy)
    are the sum of the two adjacent one-category jumps (dry<->light plus
    light<->heavy), matching the additive structure in the cited matrix.

    Returns the raw mean score (0 = perfect, higher = worse) -- not the
    `1 - SEEPS/2` skill-score variant sometimes quoted in the literature.

    Returns NaN wherever the local climatology is degenerate (always dry,
    never dry, or no distinguishable heavy category) rather than dividing
    by zero -- a real possibility at gridpoints with very little rain.
    """
    is_dry_clim = climatology < dry_threshold_mm
    p1 = is_dry_clim.mean(dim=climatology_dim)
    wet_clim = climatology.where(~is_dry_clim)
    q_light_heavy = wet_clim.quantile(2.0 / 3.0, dim=climatology_dim, skipna=True)
    if "quantile" in q_light_heavy.coords:
        q_light_heavy = q_light_heavy.drop_vars("quantile")
    p3 = (1 - p1) / 3

    valid = (p1 > 0) & (p1 < 1) & (p3 > 0)

    def _categorize(da: xr.DataArray) -> xr.DataArray:
        cat = xr.where(da < dry_threshold_mm, 0, xr.where(da < q_light_heavy, 1, 2))
        return cat

    f_cat = _categorize(forecast)
    o_cat = _categorize(obs)

    a = 1 / (1 - p1)  # S(dry, light)
    b = 1 / p1  # S(light, dry)
    d = 1 / p3  # S(light, heavy)
    e = 1 / (1 - p3)  # S(heavy, light)
    f = a + d  # S(dry, heavy)
    g = b + e  # S(heavy, dry)

    score = xr.zeros_like(forecast, dtype=float)
    score = xr.where((o_cat == 0) & (f_cat == 1), a, score)
    score = xr.where((o_cat == 0) & (f_cat == 2), f, score)
    score = xr.where((o_cat == 1) & (f_cat == 0), b, score)
    score = xr.where((o_cat == 1) & (f_cat == 2), d, score)
    score = xr.where((o_cat == 2) & (f_cat == 0), g, score)
    score = xr.where((o_cat == 2) & (f_cat == 1), e, score)
    score = score.where(valid)

    return score.mean(dim=dim, skipna=True) if dim is not None else score.mean(skipna=True)


def fss(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    neighborhood_size: int,
    dim: Dim = None,
    spatial_dims: tuple[str, str] = ("latitude", "longitude"),
) -> xr.DataArray:
    """Fractions Skill Score (Roberts & Lean 2008) at one threshold/neighborhood size.

    1.0 for a perfect forecast, 0 for no better than a forecast of the
    climatological event frequency everywhere. Computed as:

        FSS = 1 - mean((O_frac - F_frac)^2) / mean(O_frac^2 + F_frac^2)

    where O_frac/F_frac are the fraction of `neighborhood_size x
    neighborhood_size` grid cells exceeding `threshold`, computed with a
    uniform (box) filter over `spatial_dims` independently at each other
    coordinate (e.g. each time/lead step) via `scipy.ndimage.uniform_filter`.

    Returns NaN (rather than 0/0) wherever both fraction fields are exactly
    zero everywhere -- no exceedance event exists to score at that
    threshold/neighborhood/sample, so a "skill" number would be meaningless,
    not perfect.
    """
    forecast_exceeds = (forecast >= threshold).astype(float)
    obs_exceeds = (obs >= threshold).astype(float)

    def _neighborhood_fraction(arr: np.ndarray) -> np.ndarray:
        return uniform_filter(arr, size=neighborhood_size, mode="constant", cval=0.0)

    core_dims = list(spatial_dims)
    forecast_frac = xr.apply_ufunc(
        _neighborhood_fraction,
        forecast_exceeds,
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )
    obs_frac = xr.apply_ufunc(
        _neighborhood_fraction,
        obs_exceeds,
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )

    mse = ((forecast_frac - obs_frac) ** 2).mean(dim=dim, skipna=True)
    reference = (forecast_frac**2 + obs_frac**2).mean(dim=dim, skipna=True)
    return xr.where(reference > 0, 1 - mse / reference, np.nan)


def contingency_scores(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    thresholds: Sequence[float] = IMD_RAIN_THRESHOLDS_MM,
    dim: Dim = None,
) -> dict[float, dict[str, xr.DataArray]]:
    """POD, FAR, CSI, ETS at each of `thresholds` (default: IMD's own rain categories).

    For each threshold, forecast/obs are split into a dichotomous
    yes/no-event contingency table (event = value >= threshold) via
    `xskillscore.Contingency`, and scored:

    - POD (probability of detection) = hits / (hits + misses)
    - FAR (false alarm ratio) = false_alarms / (hits + false_alarms)
    - CSI (critical success index / threat score) = hits / (hits + misses + false_alarms)
    - ETS (equitable threat score) = corrected for hits expected by chance

    Returns NaN for any score whose denominator is exactly zero (e.g. FAR
    when the forecast never predicts the event) rather than raising or
    silently returning 0 -- confirmed this is `xskillscore`'s own behavior
    for `Contingency`, not assumed.
    """
    results: dict[float, dict[str, xr.DataArray]] = {}
    for threshold in thresholds:
        edges = np.array([-np.inf, threshold, np.inf])
        table = xs.Contingency(obs, forecast, edges, edges, dim=dim)
        results[threshold] = {
            "pod": table.hit_rate(),
            "far": table.false_alarm_ratio(),
            "csi": table.threat_score(),
            "ets": table.equit_threat_score(),
        }
    return results
