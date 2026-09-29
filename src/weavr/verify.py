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


def crps_large_ensemble(
    ensemble_forecast: xr.DataArray,
    obs: xr.DataArray,
    dim: Dim = None,
    member_dim: str = "member",
) -> xr.DataArray:
    """CRPS computed via the sorted-ensemble identity: same answer as `crps`,
    but `O(m log m)` time and `O(m)` memory instead of `O(m^2)`.

    `crps` delegates to `properscoring`, which forms the full `m x m` matrix
    of pairwise member differences. That is fine for the ensembles this
    project started with (a 50-member IFS ensemble, a handful of lagged AI
    members) and impossible for a large one: the 434-member climatological
    reference in `weavr.climatology`, over the 129x135 India grid, needs
    roughly **83 GB** by that route. Measured, not estimated -- 200 members
    over 2,000 cells already peaks at 2.0 GB, and the first attempt to score
    climatology with `crps` was killed by the OS.

    This uses the standard decomposition

        CRPS = (1/m) * sum_i |x_i - y|  -  (1/(2 m^2)) * sum_i sum_j |x_i - x_j|

    (Hersbach 2000) with the second term evaluated from the *sorted*
    ensemble, where the double sum collapses to a single pass:

        sum_i sum_j |x_i - x_j| = 2 * sum_i (2i - m + 1) * x_(i),   i = 0..m-1

    so no pairwise matrix is ever formed. This is an exact algebraic
    rearrangement, not an approximation:
    `tests/test_verify_significance_additions.py` checks it against `crps`
    itself to floating-point precision across ensemble sizes.

    NaN members are **not** supported here -- they would sort to the end and
    silently corrupt the rank weights. Use `crps` for ragged ensembles, or
    drop the NaN members first. `weavr.climatology` guarantees a constant
    member count for exactly this reason.
    """
    n_members = ensemble_forecast.sizes[member_dim]
    rank_weights = 2.0 * np.arange(n_members) - n_members + 1.0

    def _spread(values: np.ndarray) -> np.ndarray:
        # `values` has the member axis last, per apply_ufunc's core-dim
        # contract. Sort along it and contract against the rank weights.
        return np.sort(values, axis=-1) @ rank_weights

    spread_term = (
        xr.apply_ufunc(
            _spread,
            ensemble_forecast,
            input_core_dims=[[member_dim]],
            dask="parallelized",
            output_dtypes=[float],
        )
        / n_members**2
    )
    mean_absolute_error = abs(ensemble_forecast - obs).mean(dim=member_dim, skipna=True)
    per_cell = mean_absolute_error - spread_term

    # `dim=None` reduces over every remaining dimension, matching
    # xskillscore's own convention (and therefore `crps`'s) rather than
    # returning the per-cell field. Checked against `crps` directly in the
    # tests; the two differ by ~1e-15, not by the spread of the field, which
    # is what a mismatched convention here would look like.
    reduce_over = [d for d in per_cell.dims] if dim is None else dim
    return per_cell.mean(dim=reduce_over, skipna=True)


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


def ensemble_spread(
    ensemble_forecast: xr.DataArray,
    dim: Dim = None,
    member_dim: str = "member",
) -> xr.DataArray:
    """Ensemble spread: the square root of the mean across-member variance.

    Uses the unbiased (`ddof=1`) sample variance across members -- the
    convention `calibrated_spread_skill_ratio`'s finite-ensemble-size
    correction assumes (Fortin et al. 2014; Leutbecher & Palmer 2008), not
    the population variance `ddof=0` would give. NaN members (a short
    lagged window, see `src/weavr/ensemble.py`) are excluded via
    `skipna=True`, not treated as zero-spread.

    Averages *variance* across `dim` before taking the square root (not the
    other way around) -- `E[spread^2]` is the quantity
    `calibrated_spread_skill_ratio`'s derivation actually uses, and
    averaging standard deviations directly would understate it (Jensen's
    inequality: `sqrt` is concave).
    """
    per_sample_variance = ensemble_forecast.var(dim=member_dim, ddof=1, skipna=True)
    mean_variance = (
        per_sample_variance.mean(dim=dim, skipna=True)
        if dim is not None
        else per_sample_variance.mean(skipna=True)
    )
    return mean_variance**0.5


def calibrated_spread_skill_ratio(n_members: float) -> float:
    """The RMSE(ensemble mean)/spread ratio a *perfectly calibrated*
    ensemble of `n_members` members should have -- not 1.0.

    Verified by simulation, not assumed from a remembered formula: for a
    reliable ensemble (obs and members drawn from the same distribution),
    `E[(ensemble_mean - obs)^2] = (1 + 1/M) * E[ensemble_variance]`, so
    `RMSE(mean) / spread` is expected to be `sqrt((M+1)/M)` -- e.g. ~1.054
    for M=9, ~1.061 for M=8, ~1.080 for M=6 (the real member counts this
    project's short-window leads produce, per docs/baseline-store.md), not
    a flat 1.0 regardless of ensemble size.

    A measured ratio well *above* this calibrated target indicates
    under-dispersion: the ensemble's spread is narrower than a calibrated
    ensemble of the same size would have, relative to its actual error.
    """
    return float(np.sqrt((n_members + 1) / n_members))


def spread_skill_ratio(
    ensemble_forecast: xr.DataArray,
    obs: xr.DataArray,
    dim: Dim = None,
    member_dim: str = "member",
) -> xr.DataArray:
    """RMSE of the ensemble mean divided by the ensemble spread.

    Compare against `calibrated_spread_skill_ratio(n_members)` for the same
    `n_members`, not against 1.0 -- a calibrated ensemble's own ratio is
    already greater than 1 due to finite-ensemble-size noise (see that
    function's docstring). A ratio well above the calibrated target
    indicates under-dispersion.
    """
    ensemble_mean = ensemble_forecast.mean(dim=member_dim, skipna=True)
    spread = ensemble_spread(ensemble_forecast, dim=dim, member_dim=member_dim)
    return rmse(ensemble_mean, obs, dim=dim) / spread


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


def contingency_counts(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    thresholds: Sequence[float] = IMD_RAIN_THRESHOLDS_MM,
    dim: Dim = None,
) -> dict[float, dict[str, xr.DataArray]]:
    """The raw 2x2 counts behind `contingency_scores`, per threshold.

    Written out separately because every score above is a ratio, and a ratio
    cannot be re-aggregated over a bootstrap resample of days: the CSI of a
    resampled period is not the mean of its daily CSIs. Recording
    hits/misses/false_alarms/correct_negatives per day instead lets
    `scripts/run_scorecard.py` rebuild any categorical score inside each
    bootstrap replicate, which is what makes a CI for SEDI or CSI possible
    at all (see `weavr.score_io`).
    """
    results: dict[float, dict[str, xr.DataArray]] = {}
    for threshold in thresholds:
        edges = np.array([-np.inf, threshold, np.inf])
        table = xs.Contingency(obs, forecast, edges, edges, dim=dim)
        results[threshold] = {
            "hits": table.hits(),
            "misses": table.misses(),
            "false_alarms": table.false_alarms(),
            "correct_negatives": table.correct_negatives(),
        }
    return results


def sedi(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    dim: Dim = None,
) -> xr.DataArray:
    """Symmetric Extremal Dependence Index (Ferro & Stephenson 2011).

        SEDI = [ln F - ln H - ln(1-F) + ln(1-H)] / [ln F + ln H + ln(1-F) + ln(1-H)]

    where **H is the hit rate** `hits / (hits + misses)` and **F is the false
    alarm RATE** `false_alarms / (false_alarms + correct_negatives)`. Note
    that F here is the false alarm *rate* (POFD), not the false alarm *ratio*
    that `contingency_scores` reports as `"far"` -- they have different
    denominators, and substituting one for the other silently produces a
    plausible but wrong number. This function takes the rate from
    `xskillscore.Contingency.false_alarm_rate()`.

    1 is a perfect forecast, 0 is no skill. SEDI exists because CSI and ETS
    both degenerate towards 0 as an event gets rarer, so they cannot tell a
    genuinely skilful rare-event forecast from a useless one -- exactly the
    regime this project cares about (step 01 measured POD = 0 at 204.5 mm for
    every source). SEDI is base-rate independent and non-degenerate in that
    limit, which is why `docs/preregistration.md` uses it for H3 and H7
    rather than CSI.

    **Returns NaN when H or F is exactly 0 or 1**, where one of the logarithms
    is undefined. This is common and expected at high thresholds: F = 0 means
    the forecast never issued a false alarm, which happens whenever it never
    forecasts the event at all. NaN is returned rather than a clipped or
    substituted value, because any substitution invents skill information the
    data does not contain -- and rather than raising, because a whole-lead
    scorecard should still be computable when one threshold is degenerate.
    Callers must treat NaN as "not estimable here", not as zero skill.
    """
    edges = np.array([-np.inf, threshold, np.inf])
    table = xs.Contingency(obs, forecast, edges, edges, dim=dim)
    hit_rate = table.hit_rate()
    false_alarm_rate = table.false_alarm_rate()

    degenerate = (
        (hit_rate <= 0.0)
        | (hit_rate >= 1.0)
        | (false_alarm_rate <= 0.0)
        | (false_alarm_rate >= 1.0)
    )
    # Evaluate the logs on safe values, then mask: computing them on 0/1
    # first would emit divide-by-zero warnings for cells we are about to
    # discard anyway.
    safe_h = hit_rate.where(~degenerate, 0.5)
    safe_f = false_alarm_rate.where(~degenerate, 0.5)

    log_h, log_f = np.log(safe_h), np.log(safe_f)
    log_1mh, log_1mf = np.log(1.0 - safe_h), np.log(1.0 - safe_f)

    numerator = log_f - log_h - log_1mf + log_1mh
    denominator = log_f + log_h + log_1mf + log_1mh
    result = numerator / denominator
    return result.where(~degenerate & (denominator != 0.0))


def twcrps_ensemble(
    ensemble_forecast: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    dim: Dim = None,
    member_dim: str = "member",
) -> xr.DataArray:
    """Threshold-weighted CRPS with weight `1{z >= threshold}` (Allen et al. 2023).

    Uses the chaining-function identity: for the indicator weight
    `w(z) = 1{z >= t}`, the chaining function is `v(z) = max(z, t)`, and

        twCRPS(F, y; t) = CRPS( max(members, t), max(y, t) )

    so this is a plain CRPS of transformed inputs and reuses `crps()`
    directly rather than reimplementing the integral. Verified against
    brute-force numerical integration of the weighted CRPS definition in
    `tests/test_verify.py::TestTwcrpsEnsemble`, not assumed from the identity.

    Why it matters here: ordinary CRPS is dominated by the many light and dry
    days, so a change that only affects the heaviest rainfall barely moves
    it. twCRPS at t = 64.5 mm scores only the part of the distribution above
    IMD's heavy-rain threshold, which is what `docs/preregistration.md`'s H7
    needs to detect whether tail repair actually helped.

    Lower is better, and it is a proper scoring rule for the weighted
    forecast problem, so it cannot be gamed by hedging.

    Delegates to `crps_large_ensemble` rather than `crps`. The two are
    exactly equivalent (see that function), but this one is also called on
    the 434-member climatological reference, where the pairwise route needs
    tens of gigabytes -- which is how that limit was found here.
    """
    # `.clip(min=...)` rather than `np.maximum`: identical result on a
    # DataArray, but it keeps the static type as DataArray instead of Any.
    return crps_large_ensemble(
        ensemble_forecast.clip(min=threshold),
        obs.clip(min=threshold),
        dim=dim,
        member_dim=member_dim,
    )


def pit_values(
    ensemble_forecast: xr.DataArray,
    obs: xr.DataArray,
    member_dim: str = "member",
    seed: int = 0,
) -> xr.DataArray:
    """Randomized PIT (rank) values of the observation within the ensemble.

    For an ensemble of `m` members, with `U ~ Uniform(0, 1)` drawn per cell:

        PIT = [ #(members < obs) + U * (1 + #(members == obs)) ] / (m + 1)

    A calibrated ensemble gives PIT values uniform on (0, 1); a histogram of
    them is the rank histogram. U-shaped means under-dispersive (the
    observation keeps falling outside the ensemble -- which
    `docs/phase2-ensemble-baseline-results.md` already measured for the
    lagged AI ensembles, spread-skill 4.0-6.9 against a target near 1.05);
    dome-shaped means over-dispersive; sloped means biased.

    The randomization is what makes the values continuous rather than
    discrete ranks, and it is essential for precipitation specifically:
    many cells are exactly 0 in both the forecast and the observation, so
    ties are the norm rather than an edge case. Breaking them deterministically
    would pile every dry day into one histogram bin and make a perfectly
    calibrated forecast look badly miscalibrated. `seed` keeps this
    reproducible.
    """
    n_members = ensemble_forecast.sizes[member_dim]
    below = (ensemble_forecast < obs).sum(dim=member_dim)
    equal = (ensemble_forecast == obs).sum(dim=member_dim)

    rng = np.random.default_rng(seed)
    uniform = xr.DataArray(
        rng.random(below.shape), coords=below.coords, dims=below.dims
    )
    return (below + uniform * (1.0 + equal)) / (n_members + 1.0)


def pit_values_csgd(
    mean: xr.DataArray,
    std: xr.DataArray,
    shift: float | xr.DataArray,
    obs: xr.DataArray,
    seed: int = 0,
) -> xr.DataArray:
    """Randomized PIT for a censored, shifted gamma predictive distribution.

    The counterpart of `pit_values` for `weavr.emos`'s fitted CSGD, whose
    parameters are the gamma mean/std plus a left shift `delta <= 0`, with
    the distribution left-censored at zero (Scheuerer & Hamill 2015, eq. 1-2;
    shape `kappa = mean^2 / std^2`, scale `theta = std^2 / mean`).

    The censoring puts an atom of probability at exactly 0, so the PIT is
    **randomized over that atom** -- the standard treatment for a
    discrete-continuous mixture:

        obs > 0:  PIT = F(obs) = GammaCDF(obs - delta; kappa, theta)
        obs = 0:  PIT = U * F(0),  U ~ Uniform(0, 1)

    Without the randomization every dry day would return the same value
    `F(0)`, producing a huge spike in the histogram that looks like severe
    miscalibration but is purely an artefact of the point mass. Since most
    cells in a monsoon field are dry, that artefact would dominate the
    diagnostic entirely.

    Returns NaN where the parameters are degenerate (non-positive mean or
    std), which is what `weavr.emos`'s documented fallback produces for a
    bin it refused to fit.
    """
    from scipy.stats import gamma as _gamma

    mean_v = np.asarray(mean, dtype=float)
    std_v = np.asarray(std, dtype=float)
    shift_v = np.asarray(shift, dtype=float)
    obs_v = np.asarray(obs, dtype=float)

    valid = (mean_v > 0.0) & (std_v > 0.0)
    safe_mean = np.where(valid, mean_v, 1.0)
    safe_std = np.where(valid, std_v, 1.0)

    shape = safe_mean**2 / safe_std**2
    scale = safe_std**2 / safe_mean

    prob_at_zero = _gamma.cdf(-shift_v, a=shape, scale=scale)
    cdf_at_obs = _gamma.cdf(obs_v - shift_v, a=shape, scale=scale)

    rng = np.random.default_rng(seed)
    uniform = rng.random(np.broadcast(mean_v, obs_v).shape)

    values = np.where(obs_v > 0.0, cdf_at_obs, uniform * prob_at_zero)
    values = np.where(valid, values, np.nan)

    template = obs if isinstance(obs, xr.DataArray) else mean
    return xr.DataArray(values, coords=template.coords, dims=template.dims)


def reliability_table(
    probability_forecast: xr.DataArray,
    obs_binary: xr.DataArray,
    n_bins: int = 10,
) -> xr.Dataset:
    """Reliability (calibration) table: does "70% chance" happen 70% of the time?

    Bins the forecast probabilities into `n_bins` equal-width bins over
    [0, 1] and, for each bin, returns the mean forecast probability, the
    observed event frequency, and the count. Plotting observed frequency
    against forecast probability gives the reliability diagram; the diagonal
    is perfect calibration, and a curve below it means over-forecasting.

    This is the diagnostic behind the one claim on WEAVR's slides that is
    purely about trust rather than accuracy: an "X% chance of >115.6 mm"
    statement is only meaningful if the X is calibrated. Step 01's audit
    marked that example as illustrative precisely because nothing had
    measured it yet.

    Bins are right-closed on the final bin so that a forecast probability of
    exactly 1.0 lands in the top bin rather than falling outside the range.
    Empty bins return NaN frequencies with a count of 0, not 0.0
    frequencies -- an unoccupied bin carries no evidence, and recording it
    as a perfectly reliable zero would be an invention.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    prob = np.asarray(probability_forecast, dtype=float).ravel()
    event = np.asarray(obs_binary, dtype=float).ravel()
    keep = np.isfinite(prob) & np.isfinite(event)
    prob, event = prob[keep], event[keep]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `np.digitize` with right=False puts 1.0 into an overflow bin; clip it
    # back into the last real bin.
    index = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)

    counts = np.bincount(index, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.bincount(index, weights=event, minlength=n_bins) / counts
        forecast_mean = np.bincount(index, weights=prob, minlength=n_bins) / counts
    observed[counts == 0] = np.nan
    forecast_mean[counts == 0] = np.nan

    centres = (edges[:-1] + edges[1:]) / 2.0
    return xr.Dataset(
        {
            "forecast_probability": ("bin", forecast_mean),
            "observed_frequency": ("bin", observed),
            "count": ("bin", counts),
        },
        coords={"bin": centres},
    )


def brier_skill_score(
    brier: xr.DataArray | float, brier_reference: xr.DataArray | float
) -> xr.DataArray | float:
    """`1 - BS / BS_ref`. Positive means better than the reference.

    0 means exactly as good as the reference (usually climatology), and
    negative means worse -- which for a forecast system is the result that
    matters most to report, since "worse than climatology" is a real and
    common outcome at long leads.

    Returns NaN where the reference score is 0 (a perfect reference leaves
    no room for skill to be defined) rather than dividing by zero.
    """
    reference = xr.DataArray(brier_reference) if not isinstance(
        brier_reference, xr.DataArray
    ) else brier_reference
    with np.errstate(invalid="ignore", divide="ignore"):
        skill = 1.0 - brier / reference
    return skill.where(reference != 0.0)


def crpss(
    crps_score: xr.DataArray | float, crps_reference: xr.DataArray | float
) -> xr.DataArray | float:
    """`1 - CRPS / CRPS_ref`. Positive means better than the reference.

    The CRPS counterpart of `brier_skill_score`, with the same conventions
    and the same NaN-on-zero-reference behaviour.
    """
    reference = xr.DataArray(crps_reference) if not isinstance(
        crps_reference, xr.DataArray
    ) else crps_reference
    with np.errstate(invalid="ignore", divide="ignore"):
        skill = 1.0 - crps_score / reference
    return skill.where(reference != 0.0)
