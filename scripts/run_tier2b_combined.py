#!/usr/bin/env python3
"""Run Tier 2b combined combiners (EMOS-CSG + BMA meta-blend) -- Step 13.

Step 07 established two calibrated predictive distributions for the same
target: **EMOS-CSG** (a per-rain-bin censored shifted gamma) and **BMA** (a
per-(bin, region) mixture of those gammas fitted to each source). Step 13
asks the question neither parent can answer alone -- is a *combination* of
the two better calibrated than either one?

Arms, all under the same two-season LOYO as every other tier:

1. `emos_csg` -- parent 1. The EMOS source with the lower **train**-fold
   CRPS between GraphCast and IFS-ENS (the issue's "best source"; chosen on
   train so nothing is selected on the test season).
2. `bma` -- parent 2.
3. `per_bin` -- per (rain bin, lead), whichever parent has the lower
   **train**-fold CRPS, with the tie-break and pooled-fallback rules already
   in `weavr.stacking.select_per_bin`.
4. `quantile_avg` -- Vincentization (Lichtendahl et al. 2013): average the
   two predictive quantile functions, with the weight fitted on train by a
   grid search over CRPS. `quantile_avg_fixed` repeats it at the
   pre-registered w = 0.5, so a fitted weight can be told apart from a
   generic "half and half".
5. `linear_pool` -- average exceedance probabilities in probability space,
   then invert the pooled CDF back onto quantile levels.

**One CRPS estimator for every arm.** H10 is a claim about the *combination
rule*. Scoring CSGD by its exact closed form while scoring the blends by a
quantile approximation would confound the two effects, so a difference could
come from the blend or from the estimator. Every arm's CRPS therefore comes
from `weavr.stacking.crps_from_quantiles` applied to that arm's own
predictive quantiles.

**BMA quantiles are Monte Carlo.** `predictive_quantiles_bma` takes order
statistics of `n_samples` draws, so its 0.99 quantile is only as resolved as
the draw count allows; `--n-samples` trades runtime against tail resolution.

**H10 (pre-registered)**: a combined arm beats *both* parents on CRPS at 3
or more of 5 lead times, with the 95% paired block-bootstrap CI on the
per-day CRPS difference excluding 0. Days are paired, so a negative
difference means the combination is better. The primary combined arm is
nominated by **train**-fold CRPS rather than by test score, and every arm's
verdict is reported separately so a fail is as visible as a pass.

## Status

`main()` is still the original stub. What has landed is the numerical layer
every arm is built from, plus its tests: `cdf_at_threshold`,
`quantiles_from_cdf`, `predictive_mean_from_cdf`, `pit_summary`,
`fit_quantile_weight` and `select_emos_source`.

The remaining work is the fold loop: fitting the parents, filling per-cell
quantile grids group by group as Tier 2 does, then writing the domain,
per-day and paired-difference files.

Two bugs found while testing this layer are worth recording, because both
were silent rather than loud. The CDF inversion searched for `F <= tau`
where the definition of a quantile is `inf{v : F(v) >= tau}`, which returns
the wrong end of the distribution for every level. And `pit_summary` handed
a per-cell array of observations to a scalar-threshold signature, which only
surfaced once a real array met it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tier2_hierarchical_baseline import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    DEFAULT_IFS_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
    align_all_sources,
    load_graphcast_ensemble,
    load_hres_forecast,
    load_ifs_ensemble,
)

from weavr.bma import fit_hierarchical_bma, sample_bma_mixture  # noqa: E402
from weavr.emos import fit_emos_csg, predict_csgd_params  # noqa: E402
from weavr.rain_bins import classify_rain_bin  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.score_io import (  # noqa: E402
    guard_result_overwrites,
    resolve_result_paths,
    write_per_day_scores,
)
from weavr.significance import paired_difference_ci  # noqa: E402
from weavr.splits import iter_evaluation_folds  # noqa: E402
from weavr.stacking import (
    DEFAULT_QUANTILE_LEVELS,
    crps_from_quantiles,
    predictive_quantiles_bma,
    predictive_quantiles_csgd,
    quantile_average,
    select_per_bin,
)
from weavr.stores import (  # noqa: E402
    open_multi_season,
    resolve_store_paths,
)

LEAD_HOURS = [24, 48, 72, 96, 120]
IMD_THRESHOLDS = [7.5, 35.5, 64.5, 115.6, 204.5]

#: The EMOS source used as parent 1 is chosen between these on the TRAIN
#: fold, never on the test season.
EMOS_SOURCES = ("graphcast", "ifs_ens")

#: Arms that combine the two parents, in report order.
COMBINED_ARMS = ("per_bin", "quantile_avg", "quantile_avg_fixed", "linear_pool")

#: Every arm the runner scores, parents first, in report order.
ALL_ARMS = ("emos_csg", "bma") + COMBINED_ARMS

#: Shared value grid for the linear pool's CDF and for every arm's
#: predictive mean. Daily accumulations above 600 mm sit far outside any
#: fitted tail here, so truncating costs the CRPS integral nothing
#: measurable while keeping the `(cells, grid)` arrays near 24 MB.
VALUE_GRID_MM = np.arange(0.0, 600.0 + 1.0, 1.0)

#: Candidate Vincentization weights for the train-fold fit. 0.0 and 1.0 are
#: included deliberately: a blend that cannot beat a parent should be
#: reported as collapsing onto that parent, not forced into a spurious
#: interior optimum.
WEIGHT_GRID = np.round(np.arange(0.0, 1.0 + 0.05, 0.05), 2)


def combine_predictive_quantiles(
    csgd_mean: np.ndarray,
    csgd_std: np.ndarray,
    csgd_shift: np.ndarray,
    bma_samples: np.ndarray,
    levels: np.ndarray | None = None,
    weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q_csgd, q_bma, q_avg) predictive quantiles."""
    if levels is None:
        levels = DEFAULT_QUANTILE_LEVELS

    q_csgd = predictive_quantiles_csgd(csgd_mean, csgd_std, csgd_shift, levels=levels)
    q_bma = predictive_quantiles_bma(bma_samples, levels=levels)
    q_avg = quantile_average(q_csgd, q_bma, weight=weight)
    return q_csgd, q_bma, q_avg


def compute_brier_score(prob: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    """Mean Brier score for event obs > threshold."""
    event = (obs > threshold).astype(float)
    return float(np.mean((prob - event) ** 2))


def score_quantiles_crps(
    quantiles: np.ndarray,
    levels: np.ndarray,
    obs: np.ndarray,
) -> float:
    """Compute mean CRPS from predictive quantiles against observations."""
    crps_vals = crps_from_quantiles(quantiles, levels, obs)
    return float(np.mean(crps_vals))


def cdf_at_threshold(
    quantiles: np.ndarray, levels: np.ndarray, threshold: float | np.ndarray
) -> np.ndarray:
    """`F(threshold)` for every row of `quantiles`, by linear interpolation.

    `quantiles` is `(n_cells, n_levels)`, non-decreasing along the last axis;
    `levels` is the matching `(n_levels,)` vector. `threshold` is either one
    scalar shared by every cell, or one value per cell -- the PIT summary
    needs the second form, since each cell is asking about its own
    observation.

    A plain step count would quantise every probability to `1 / n_levels`
    (1% at the 99 default levels), far too coarse for a Brier score, so the
    value is interpolated between the two bracketing levels. A flat stretch
    -- the point mass at zero, where many levels share `q = 0` -- would
    divide by zero, so the span is guarded and the step value is used there.

    A threshold outside the reported quantile range saturates rather than
    extrapolating: below every quantile the answer is 0, above every
    quantile it is 1. That is the honest answer for an exceedance
    probability, where such a threshold means the event is certain or
    impossible, and it is why the clamp is on the probability and not on
    the grid index.
    """
    q = np.asarray(quantiles, dtype=float)
    lv = np.asarray(levels, dtype=float)
    n_levels = q.shape[-1]

    t = np.asarray(threshold, dtype=float)
    if t.ndim == 0:
        t = np.full(q.shape[0], float(t))
    else:
        t = np.broadcast_to(t, q.shape[:-1])

    # Index of the first level at or above the threshold, per cell. Clamped
    # to [1, n_levels - 1] so a bracketing pair always exists; the clamped
    # case is the saturating one, handled by the clip on the way out.
    above = q <= t[..., None]
    k = np.clip(above.sum(axis=-1), 1, n_levels - 1)
    rows = np.arange(q.shape[0])
    q_lo = q[rows, k - 1]
    q_hi = q[rows, k]
    span = q_hi - q_lo
    safe = np.where(span > 1e-12, span, 1.0)
    frac = np.where(span > 1e-12, (t - q_lo) / safe, 0.0)
    # Interpolate in probability space, not index space, so a caller passing
    # non-uniformly spaced levels still gets the right answer.
    return np.clip(lv[k - 1] + frac * (lv[k] - lv[k - 1]), 0.0, 1.0)


def quantiles_from_cdf(
    cdf: np.ndarray, grid: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Invert a non-increasing CDF sampled on `grid` back onto `levels`.

    `cdf` is `(n_cells, n_grid)`, non-increasing along the grid axis. For
    each level `tau` this takes `Q(tau) = inf{v : F(v) >= tau}` -- the
    first grid value at which the CDF has climbed to `tau`. Because `F` is
    non-increasing, that is the first index where `cdf >= tau`, a
    first-crossing search that vectorises over cells where `np.interp` does
    not.

    Where the CDF never climbs that far -- all the mass sitting above the
    grid -- `Q` is beyond the last grid value, so it is truncated there
    rather than collapsed to the bottom of the grid.
    """
    cdf = np.asarray(cdf, dtype=float)
    grid = np.asarray(grid, dtype=float)
    out = np.empty((cdf.shape[0], len(levels)), dtype=float)
    for j, tau in enumerate(levels):
        reached = cdf >= tau
        has = reached.any(axis=-1)
        idx = np.where(has, reached.argmax(axis=-1), cdf.shape[-1] - 1)
        out[:, j] = grid[idx]
    return np.maximum.accumulate(out, axis=-1)


def predictive_mean_from_cdf(cdf: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """`E[Y]` from a CDF sampled on `grid`, as the integral of `1 - F`.

    Applied to every arm rather than to each parent's own native mean, so
    that `rmse_mm` and `bias_mm` are computed from comparable quantities.
    BMA's Monte Carlo mean is not reproducible across arms and would put
    sampling noise into the very differences H10 is judged on.
    """
    cdf = np.asarray(cdf, dtype=float)
    return np.trapezoid(1.0 - cdf, np.asarray(grid, dtype=float), axis=-1)


def pit_summary(quantiles: np.ndarray, levels: np.ndarray, obs: np.ndarray) -> dict:
    """PIT / reliability summary read off predictive quantiles.

    An observation's PIT value is the probability the forecast assigns at
    or below it, `F(obs)`. A calibrated forecast makes that uniform on
    [0, 1], so the mean absolute deviation from 0.5 is a monotone
    reliability signal, and the share of observations falling inside the
    forecast's own central 90% interval should sit near 0.9.
    """
    f_obs = cdf_at_threshold(quantiles, levels, obs)
    central = (f_obs >= 0.05) & (f_obs <= 0.95)
    return {
        "pit_mean": float(np.mean(f_obs)),
        "pit_mad_from_half": float(np.mean(np.abs(f_obs - 0.5))),
        "pit_central_90_coverage": float(np.mean(central)),
        "n_pit_cells": int(f_obs.size),
    }


def fit_quantile_weight(
    q_csgd: np.ndarray,
    q_bma: np.ndarray,
    levels: np.ndarray,
    obs: np.ndarray,
    grid: np.ndarray = WEIGHT_GRID,
) -> tuple[float, float]:
    """Grid-search the Vincentization weight against train-fold CRPS.

    Returns `(weight, train_crps)`. `grid` spans 0.0 to 1.0 inclusive so a
    blend that cannot beat either parent is reported as having collapsed
    onto it, instead of the search inventing an interior mixture that only
    looks competitive.
    """
    best_w, best_crps = 0.5, np.inf
    for w in grid:
        q = quantile_average(q_csgd, q_bma, weight=float(w))
        crps = float(np.mean(crps_from_quantiles(q, levels, obs)))
        if crps < best_crps:
            best_w, best_crps = float(w), crps
    return best_w, best_crps


def select_emos_source(
    train_crps: dict[str, float], tie_breaker: str = "graphcast"
) -> str:
    """Pick the EMOS source with the lower train CRPS, with a fixed tie-break.

    Non-finite entries are dropped first: a source with no trainable cells
    reports NaN, and letting that win a `min()` would silently select an
    empty model.
    """
    finite = {s: v for s, v in train_crps.items() if np.isfinite(v)}
    if not finite:
        return tie_breaker
    best = min(finite.values())
    tied = [s for s, v in finite.items() if abs(v - best) <= 1e-9]
    return tie_breaker if tie_breaker in tied else sorted(tied)[0]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Step 07 made every other runner multi-season; this one still wasn't.

    The fold loop has to iterate `iter_evaluation_folds` over two seasons, so
    it needs the same `--*-stores` (plural) interface as Tier 2 and Tier 3, with
    the singular `--store` forms kept for the archived v1 single-season runs.
    Defaults point at the daily 2018 + 2020 stores, which is the only evidence
    base a LOYO evaluation is meaningful on.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stores", nargs="+", default=None,
        help="Baseline stores (default: 2018 + 2020 daily)",
    )
    parser.add_argument("--baseline-store", default=None, help="Legacy single store")
    parser.add_argument(
        "--lagged-stores", nargs="+", default=None,
        help="Lagged ensemble stores (default: 2018 + 2020 daily)",
    )
    parser.add_argument("--lagged-store", default=None, help="Legacy single store")
    parser.add_argument(
        "--ifs-ensemble-stores", nargs="+", default=None,
        help="IFS-ENS stores (default: 2018 + 2020 daily)",
    )
    parser.add_argument("--ifs-ensemble-store", default=None, help="Legacy single store")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument(
        "--results-dir", default="results",
        help=(
            "Directory for the per-day files and the default parent for the "
            "CSVs. Point this at a scratch directory to keep a run out of the "
            "committed results/."
        ),
    )
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="default: <results-dir>/tier2b_combined.csv")
    parser.add_argument("--paired-out-csv", type=Path, default=None,
                        help="default: <results-dir>/tier2b_combined_paired.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite existing result CSVs.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Monte Carlo draws for the BMA quantiles.")
    parser.add_argument(
        "--train-out-csv", type=Path, default=None,
        help="Train-fold diagnostics CSV (default: <results-dir>/tier2b_combined_train.csv)")
    parser.add_argument(
        "--train-stride", type=int, default=4,
        help="Subsample stride for the train-fold Vincentization weight grid.")
    parser.add_argument(
        "--quantile-weight", type=float, default=None,
        help="Fixed Vincentization weight for the `quantile_avg_fixed` arm.",
    )
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument(
        "--leads", nargs="+", type=int, default=None,
        help=(
            "Subset of lead times to run. Exists to time and rehearse a single "
            "lead against the real stores; the default is all five, which is "
            "what H10's 3-of-5 rule needs."
        ),
    )
    return parser.parse_args(args)


def _values(da: xr.DataArray) -> np.ndarray:
    """`da`'s values with dims forced to `(sample, latitude, longitude)`."""
    return da.transpose("sample", "latitude", "longitude").values.astype(float)


def _labels(da: xr.DataArray) -> np.ndarray:
    """Rain-bin labels as strings.

    `classify_rain_bin` returns strings ("dry", "light", ...), so this cannot
    share `_values`' float cast -- the mask comparisons below work the same on
    a string array.
    """
    return da.transpose("sample", "latitude", "longitude").values.astype(str)


def cross_cdf(
    quantiles: np.ndarray, levels: np.ndarray, targets: np.ndarray, chunk_rows: int = 4000
) -> np.ndarray:
    """`F(targets[i, j])` for every row `i` and every target `j`.

    `targets` shares the level axis with `quantiles`, so this evaluates one
    parent's CDF at the *other* parent's quantiles. That is all the linear pool
    needs to bracket its own inverse.

    Rows are chunked because the comparison is `(rows, n_levels, n_targets)`
    booleans: at the 99-level default and a 4000-row chunk that is 39M, which
    is fine, but an unchunked group of 70k cells would be 686M and would need
    half a gigabyte for a single temporary.
    """
    n_levels = quantiles.shape[-1]
    out = np.empty(targets.shape, dtype=float)
    for start in range(0, quantiles.shape[0], chunk_rows):
        stop = min(start + chunk_rows, quantiles.shape[0])
        q_chunk = quantiles[start:stop]
        t_chunk = targets[start:stop]
        k = np.sum(q_chunk[:, :, None] <= t_chunk[:, None, :], axis=1)
        k_lo = np.clip(k - 1, 0, n_levels - 1)
        k_hi = np.clip(k, 0, n_levels - 1)
        q_lo = np.take_along_axis(q_chunk, k_lo, axis=-1)
        q_hi = np.take_along_axis(q_chunk, k_hi, axis=-1)
        span = q_hi - q_lo
        safe = np.where(span > 1e-12, span, 1.0)
        frac = np.where(span > 1e-12, (t_chunk - q_lo) / safe, 0.0)
        lv_lo = levels[k_lo]
        out[start:stop] = np.clip(lv_lo + frac * (levels[k_hi] - lv_lo), 0.0, 1.0)
    return out


def linear_pool_quantiles(
    q_a: np.ndarray,
    q_b: np.ndarray,
    levels: np.ndarray,
    weight: float = 0.5,
) -> np.ndarray:
    """Invert the linear pool `(1 - w) F_a + w F_b` back onto `levels`.

    Averaging in probability space is not the same as averaging in quantile
    space, so this arm needs its own inverse, and there is no closed form for
    `F_pool^-1`.

    The bracket is exact rather than heuristic. `F_a(q_a(tau)) = tau`, so
    `F_pool(q_a(tau))` sits above `tau` exactly when `F_b(q_a(tau))` does --
    which is exactly when `q_b(tau) < q_a(tau)`, and symmetrically for the other
    parent. So `F_pool(lo(tau)) <= tau <= F_pool(hi(tau))` at every level, where
    `lo` and `hi` are the parents' own quantiles at that level, and the pooled
    quantile is recovered by interpolating the inverse across that bracket.

    That interpolation is exact whenever `F_pool` is linear between `lo` and
    `hi`, and slightly low in curvature otherwise, so this arm carries a
    resolution the other five do not. It is kept as the plan's comparison-only
    arm precisely so it cannot quietly become the recommended combiner, and
    `test_tier2b_line_pool_is_comparable_to_quantile_averaging` pins it inside
    the parents' CRPS range so an inversion failure cannot pass unnoticed.
    """
    tau = np.asarray(levels, dtype=float)
    f_b_at_a = cross_cdf(q_b, levels, q_a)
    f_a_at_b = cross_cdf(q_a, levels, q_b)
    a_le_b = q_a <= q_b
    one_minus = 1.0 - weight
    f_lo = np.where(
        a_le_b, one_minus * tau + weight * f_b_at_a, one_minus * f_a_at_b + weight * tau
    )
    f_hi = np.where(
        a_le_b, one_minus * f_a_at_b + weight * tau, one_minus * tau + weight * f_b_at_a
    )
    lo = np.minimum(q_a, q_b)
    hi = np.maximum(q_a, q_b)
    span = f_hi - f_lo
    safe = np.where(span > 1e-12, span, 1.0)
    frac = np.clip(np.where(span > 1e-12, (tau - f_lo) / safe, 0.5), 0.0, 1.0)
    return np.maximum.accumulate(lo + frac * (hi - lo), axis=-1)


def arm_cell_metrics(
    quantiles: np.ndarray,
    levels: np.ndarray,
    obs_cells: np.ndarray,
    thresholds: list[float],
) -> dict[str, np.ndarray]:
    """Per-cell CRPS, PIT and exceedance probabilities for one arm.

    Every arm is reduced here so a difference between arms can only come from
    the predictive distribution and not from one arm being scored differently
    from another.
    """
    exceed = np.empty((len(thresholds), quantiles.shape[0]), dtype=float)
    for i, threshold in enumerate(thresholds):
        exceed[i] = 1.0 - cdf_at_threshold(quantiles, levels, float(threshold))
    return {
        "crps": crps_from_quantiles(quantiles, levels, obs_cells),
        "pit": cdf_at_threshold(quantiles, levels, obs_cells),
        "exceed": exceed,
    }


def _accumulate_day_sums(
    acc: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    day_index: np.ndarray,
    thresholds: list[float],
) -> None:
    """Add one group's per-cell metrics into per-day sums and cell counts."""
    np.add.at(acc["crps_sum"], day_index, np.nan_to_num(metrics["crps"]))
    np.add.at(acc["pit_sum"], day_index, np.nan_to_num(metrics["pit"]))
    for i, _ in enumerate(thresholds):
        residual = metrics["exceed"][i] - metrics["obs_binary"]
        np.add.at(acc["brier_num"][i], day_index, residual**2)
    np.add.at(acc["n_cells"], day_index, 1)


def _new_day_accumulator(n_days: int, thresholds: list[float]) -> dict[str, np.ndarray]:
    return {
        "crps_sum": np.zeros(n_days, dtype=float),
        "pit_sum": np.zeros(n_days, dtype=float),
        "brier_num": np.zeros((len(thresholds), n_days), dtype=float),
        "n_cells": np.zeros(n_days, dtype=float),
    }


def _day_table(
    acc: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    fold: str,
    thresholds: list[float],
) -> pd.DataFrame:
    """Turn per-day sums into the one-row-per-day table the scorecard reads."""
    n = acc["n_cells"]
    keep = n > 0
    table = pd.DataFrame(
        {
            "date": dates[keep],
            "fold": fold,
            "n_cells": n[keep].astype(int),
            "crps_mm": acc["crps_sum"][keep] / n[keep],
        }
    )
    for i, threshold in enumerate(thresholds):
        table[f"brier_{threshold}"] = acc["brier_num"][i][keep] / n[keep]
    table["pit_mean"] = acc["pit_sum"][keep] / n[keep]
    return table.reset_index(drop=True)


def train_fold_diagnostics(
    emos_results: dict[str, dict],
    bma_results: dict,
    forecasts: dict[str, xr.DataArray],
    obs_aligned: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    region_v: np.ndarray,
    valid: np.ndarray,
    levels: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    stride: int,
    thresholds: list[float],
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, float]:
    """Score both parents and the whole weight grid on the **train** fold.

    Returns the per-(bin, lead, combiner) train CRPS table that
    `select_per_bin` needs, the mean train CRPS per combiner (for
    `select_emos_source`), the per-weight CRPS sums for both EMOS sources, and
    the number of train cells the weight grid was fitted on.

    `stride` subsamples train cells for the weight grid. The grid is 21
    weights and the fold holds about 1.06M cells per lead, so evaluating every
    weight on every cell is 21 full CRPS passes for a single scalar; striding
    keeps the fit affordable and costs the weight estimate a little precision,
    which is why the pre-registered fixed weight of 0.5 is reported alongside.
    """
    obs_v = _values(obs_aligned)
    bin_v = _labels(rain_bin_labels)
    mean_arrays: dict[str, np.ndarray] = {}
    spread_arrays: dict[str, np.ndarray | None] = {}
    group_valid = valid.copy()
    for source, da in forecasts.items():
        if "member" in da.dims:
            mean_arrays[source] = _values(da.mean(dim="member", skipna=True))
            spread_arrays[source] = _values(da.std(dim="member", ddof=1, skipna=True))
        else:
            mean_arrays[source] = _values(da)
            spread_arrays[source] = None
        group_valid = group_valid & ~np.isnan(mean_arrays[source])

    cell_totals: dict[tuple[str, str], list[float]] = {}
    weight_sum = {source: np.zeros(len(WEIGHT_GRID)) for source in EMOS_SOURCES}
    weight_count = 0
    rows: list[dict] = []

    def _record(combiner: str, bin_label: str, crps: np.ndarray) -> None:
        finite = crps[np.isfinite(crps)]
        if finite.size == 0:
            return
        entry = cell_totals.setdefault((combiner, bin_label), [0.0, 0.0])
        entry[0] += float(finite.sum())
        entry[1] += float(finite.size)

    for (bin_label, region), bma_result in bma_results.items():
        cell_mask = (bin_v == bin_label) & (region_v == region) & group_valid
        indices = np.flatnonzero(cell_mask.reshape(-1))
        if indices.size == 0:
            continue
        take = indices[::stride] if stride > 1 else indices
        cell_mean = {s: mean_arrays[s].reshape(-1)[take] for s in mean_arrays}
        cell_spread = {
            s: (None if spread_arrays[s] is None else spread_arrays[s].reshape(-1)[take])
            for s in mean_arrays
        }
        obs_cells = obs_v.reshape(-1)[take]

        quantiles: dict[str, np.ndarray] = {}
        for source in EMOS_SOURCES:
            result = emos_results[source].get(bin_label)
            spread = cell_spread[source]
            if result is None or spread is None:
                quantiles[source] = np.zeros((take.size, len(levels)), dtype=float)
                continue
            loc, scale, shift = predict_csgd_params(result, cell_mean[source], spread)
            quantiles[source] = predictive_quantiles_csgd(loc, scale, shift, levels)
            _record(
                f"emos_{source}", bin_label,
                crps_from_quantiles(quantiles[source], levels, obs_cells),
            )

        bma_samples = sample_bma_mixture(
            bma_result, cell_mean, cell_spread, rng, n_samples=n_samples
        )
        q_bma = predictive_quantiles_bma(bma_samples, levels)
        _record("bma", bin_label, crps_from_quantiles(q_bma, levels, obs_cells))

        for source, q_src in quantiles.items():
            for i, w in enumerate(WEIGHT_GRID):
                blended = quantile_average(q_src, q_bma, weight=float(w))
                weight_sum[source][i] += float(
                    np.sum(crps_from_quantiles(blended, levels, obs_cells))
                )
        weight_count += int(take.size)
        del bma_samples

    rows = [
        {
            "bin": bin_label,
            "lead": 0,
            "combiner": combiner,
            "crps": total / n_cells,
            "n_cells": n_cells,
        }
        for (combiner, bin_label), (total, n_cells) in cell_totals.items()
        if n_cells > 0
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {}, weight_sum, 0
    mean_by_combiner = {
        combiner: float(total / n_cells)
        for combiner, group in table.groupby("combiner")
        for total, n_cells in [(group["crps"].sum(), group["n_cells"].sum())]
    }
    return table, mean_by_combiner, weight_sum, weight_count


def main() -> int:
    args = parse_args()
    paths = resolve_result_paths(
        args.results_dir,
        {
            "out_csv": "tier2b_combined.csv",
            "train_out_csv": "tier2b_combined_train.csv",
            "paired_out_csv": "tier2b_combined_paired.csv",
        },
        {
            "out_csv": args.out_csv,
            "train_out_csv": args.train_out_csv,
            "paired_out_csv": args.paired_out_csv,
        },
    )
    args.out_csv = paths["out_csv"]
    args.train_out_csv = paths["train_out_csv"]
    args.paired_out_csv = paths["paired_out_csv"]
    guard_result_overwrites(paths.values(), force=args.force)

    baseline_paths = resolve_store_paths(
        args.baseline_stores, args.baseline_store, DEFAULT_BASELINE_DAILY_STORES,
        "data/baseline_2020_jjas.zarr",
    )
    lagged_paths = resolve_store_paths(
        args.lagged_stores, args.lagged_store, DEFAULT_LAGGED_DAILY_STORES,
        "data/lagged_ensemble_inputs_2020_jjas.zarr",
    )
    ifs_paths = resolve_store_paths(
        args.ifs_ensemble_stores, args.ifs_ensemble_store, DEFAULT_IFS_DAILY_STORES,
        "data/ifs_ens_2020_jjas.zarr",
    )

    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    rng = np.random.default_rng(0)
    levels = DEFAULT_QUANTILE_LEVELS

    print("Tier 2b combined combiners: EMOS-CSG, BMA and their combinations under LOYO")
    print(f"Baseline store(s): {baseline_paths}")
    print(f"Lagged store(s): {lagged_paths}")
    print(f"IFS ensemble store(s): {ifs_paths}")
    print(f"Quantile levels: {len(levels)} | BMA draws: {args.n_samples}")

    domain_rows: list[dict] = []
    train_rows: list[dict] = []
    leads = list(args.leads or LEAD_HOURS)
    per_day_pool: dict[str, dict[int, list[pd.DataFrame]]] = {
        arm: {lead: [] for lead in leads} for arm in ALL_ARMS
    }

    for lead_hours in (args.leads or LEAD_HOURS):
        graphcast_ensemble = load_graphcast_ensemble(lagged_paths, lead_hours)
        ifs_ensemble = load_ifs_ensemble(ifs_paths, lead_hours)
        hres_forecast = load_hres_forecast(baseline_paths, lead_hours)
        forecasts, obs_aligned = align_all_sources(
            graphcast_ensemble, ifs_ensemble, hres_forecast, obs
        )
        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))
        rain_bin_labels = classify_rain_bin(
            forecasts["graphcast"].mean(dim="member", skipna=True)
        )
        dates = pd.DatetimeIndex(obs_aligned["sample"].values)
        region_2d = region_labels.transpose("latitude", "longitude").values
        region_v = np.broadcast_to(region_2d[None, :, :], _values(obs_aligned).shape)

        for train_mask, test_mask, split_label in folds:
            emos_results = {
                source: fit_emos_csg(
                    forecasts[source],
                    obs_aligned,
                    rain_bin_labels,
                    train_mask,
                    source=source,
                )
                for source in EMOS_SOURCES
            }
            bma_results = fit_hierarchical_bma(
                forecasts, obs_aligned, rain_bin_labels, region_labels, train_mask
            )

            base_valid = ~np.isnan(_values(obs_aligned))
            train_valid = base_valid & np.broadcast_to(
                train_mask[:, None, None], base_valid.shape
            )
            test_valid = base_valid & np.broadcast_to(
                test_mask[:, None, None], base_valid.shape
            )

            train_table, train_mean, weight_sum, weight_count = train_fold_diagnostics(
                emos_results, bma_results, forecasts, obs_aligned, rain_bin_labels,
                region_v, train_valid, levels, rng, args.n_samples, args.train_stride,
                IMD_THRESHOLDS,
            )
            best_source = select_emos_source(
                {
                    source: train_mean.get(f"emos_{source}", float("nan"))
                    for source in EMOS_SOURCES
                }
            )
            # `select_per_bin` chooses between the two *parents*, so the
            # already-chosen EMOS source's row is relabelled `emos_csg` and the
            # other source's row is dropped. Left in, the selection would
            # choose between graphcast and ifs_ens -- which is what
            # `select_emos_source` already decided -- and `bma` would never
            # win a bin, making the arm a copy of the EMOS parent.
            selection_table = train_table[
                train_table["combiner"].isin([f"emos_{best_source}", "bma"])
            ].copy()
            selection_table["combiner"] = selection_table["combiner"].replace(
                {f"emos_{best_source}": "emos_csg"}
            )
            selection = select_per_bin(selection_table, lead_col="lead")
            if weight_count:
                train_crps_grid = weight_sum[best_source] / weight_count
                fitted_w = float(WEIGHT_GRID[int(np.argmin(train_crps_grid))])
                best_w_crps = float(train_crps_grid.min())
                # An interior optimum only counts as a fit if it beats *both*
                # endpoints by a real margin. A blend that cannot improve on
                # either parent is reported as collapsing onto it at w=0 or
                # w=1, not as an interior mixture that happens to look best.
                fitted = (
                    0.0 < fitted_w < 1.0
                    and best_w_crps < train_crps_grid[0] - 1e-9
                    and best_w_crps < train_crps_grid[-1] - 1e-9
                )
            else:
                train_crps_grid = np.full(len(WEIGHT_GRID), np.inf)
                fitted_w, best_w_crps, fitted = 0.5, float("nan"), False

            if args.quantile_weight is not None:
                best_w, fitted = args.quantile_weight, True
            elif fitted:
                best_w = fitted_w
            else:
                best_w = 0.5

            per_bin_train_crps = float("nan")
            if not train_table.empty:
                chosen_rows = train_table.copy()
                chosen_rows["chosen"] = [
                    selection.mapping.get((str(row.bin), lead_hours), selection.pooled_best)
                    for row in chosen_rows.itertuples()
                ]
                picked = chosen_rows[
                    chosen_rows["combiner"] == chosen_rows["chosen"]
                ]
                if not picked.empty:
                    per_bin_train_crps = float(
                        picked["crps"].sum() / picked["n_cells"].sum()
                    )
            # A per-bin map that picks the same parent for every bin is not a
            # combination: its predictive distribution *is* that parent, so it
            # can tie the parent it collapsed onto but never beat it, and H10
            # asks whether a combination beats both. Nominated anyway, it would
            # guarantee a zero difference and turn H10 into a foregone fail.
            # It is reported as collapsed, and excluded from nomination for the
            # same reason an endpoint weight is reported as not fitted.
            per_bin_degenerate = len(set(selection.mapping.values())) < 2
            if per_bin_degenerate:
                nominated = "quantile_avg"
                nomination_reason = (
                    f"per_bin collapsed onto {selection.pooled_best} at every bin, "
                    "so it is a copy of that parent rather than a combination"
                )
            elif best_w_crps <= per_bin_train_crps:
                nominated = "quantile_avg"
                nomination_reason = "lowest train-fold CRPS among the combinations"
            else:
                nominated = "per_bin"
                nomination_reason = "lowest train-fold CRPS among the combinations"
            train_rows.append(
                {
                    "lead_hours": lead_hours,
                    "fold": split_label,
                    "emos_source": best_source,
                    "train_crps_emos_graphcast": train_mean.get("emos_graphcast", float("nan")),
                    "train_crps_emos_ifs_ens": train_mean.get("emos_ifs_ens", float("nan")),
                    "train_crps_bma": train_mean.get("bma", float("nan")),
                    "vincentization_weight": best_w,
                    "vincentization_weight_fitted": fitted,
                    "vincentization_train_crps": best_w_crps,
                    "per_bin_train_crps": per_bin_train_crps,
                    "nominated_primary_arm": nominated,
                    "nomination_reason": nomination_reason,
                    "per_bin_degenerate": per_bin_degenerate,
                    "per_bin_fallback_bins": ",".join(sorted(selection.fallback_bins)),
                    "weight_grid_cells": weight_count,
                }
            )

            accumulators = {
                arm: _new_day_accumulator(len(dates), IMD_THRESHOLDS)
                for arm in ALL_ARMS
            }
            per_bin_choice = selection.pooled_best
            test_day_index = np.broadcast_to(np.arange(len(dates))[:, None, None], test_valid.shape)
            obs_v = _values(obs_aligned)

            for (bin_label, region), bma_result in bma_results.items():
                bin_v = _labels(rain_bin_labels)
                cell_mask = (bin_v == bin_label) & (region_v == region) & test_valid
                indices = np.flatnonzero(cell_mask.reshape(-1))
                if indices.size == 0:
                    continue
                cell_obs = obs_v.reshape(-1)[indices]
                cell_mean: dict[str, np.ndarray] = {}
                cell_spread: dict[str, np.ndarray | None] = {}
                for source, da in forecasts.items():
                    if "member" in da.dims:
                        member_mean = _values(da.mean(dim="member", skipna=True))
                        member_spread = _values(da.std(dim="member", ddof=1, skipna=True))
                        cell_mean[source] = member_mean.reshape(-1)[indices]
                        cell_spread[source] = member_spread.reshape(-1)[indices]
                    else:
                        cell_mean[source] = _values(da).reshape(-1)[indices]
                        cell_spread[source] = None

                emos_result = emos_results[best_source].get(bin_label)
                emos_spread = cell_spread[best_source]
                if emos_result is not None and emos_spread is not None:
                    loc, scale, shift = predict_csgd_params(
                        emos_result, cell_mean[best_source], emos_spread
                    )
                    q_emos = predictive_quantiles_csgd(loc, scale, shift, levels)
                else:
                    q_emos = np.zeros((indices.size, len(levels)), dtype=float)

                bma_samples = sample_bma_mixture(
                    bma_result, cell_mean, cell_spread, rng, n_samples=args.n_samples
                )
                q_bma = predictive_quantiles_bma(bma_samples, levels)
                del bma_samples

                per_bin_choice = selection.mapping.get(
                    (str(bin_label), lead_hours), selection.pooled_best
                )
                q_per_bin = q_emos if per_bin_choice == "emos_csg" else q_bma
                arms = {
                    "emos_csg": q_emos,
                    "bma": q_bma,
                    "per_bin": q_per_bin,
                    "quantile_avg": quantile_average(q_emos, q_bma, weight=best_w),
                    "quantile_avg_fixed": quantile_average(q_emos, q_bma, weight=0.5),
                    "linear_pool": linear_pool_quantiles(q_emos, q_bma, levels, weight=0.5),
                }
                day_index = test_day_index.reshape(-1)[indices]
                for arm, q in arms.items():
                    metrics = arm_cell_metrics(q, levels, cell_obs, IMD_THRESHOLDS)
                    metrics["obs_binary"] = cell_obs
                    _accumulate_day_sums(accumulators[arm], metrics, day_index, IMD_THRESHOLDS)

            print(
                f"[lead {lead_hours:>3}h | fold {split_label}] "
                f"source={best_source} w={best_w:.2f} "
                f"(fitted={fitted}) per_bin->{per_bin_choice} primary={nominated}"
                f"{' [per_bin collapsed]' if per_bin_degenerate else ''}"
            )
            for arm in ALL_ARMS:
                table = _day_table(accumulators[arm], dates, split_label, IMD_THRESHOLDS)
                per_day_pool[arm][lead_hours].append(table)
                n = table["n_cells"].to_numpy(dtype=float)
                crps = table["crps_mm"].to_numpy(dtype=float)
                row = {
                    "lead_hours": lead_hours,
                    "fold": split_label,
                    "arm": arm,
                    "emos_source": best_source,
                    "vincentization_weight": best_w,
                    "n_cells": int(n.sum()),
                    "crps_mm": float(np.average(crps, weights=n)) if n.sum() else float("nan"),
                    "pit_mean": float(
                        np.average(table["pit_mean"].to_numpy(dtype=float), weights=n)
                    )
                    if n.sum()
                    else float("nan"),
                }
                for threshold in IMD_THRESHOLDS:
                    key = f"brier_{threshold}"
                    row[key] = (
                        float(np.average(table[key].to_numpy(dtype=float), weights=n))
                        if n.sum()
                        else float("nan")
                    )
                domain_rows.append(row)

        # Per-day files are written once per (arm, lead) *after* both folds.
        # The filename carries the method and lead but not the fold, so writing
        # inside the fold loop let the second season overwrite the first: the
        # file looked complete and held half the evidence.
        for arm in ALL_ARMS:
            write_per_day_scores(
                f"tier2b_{arm}",
                lead_hours,
                pd.concat(per_day_pool[arm][lead_hours], ignore_index=True),
                out_dir=args.results_dir,
            )

    domain = pd.DataFrame(domain_rows)
    domain.to_csv(args.out_csv, index=False)
    pd.DataFrame(train_rows).to_csv(args.train_out_csv, index=False)

    print("")
    print("=" * 78)
    print("H10 (pre-registered): a combined arm beats BOTH parents on CRPS at")
    print("3 or more of 5 leads, with the paired block-bootstrap CI excluding 0.")
    print("=" * 78)

    pooled: dict[str, dict[int, pd.DataFrame]] = {}
    for arm in ALL_ARMS:
        pooled[arm] = {}
        for lead_hours in leads:
            frames = per_day_pool[arm][lead_hours]
            pooled[arm][lead_hours] = (
                pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
                if frames
                else pd.DataFrame()
            )

    train_frame = pd.DataFrame(train_rows)
    paired_rows: list[dict] = []
    nominated_by_lead = (
        train_frame.groupby("lead_hours")["nominated_primary_arm"].first().to_dict()
        if not train_frame.empty
        else {}
    )
    for lead_hours in leads:
        nominated = nominated_by_lead.get(lead_hours)
        if nominated is None:
            continue
        for parent in ("emos_csg", "bma"):
            combined = pooled[nominated][lead_hours]["crps_mm"].to_numpy(dtype=float)
            other = pooled[parent][lead_hours]["crps_mm"].to_numpy(dtype=float)
            if combined.size == 0 or combined.size != other.size:
                continue
            ci = paired_difference_ci(
                combined, other,
                block_days=args.block_days, n_resamples=args.n_resamples,
            )
            beats = bool(ci.estimate < 0.0 and ci.ci_lo > 0.0 and not ci.degenerate)
            paired_rows.append(
                {
                    "lead_hours": lead_hours,
                    "arm": nominated,
                    "vs_parent": parent,
                    "crps_difference_mm": float(ci.estimate),
                    "ci_low": float(ci.ci_lo),
                    "ci_high": float(ci.ci_hi),
                    "beats_parent": beats,
                    "n_days": int(combined.size),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.paired_out_csv, index=False)

    leads_beating_both = 0
    for lead_hours in leads:
        rows = paired[paired["lead_hours"] == lead_hours] if not paired.empty else pd.DataFrame()
        if rows.empty:
            continue
        both = bool(rows["beats_parent"].all()) and len(rows) == 2
        leads_beating_both += int(both)
        detail = " | ".join(
            f"vs {r.vs_parent}: d={r.crps_difference_mm:+.5f} "
            f"CI[{r.ci_low:+.5f},{r.ci_high:+.5f}]"
            f"{' PASS' if r.beats_parent else ' fail'}"
            + (" (tie: nominated arm is this parent)"
               if r.crps_difference_mm == 0.0 and r.ci_high == 0.0 else "")
            for r in rows.itertuples()
        )
        print(f"  lead {lead_hours:>3}h [{nominated_by_lead.get(lead_hours)}]: {detail}")

    print("")
    if leads_beating_both >= 3:
        print(
            f"H10 PASS: the nominated combined arm beat both parents at "
            f"{leads_beating_both}/{len(leads)} leads."
        )
        print("Slide 2's 'strongest combiner per bin' claim may return, with the CI.")
    else:
        print(
            f"H10 FAIL: the nominated combined arm beat both parents at "
            f"{leads_beating_both}/5 leads (needs 3)."
        )
        print("Slide 2 keeps no 'strongest combiner per bin' claim.")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.train_out_csv}")
    print(f"Wrote {args.paired_out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
