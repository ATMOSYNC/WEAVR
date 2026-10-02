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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weavr.stacking import (
    DEFAULT_QUANTILE_LEVELS,
    crps_from_quantiles,
    predictive_quantiles_bma,
    predictive_quantiles_csgd,
    quantile_average,
)

LEAD_HOURS = [24, 48, 72, 96, 120]
IMD_THRESHOLDS = [7.5, 35.5, 64.5, 115.6, 204.5]

#: The EMOS source used as parent 1 is chosen between these on the TRAIN
#: fold, never on the test season.
EMOS_SOURCES = ("graphcast", "ifs_ens")

#: Arms that combine the two parents, in report order.
COMBINED_ARMS = ("per_bin", "quantile_avg", "quantile_avg_fixed", "linear_pool")

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
        "--quantile-weight", type=float, default=0.5,
        help="Fixed Vincentization weight for the `quantile_avg_fixed` arm.",
    )
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    print(f"Tier 2b combined combiners runner initialized with output at {args.out_csv}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
