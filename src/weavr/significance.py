"""Confidence intervals and significance tests for paired forecast scores.

Every comparison in this project is of the form "method A beats method B on
the same days". Without an interval around that difference, a 3-day sampling
artefact is indistinguishable from a real improvement -- and step 01
demonstrated the problem concretely: the train-chosen and test-chosen best
single source disagreed at 2 of 5 leads, which is what noise looks like when
it is mistaken for a result. `docs/preregistration.md` therefore makes "the
CI of the paired difference excludes 0" part of the pass rule for every
claim, and this module is what computes it.

Two things here are easy to get subtly wrong, so both are done explicitly:

- **Days are resampled in blocks, not independently.** Daily forecast errors
  are autocorrelated -- a monsoon spell is wet for a week at a time -- and an
  i.i.d. bootstrap over such a series produces intervals that are far too
  narrow, making everything look significant. `block_bootstrap_indices`
  resamples contiguous blocks (Kunsch 1989's moving-block bootstrap), which
  preserves dependence up to the block length.
- **RMSE is never averaged.** `sqrt` is concave, so the mean of daily RMSEs
  is not the RMSE of the period (Jensen's inequality) and is biased low.
  `paired_difference_ci(aggregate="rmse")` therefore bootstraps daily *MSE*
  and takes the square root of the mean inside each resample. The same
  reasoning `weavr.verify.ensemble_spread` already applies to spread.

The bootstrap is paired throughout: both methods are resampled with the
*same* day indices in each replicate, so the shared day-to-day variation
(some days are simply harder to forecast) cancels instead of inflating the
interval.

References:
- Kunsch, H. R. (1989), "The jackknife and the bootstrap for general
  stationary observations", Annals of Statistics 17(3).
- Diebold, F. X. & Mariano, R. S. (1995), "Comparing predictive accuracy",
  Journal of Business & Economic Statistics 13(3).
- Newey, W. K. & West, K. D. (1987), heteroskedasticity- and
  autocorrelation-consistent covariance estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

Aggregate = Literal["mean", "rmse"]


@dataclass(frozen=True)
class DifferenceCI:
    """The paired difference `aggregate(a) - aggregate(b)` with its interval.

    `estimate` is computed on the real days, not as the mean of the bootstrap
    replicates -- the bootstrap supplies the interval, not the point value.
    Negative `estimate` means method A has the lower (better) loss.

    `degenerate` is True when the series is too short for the block bootstrap
    to resample at all (see `bootstrap_is_degenerate`). The bounds are then
    NaN and `significant` is False, whatever the estimate.
    """

    estimate: float
    ci_lo: float
    ci_hi: float
    n_days: int
    block_days: int
    n_resamples: int
    degenerate: bool = False

    @property
    def significant(self) -> bool:
        """True when the interval excludes 0 -- the pre-registration's rule.

        A NaN bound is never significant. This matters: a degenerate
        bootstrap produces a zero-width interval, which would otherwise read
        as "excludes 0" for *any* non-zero difference and report a 3-day
        sampling artefact as a proven result.
        """
        if not (np.isfinite(self.ci_lo) and np.isfinite(self.ci_hi)):
            return False
        return (self.ci_lo > 0.0) or (self.ci_hi < 0.0)


def bootstrap_is_degenerate(n_days: int, block_days: int) -> bool:
    """True when the moving-block bootstrap has only one possible resample.

    A moving block of length `L` over `n` days has `n - L + 1` possible start
    positions. When `L >= n` there is exactly one, so every replicate is the
    original series, the bootstrap distribution is a point mass, and the
    resulting interval has zero width.

    That is not a narrow interval -- it is *no* interval, and treating it as
    one is actively dangerous: with the 3-4 test days in the pre-step-07
    store and a 7-day block, it made 57% of scorecard comparisons look
    significant on a first run here. Callers must report NaN bounds instead,
    which `paired_difference_ci` does.
    """
    return n_days - min(block_days, n_days) + 1 <= 1


@dataclass(frozen=True)
class DieboldMarianoResult:
    """The DM statistic and its two-sided p-value."""

    statistic: float
    p_value: float
    n_days: int
    horizon: int


def block_bootstrap_indices(
    n_days: int,
    block_days: int = 7,
    n_resamples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Moving-block bootstrap day indices, shape `(n_resamples, n_days)`.

    Each replicate is built by drawing `ceil(n_days / block_days)` contiguous
    blocks of length `block_days`, with replacement, from the
    `n_days - block_days + 1` possible start positions, concatenating them
    and truncating to exactly `n_days`.

    **Blocks never wrap past the end of the series.** Start positions are
    drawn from `0 .. n_days - block_days` inclusive, so a block is always a
    genuinely contiguous stretch of real days. The cost is the standard
    moving-block edge effect: days near the start and end appear in fewer
    possible blocks and are therefore slightly under-represented. The
    circular alternative (wrapping) removes that bias but joins September to
    June, inventing a transition that never happened in a monsoon season.
    Under-weighting the edges is the lesser distortion here, and it is
    documented rather than silently chosen.

    `block_days` is clamped to `n_days` when the series is shorter than one
    block, which degenerates to resampling the whole series -- correct, and
    it keeps callers from having to special-case tiny test sets. With
    today's weekly store that is exactly what happens.

    The same `seed` always gives the same indices, so every CI in `results/`
    is reproducible.
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    if block_days < 1:
        raise ValueError(f"block_days must be >= 1, got {block_days}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    block = min(block_days, n_days)
    n_starts = n_days - block + 1
    n_blocks = int(np.ceil(n_days / block))

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_starts, size=(n_resamples, n_blocks))
    offsets = np.arange(block)
    # (n_resamples, n_blocks, block) -> flatten the block axis, then truncate.
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)
    return indices[:, :n_days]


def _paired_finite(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop days that are not finite in *both* series, keeping the pairing.

    Dropping per-series would compare the two methods on different days,
    which is exactly what the paired design exists to avoid.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"loss_a and loss_b must have the same length, got {a.shape} and {b.shape}"
        )
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def _aggregate(values: np.ndarray, how: Aggregate, axis: int = -1) -> np.ndarray:
    if how == "mean":
        return np.mean(values, axis=axis)
    if how == "rmse":
        # `values` are daily MSE; the RMSE of the period is sqrt of their
        # mean, never the mean of daily RMSEs (see the module docstring).
        return np.sqrt(np.mean(values, axis=axis))
    raise ValueError(f"aggregate must be 'mean' or 'rmse', got {how!r}")


def paired_difference_ci(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    block_days: int = 7,
    n_resamples: int = 1000,
    seed: int = 0,
    aggregate: Aggregate = "mean",
    confidence: float = 0.95,
) -> DifferenceCI:
    """Percentile-bootstrap CI for `aggregate(loss_a) - aggregate(loss_b)`.

    `loss_a` and `loss_b` are per-day domain-wide losses for the two methods
    on the *same* days, in the same order. Lower loss is better, so a
    negative estimate means A is better than B.

    With `aggregate="rmse"`, both inputs must be daily **MSE** (mm^2), not
    daily RMSE. The function takes the square root itself, inside each
    bootstrap replicate.

    Days are resampled jointly: replicate `i` uses one set of day indices for
    both methods. This is what makes the interval a *paired* one, and it is
    typically much narrower than two independent intervals because the
    shared "some days are hard" variation cancels.
    """
    a, b = _paired_finite(loss_a, loss_b)
    n_days = a.size
    if n_days < 1:
        raise ValueError("No days are finite in both series; nothing to compare.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    estimate = float(_aggregate(a, aggregate) - _aggregate(b, aggregate))

    if bootstrap_is_degenerate(n_days, block_days):
        return DifferenceCI(
            estimate=estimate,
            ci_lo=float("nan"),
            ci_hi=float("nan"),
            n_days=n_days,
            block_days=min(block_days, n_days),
            n_resamples=n_resamples,
            degenerate=True,
        )

    indices = block_bootstrap_indices(n_days, block_days, n_resamples, seed)
    replicate_diff = _aggregate(a[indices], aggregate, axis=1) - _aggregate(
        b[indices], aggregate, axis=1
    )

    tail = (1.0 - confidence) / 2.0
    ci_lo, ci_hi = np.quantile(replicate_diff, [tail, 1.0 - tail])
    return DifferenceCI(
        estimate=estimate,
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
        n_days=n_days,
        block_days=min(block_days, n_days),
        n_resamples=n_resamples,
    )


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
) -> DieboldMarianoResult:
    """Diebold-Mariano test of equal predictive accuracy (1995).

    Tests H0: `E[loss_a - loss_b] = 0`. The statistic is
    `mean(d) / sqrt(avar(mean(d)))` where `d = loss_a - loss_b` and the
    variance is the Newey-West HAC estimator using `horizon - 1` lags with
    Bartlett weights `1 - k/horizon`:

        avar(mean(d)) = (1/n) * [ gamma_0 + 2 * sum_{k=1}^{h-1} (1 - k/h) * gamma_k ]

    At `horizon=1` (a one-step-ahead forecast) the sum is empty and this
    reduces to the plain sample variance, which is the standard DM case. Use
    a larger `horizon` when the forecast errors are expected to be
    autocorrelated up to that many days.

    The p-value is two-sided from the standard normal, which is DM's own
    asymptotic reference distribution. It is asymptotic: with the handful of
    test days in today's weekly store, treat it as indicative and let the
    bootstrap CI carry the verdict. `docs/preregistration.md`'s pass rule
    deliberately depends on the CI, not on this p-value.

    Two degenerate cases are handled explicitly rather than producing a
    `nan` that would quietly propagate into a scorecard:

    - **Identical losses** (`d` all zero): returns statistic 0, p = 1. The
      methods are indistinguishable, which is the correct answer, not a
      failure.
    - **Non-zero mean with zero variance** (a constant, non-zero difference):
      returns `+/-inf` and p = 0. Every day agrees on the direction, so the
      evidence is as strong as this test can express.

    The Bartlett-weighted estimator is guaranteed non-negative, so a negative
    variance cannot arise here; a non-finite one still falls back to NaN with
    p = NaN rather than being reported as a result.
    """
    a, b = _paired_finite(loss_a, loss_b)
    n_days = a.size
    if n_days < 2:
        raise ValueError(f"Diebold-Mariano needs at least 2 paired days, got {n_days}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    d = a - b
    d_bar = float(np.mean(d))
    centred = d - d_bar

    gamma_0 = float(np.mean(centred**2))
    hac = gamma_0
    for lag in range(1, min(horizon, n_days)):
        gamma_k = float(np.mean(centred[lag:] * centred[:-lag]))
        hac += 2.0 * (1.0 - lag / horizon) * gamma_k

    if hac <= 0.0:
        if d_bar == 0.0:
            return DieboldMarianoResult(0.0, 1.0, n_days, horizon)
        signed_inf = float(np.sign(d_bar) * np.inf)
        return DieboldMarianoResult(signed_inf, 0.0, n_days, horizon)
    if not np.isfinite(hac):
        return DieboldMarianoResult(float("nan"), float("nan"), n_days, horizon)

    statistic = d_bar / np.sqrt(hac / n_days)
    p_value = float(2.0 * stats.norm.sf(abs(statistic)))
    return DieboldMarianoResult(float(statistic), p_value, n_days, horizon)


def lag1_autocorrelation(series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a daily series, used to choose a block length.

    `docs/preregistration.md` fixes the bootstrap block length at 7 days
    unless step 07 measures a different autocorrelation length, in which case
    the doc is amended in its own commit *before* any verdict is computed.
    This function is that measurement.

    Returns NaN for a series shorter than 3 points or with zero variance --
    a constant series has no meaningful autocorrelation, and returning 0
    would falsely suggest independence.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    centred = x - x.mean()
    denominator = float(np.sum(centred**2))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(centred[1:] * centred[:-1]) / denominator)
