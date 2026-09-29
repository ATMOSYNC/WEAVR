"""Stacking and combining combiners (EMOS-CSG + BMA) -- Step 13.

Combines the parametric precision and discrimination of EMOS-CSG with the
calibration and reliability of BMA (Javanshiri 2021, Ji 2025):
1. Per-bin selection: select the best combiner per (rain bin, lead) on the train fold.
2. Quantile averaging (Vincentization): average predictive quantiles across combiners
   (Lichtendahl et al. 2013), preserving sharpness without over-dispersing.
3. Linear pool: standard probability averaging across predictive distributions.

Quantile-based CRPS approximation:
Approximates CRPS via integration over pinball losses across quantile levels:
CRPS(F, y) = 2 * int_0^1 (y - q_alpha) * (alpha - I(y < q_alpha)) d alpha
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

from weavr.emos import _TINY

# Standard 99 quantile levels (0.01 to 0.99) for accurate numerical CRPS
DEFAULT_QUANTILE_LEVELS = np.linspace(0.01, 0.99, 99)


@dataclass
class PerBinSelectionResult:
    """Mapping of (bin, lead) -> chosen combiner, with fallback annotations."""

    mapping: dict[tuple[str, int], str]
    fallback_bins: set[tuple[str, int]] = field(default_factory=set)
    pooled_best: str = "emos_csg"
    tie_breaker: str = "emos_csg"

    def get(self, rain_bin: str, lead: int) -> str:
        """Get chosen combiner for (bin, lead), falling back to pooled_best if unseen."""
        return self.mapping.get((rain_bin, lead), self.pooled_best)


def select_per_bin(
    train_scores: pd.DataFrame,
    bin_col: str = "bin",
    lead_col: str = "lead",
    combiner_col: str = "combiner",
    score_col: str = "crps",
    tie_breaker: str = "emos_csg",
) -> PerBinSelectionResult:
    """Select the best combiner per (rain bin, lead) based on minimum CRPS on train fold.

    Tie-break rule:
    If two combiners have identical scores (within 1e-9 tolerance), prefer `tie_breaker`
    (default: 'emos_csg') to reward parametric efficiency.

    Sparse/missing bins:
    If a bin has no valid entries, falls back to the overall pooled-best combiner
    at that lead or domain-wide, flagged in `fallback_bins`.
    """
    if train_scores.empty:
        return PerBinSelectionResult(mapping={}, fallback_bins=set(), pooled_best=tie_breaker)

    df = train_scores.dropna(subset=[bin_col, lead_col, combiner_col, score_col]).copy()
    if df.empty:
        return PerBinSelectionResult(mapping={}, fallback_bins=set(), pooled_best=tie_breaker)

    # Determine overall pooled best combiner across all available bins
    pooled_scores = df.groupby(combiner_col)[score_col].mean()
    if pooled_scores.empty:
        pooled_best = tie_breaker
    else:
        min_pooled = pooled_scores.min()
        candidates = pooled_scores[np.isclose(pooled_scores, min_pooled, atol=1e-9)].index.tolist()
        pooled_best = tie_breaker if tie_breaker in candidates else sorted(candidates)[0]

    # Map per (bin, lead)
    mapping: dict[tuple[str, int], str] = {}
    grouped = df.groupby([bin_col, lead_col])

    for (b, lead_val), group in grouped:
        lead_int = int(lead_val)
        bin_str = str(b)
        combiner_means = group.groupby(combiner_col)[score_col].mean()
        if combiner_means.empty:
            continue
        min_score = combiner_means.min()
        best_candidates = combiner_means[
            np.isclose(combiner_means, min_score, atol=1e-9)
        ].index.tolist()

        if tie_breaker in best_candidates:
            chosen = tie_breaker
        else:
            chosen = sorted(best_candidates)[0]

        mapping[(bin_str, lead_int)] = chosen

    return PerBinSelectionResult(
        mapping=mapping,
        fallback_bins=set(),
        pooled_best=pooled_best,
        tie_breaker=tie_breaker,
    )


def predictive_quantiles_csgd(
    mean: np.ndarray | float,
    std: np.ndarray | float,
    shift: np.ndarray | float,
    levels: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Compute predictive quantiles Q_Y(tau) for Censored Shifted Gamma Distribution.

    For CSGD:
    Y = max(0, X + delta), where X ~ Gamma(kappa, theta), delta <= 0.
    Point mass at 0: p0 = P(Y = 0) = F_X(-delta).
    For tau in (0, 1):
    If tau <= p0: Q_Y(tau) = 0.
    If tau > p0:  Q_Y(tau) = delta + theta * F_X^{-1}(tau).

    Parameters
    ----------
    mean : array_like or float
        Underlying unshifted gamma mean (location).
    std : array_like or float
        Underlying unshifted gamma standard deviation.
    shift : array_like or float
        Shift parameter delta (typically <= 0).
    levels : array_like, optional
        Quantile levels tau in (0, 1). Defaults to DEFAULT_QUANTILE_LEVELS.

    Returns
    -------
    quantiles : np.ndarray
        Array of shape `input_shape + (len(levels),)`. Monotone non-decreasing
        in the last axis, non-negative.
    """
    if levels is None:
        levels_arr = DEFAULT_QUANTILE_LEVELS
    else:
        levels_arr = np.asarray(levels, dtype=float)

    if np.any((levels_arr <= 0.0) | (levels_arr >= 1.0)):
        raise ValueError(f"Quantile levels must be strictly in (0, 1), got {levels_arr}")

    # Ensure levels are sorted
    if not np.all(np.diff(levels_arr) >= 0):
        levels_arr = np.sort(levels_arr)

    mean_arr = np.asarray(mean, dtype=float)
    std_arr = np.asarray(std, dtype=float)
    shift_arr = np.asarray(shift, dtype=float)

    broadcast_shape = np.broadcast_shapes(mean_arr.shape, std_arr.shape, shift_arr.shape)
    mean_b = np.clip(np.broadcast_to(mean_arr, broadcast_shape).astype(float), _TINY, None)
    std_b = np.clip(np.broadcast_to(std_arr, broadcast_shape).astype(float), _TINY, None)
    shift_b = np.broadcast_to(shift_arr, broadcast_shape).astype(float)

    kappa = mean_b**2 / std_b**2
    theta = std_b**2 / mean_b

    # Point mass at 0: P(X <= -shift)
    # If -shift <= 0 (e.g. shift >= 0), point mass is 0
    c = np.clip(-shift_b / theta, 0.0, None)
    p0 = gamma_dist.cdf(c, a=np.clip(kappa, _TINY, None))

    # Add trailing dimension for levels: broadcast_shape + (len(levels_arr),)
    # p0 shape: broadcast_shape + (1,)
    p0_expanded = np.expand_dims(p0, axis=-1)
    kappa_expanded = np.expand_dims(kappa, axis=-1)
    theta_expanded = np.expand_dims(theta, axis=-1)
    shift_expanded = np.expand_dims(shift_b, axis=-1)

    levels_expanded = levels_arr.reshape((1,) * len(broadcast_shape) + (-1,))

    # Compute gamma.ppf for each level
    # When level <= p0, quantile is 0.0
    # gamma.ppf is strictly positive
    raw_ppf = gamma_dist.ppf(levels_expanded, a=np.clip(kappa_expanded, _TINY, None))
    uncensored_q = shift_expanded + theta_expanded * raw_ppf

    # Zero out where level <= p0 or uncensored_q <= 0 or mean is degenerate
    zero_mask = mean_b <= _TINY
    quantiles = np.where(levels_expanded <= p0_expanded, 0.0, np.clip(uncensored_q, 0.0, None))
    quantiles = np.where(np.expand_dims(zero_mask, axis=-1), 0.0, quantiles)

    # Ensure numerical monotonicity
    quantiles = np.maximum.accumulate(quantiles, axis=-1)
    return quantiles


def predictive_quantiles_bma(
    bma_samples: np.ndarray,
    levels: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Compute empirical predictive quantiles from BMA mixture Monte Carlo samples.

    Parameters
    ----------
    bma_samples : np.ndarray
        Array of shape `(..., n_samples)` produced by `weavr.bma.sample_bma_mixture`.
    levels : array_like, optional
        Quantile levels tau in (0, 1).

    Returns
    -------
    quantiles : np.ndarray
        Array of shape `(...) + (len(levels),)`.
    """
    if levels is None:
        levels_arr = DEFAULT_QUANTILE_LEVELS
    else:
        levels_arr = np.asarray(levels, dtype=float)

    if np.any((levels_arr <= 0.0) | (levels_arr >= 1.0)):
        raise ValueError(f"Quantile levels must be strictly in (0, 1), got {levels_arr}")

    # Compute quantiles along last axis
    q = np.quantile(bma_samples, levels_arr, axis=-1)
    # Transpose so levels are on the last axis
    # np.quantile puts quantile axis first
    q = np.moveaxis(q, 0, -1)
    q = np.clip(q, 0.0, None)
    q = np.maximum.accumulate(q, axis=-1)
    return q


def quantile_average(
    q_a: np.ndarray,
    q_b: np.ndarray,
    weight: float = 0.5,
) -> np.ndarray:
    """Vincentization: quantile averaging of two predictive distributions.

    q_avg(tau) = (1 - weight) * q_a(tau) + weight * q_b(tau).
    Lichtendahl et al. (2013) establish that quantile averaging preserves sharpness
    and unimodality compared to linear probability pooling.

    Parameters
    ----------
    q_a, q_b : np.ndarray
        Predictive quantiles with matching shapes, where the last axis represents quantile levels.
    weight : float
        Weight on distribution b (1 - weight on distribution a). Must be in [0, 1].

    Returns
    -------
    q_avg : np.ndarray
        Averaged quantiles, guaranteed non-decreasing along the last axis.
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight must be between 0 and 1, got {weight}")

    q_a_arr = np.asarray(q_a, dtype=float)
    q_b_arr = np.asarray(q_b, dtype=float)

    if q_a_arr.shape != q_b_arr.shape:
        raise ValueError(
            f"Quantile arrays must have identical shapes, got {q_a_arr.shape} vs {q_b_arr.shape}"
        )

    q_avg = (1.0 - weight) * q_a_arr + weight * q_b_arr
    q_avg = np.clip(q_avg, 0.0, None)

    # Monotonicity check
    diffs = np.diff(q_avg, axis=-1)
    if np.any(diffs < -1e-9):
        # Numerical precision safeguard
        q_avg = np.maximum.accumulate(q_avg, axis=-1)

    return q_avg


def linear_pool(
    prob_a: np.ndarray | float,
    prob_b: np.ndarray | float,
    weight: float = 0.5,
) -> np.ndarray | float:
    """Linear probability pool of two probabilistic forecasts: (1 - w)*P_a + w*P_b."""
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight must be between 0 and 1, got {weight}")
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    return np.clip((1.0 - weight) * pa + weight * pb, 0.0, 1.0)


def crps_from_quantiles(
    quantiles: np.ndarray,
    levels: np.ndarray | list[float],
    obs: np.ndarray | float,
) -> np.ndarray | float:
    """Approximate CRPS using pinball loss integration over predictive quantiles.

    CRPS(F, y) = 2 * int_0^1 rho_tau(y - q(tau)) d tau
    where rho_tau(u) = u * (tau - I(u < 0)) = max(tau*u, (tau-1)*u).

    Evaluated via trapezoidal integration across sorted levels.

    Parameters
    ----------
    quantiles : np.ndarray
        Array of predictive quantiles with levels on the last axis, shape (..., K).
    levels : array_like
        Sorted quantile levels in (0, 1), shape (K,).
    obs : array_like or float
        Observed values, shape (...).

    Returns
    -------
    crps : np.ndarray or float
        Approximated CRPS values matching the shape of `obs`.
    """
    levels_arr = np.asarray(levels, dtype=float)
    q_arr = np.asarray(quantiles, dtype=float)
    obs_arr = np.asarray(obs, dtype=float)

    if q_arr.shape[-1] != len(levels_arr):
        msg = (
            f"Last dimension of quantiles ({q_arr.shape[-1]}) must match "
            f"levels length ({len(levels_arr)})"
        )
        raise ValueError(msg)

    # Broadcast obs to align with quantiles
    # obs shape: (...) -> (... , 1)
    obs_expanded = np.expand_dims(obs_arr, axis=-1)
    levels_expanded = levels_arr.reshape((1,) * len(obs_arr.shape) + (-1,))

    u = obs_expanded - q_arr
    # Pinball loss: u * (tau - (u < 0))
    pinball = np.maximum(levels_expanded * u, (levels_expanded - 1.0) * u)

    # Integrate 2 * pinball over tau in [0, 1] using composite trapezoidal rule
    # with boundary points tau=0 (q=0 or q(0)) and tau=1 (q(1))
    # Standard practice with equispaced or general levels: np.trapezoid along the last axis
    # Note: 2 * trapezoid(pinball, levels_arr, axis=-1)
    trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapezoid_fn is None:
        raise RuntimeError("Neither np.trapezoid nor np.trapz is available")
    crps = 2.0 * trapezoid_fn(pinball, x=levels_arr, axis=-1)

    # Return float if obs was scalar
    if obs_arr.shape == ():
        return float(np.clip(crps, 0.0, None))
    return np.clip(crps, 0.0, None)
