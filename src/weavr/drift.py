"""Rolling verification and drift detection -- issue #8's last checkbox.

Reuses `weavr.verify`'s existing metrics on a trailing window of
accumulated samples, rather than the one-shot, full-season scoring every
`run_tierN_*.py` script does today. Two real findings, checked rather than
assumed, shape this module:

## No explicit "model upgraded" signal exists anywhere in this project

The step-3 prompt this module implements named "a new GraphCast/Pangu/IFS
version" as the event drift detection should catch, and asked to check
whether any model-version identifier already exists before writing
detection logic that assumes one. Checked in both places this project
actually touches a model's output:

1. **The three historical store-building scripts**
   (`scripts/build_baseline_store.py`, `scripts/build_lagged_ensemble_store.py`,
   `scripts/build_ifs_ensemble_store.py`) record `status`/`variables`/day
   counts per source in their manifests, but never the WeatherBench 2
   archive path actually pulled -- so two different archive vintages of
   the same named source would look identical in the manifest. Fixed here:
   each now also records `source_archive_path` (the exact `zarr_path`
   string, WeatherBench 2's own closest thing to a version identifier --
   it changes when WB2 republishes a model under a new date-range/path) in
   its manifest entry, documented in `docs/baseline-store.md`.
2. **The live `ecmwf_open_data.py` fetch path** -- what
   `docs/phase6-operational-scope.md` decided the daily pipeline actually
   uses (AIFS/IFS/HRES, not the frozen WeatherBench 2 archives). Live-tested
   directly (not assumed): a real fetched IFS dataset's own GRIB metadata
   (`GRIB_centre`, `GRIB_subCentre`, `GRIB_edition`, ...) carries no model
   cycle or version field at all. ECMWF publishes model-version changes via
   its own external service-status channel, not embedded in the open-data
   GRIB output this project's client reads.

**So there is no machine-readable "model upgraded" event to hook anywhere
in this project's real data paths.** The practical, working proxy for "a
model changed" is therefore the same statistical drift check this module
needs to build for ordinary skill monitoring anyway: a sustained shift in
rolling skill/bias beyond what this project's own real historical record
shows as ordinary day-to-day variance. A version-string change (once one is
ever published somewhere this project can reach) would be a strictly easier
future signal to add on top of this, not a replacement for it.

## Trailing window: 14 samples, justified against this project's own numbers

Real per-sample domain-wide RMSE was computed against the actual baseline
store (`data/baseline_2020_jjas.zarr`, GraphCast precipitation, all 5
leads) to ground this rather than guessing: mean ranged 13.5-15.0mm, std
3.8-4.7mm across leads (std/mean ~28-33%, consistent across leads) -- see
`scripts/run_daily_verification.py`'s own module docstring for the exact
per-lead numbers. A trailing window needs to comfortably clear this
project's own established small-sample floor
(`weavr.bma.MIN_TRAIN_DAYS_PER_BIN = 5`) while staying short enough to
react within one real monsoon regime spell -- `docs/phase5-regime-covariate-scope.md`
found the 2020 JJAS season's one real "active" monsoon spell ran **14 real
days**. `TRAILING_WINDOW_SAMPLES = 14` reuses that same, already-established
real timescale rather than picking a new number.

## Drift threshold: baseline mean + 2 * baseline std

Chosen against the same real per-lead spread above: `std/mean` sits at
~28-33% consistently across leads, so a 2-standard-deviation threshold is a
real, checkable outlier bar (roughly a 95%-one-sided cutoff for
approximately-normal per-sample error, not an arbitrary percentage) rather
than a threshold tuned to this project's own small sample after the fact --
`DRIFT_THRESHOLD_STD_MULTIPLIER` is fixed before being applied to any new
data, the same convention `MIN_TRAIN_SAMPLES`/`MIN_TRAIN_DAYS_PER_BIN` each
used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

# See this module's own docstring for the real historical-spread numbers
# (computed against data/baseline_2020_jjas.zarr) these two constants are
# grounded in -- not picked without checking.
TRAILING_WINDOW_SAMPLES = 14
DRIFT_THRESHOLD_STD_MULTIPLIER = 2.0

MetricFn = Callable[[xr.DataArray, xr.DataArray], xr.DataArray]


@dataclass
class BaselineStats:
    """Mean/std of a metric's real historical score series, the reference a
    rolling value is checked against. `n_samples` is surfaced directly
    (rather than folded silently into `std`) since this project's own
    established convention (every combiner's own `is_fallback`) is to make
    a too-small sample count visible, not implicit.
    """

    metric_name: str
    mean: float
    std: float
    n_samples: int


def compute_baseline_stats(scores: Sequence[float], metric_name: str) -> BaselineStats:
    """Mean/std of a real historical score series for one metric.

    Requires at least 2 scores (`weavr.weighting.MIN_TRAIN_SAMPLES`'s own
    floor for a std to mean anything at all) -- raises rather than silently
    returning a std of 0 or NaN for a degenerate 0-1 sample series, which
    would make every later `detect_drift` call spuriously trigger (a
    threshold of `mean + k * 0` equals `mean` itself).
    """
    if len(scores) < 2:
        raise ValueError(
            f"need at least 2 historical scores to compute a baseline std, got {len(scores)}"
        )
    arr = np.asarray(scores, dtype=float)
    return BaselineStats(
        metric_name=metric_name,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
        n_samples=len(scores),
    )


@dataclass
class DriftResult:
    """One drift check's outcome. `is_drift` follows the same honest,
    visible-flag convention every combiner's own `is_fallback` uses --
    `reason` always states the real numbers compared, not just a boolean.
    """

    metric_name: str
    rolling_value: float
    baseline: BaselineStats
    threshold: float
    is_drift: bool
    reason: str


def detect_drift(
    rolling_value: float,
    baseline: BaselineStats,
    k: float = DRIFT_THRESHOLD_STD_MULTIPLIER,
) -> DriftResult:
    """Flags `rolling_value` as drifted if it exceeds `baseline.mean + k *
    baseline.std` -- a real, degrading shift (higher error/CRPS/|bias|),
    not a two-sided check, since this module's metrics (RMSE, CRPS, |bias|)
    are all "lower is better" with no meaningful "too low" failure mode.

    A baseline with too few samples to be trustworthy
    (`baseline.n_samples < weavr.bma.MIN_TRAIN_DAYS_PER_BIN`) still runs
    the check -- underdetection on a thin baseline is a real, named
    limitation (see this module's own docstring and
    `scripts/run_daily_verification.py`'s output), not silently hidden by
    refusing to flag at all.
    """
    threshold = baseline.mean + k * baseline.std
    is_drift = rolling_value > threshold
    reason = (
        f"rolling {baseline.metric_name}={rolling_value:.3f} vs. baseline "
        f"mean={baseline.mean:.3f} + {k}*std={baseline.std:.3f} = threshold "
        f"{threshold:.3f} (n={baseline.n_samples} historical samples)"
    )
    return DriftResult(
        metric_name=baseline.metric_name,
        rolling_value=rolling_value,
        baseline=baseline,
        threshold=threshold,
        is_drift=is_drift,
        reason=reason,
    )


def trailing_window_value(
    scores: Sequence[float], window: int = TRAILING_WINDOW_SAMPLES
) -> float:
    """Mean of the most recent `window` entries in an accumulating score
    series (each entry one real sample's domain-wide metric value, in
    whatever order the caller appends them).

    Uses fewer than `window` entries, rather than raising, when the series
    is shorter -- an accumulating daily pipeline (issue #8's real target)
    starts with 0 history and grows one real day at a time; refusing to
    produce a rolling value until exactly `window` days have accumulated
    would make this module unusable for its first two weeks of real
    operation. The shortfall is visible to the caller via `len(scores)`,
    the same "state it plainly" convention `BaselineStats.n_samples` uses.
    """
    if not scores:
        raise ValueError("no scores to compute a trailing-window value from")
    tail = scores[-window:]
    return float(np.mean(tail))


def rescore_trailing_window(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    metric_fn: MetricFn,
    sample_dim: str = "sample",
    window: int = TRAILING_WINDOW_SAMPLES,
) -> float:
    """Domain-wide `metric_fn` (one of `weavr.verify`'s own -- `rmse`,
    `bias`, or any other `(forecast, obs) -> scalar DataArray` metric) over
    the trailing `window` samples of an already `sample_dim`-aligned
    forecast/obs pair, reusing the exact metric function every
    `run_tierN_*.py` script already calls rather than a new scoring path.

    `forecast`/`obs` must already share `sample_dim` (e.g. via each
    `run_tierN_*.py` script's own IMD-day alignment) -- this function does
    no alignment of its own, the same division of responsibility
    `weavr.renormalize`'s own docstring uses (detection/alignment is the
    caller's job; this module scores what it's handed).
    """
    n = forecast.sizes[sample_dim]
    if n == 0:
        raise ValueError("no samples to rescore")
    tail_n = min(window, n)
    tail_forecast = forecast.isel({sample_dim: slice(-tail_n, None)})
    tail_obs = obs.isel({sample_dim: slice(-tail_n, None)})
    return float(metric_fn(tail_forecast, tail_obs))
