"""Weight renormalization for a missing/late source at *prediction* time --
issue #8's real operational gap, distinct from every combiner's existing
`is_fallback` convention (which handles a source missing/degenerate *at fit
time*, already produced by `fit_region_weights`/`fit_regime_weights`/
`fit_emos_csg`/`fit_hierarchical_bma`).

This module does not re-fit anything. A weight/mixture-share set is already
fixed by training; when a forecast day arrives with one or more sources
missing, their already-fit weight is redistributed proportionally across
whichever sources actually showed up, matching Wanders & Wood-style
weighted-average semantics rather than treating an absent source as a zero
forecast (which would silently drag the blend toward zero) or crashing on a
missing dict key.

## Detecting "missing", checked against this project's real stores, not
assumed

A source failing to arrive shows up in two genuinely different shapes here,
both real, not hypothetical:

1. **The source never got pulled at all.** `scripts/build_baseline_store.py`,
   `scripts/build_lagged_ensemble_store.py`, and
   `scripts/build_ifs_ensemble_store.py` each write one independent Zarr
   *group* (or per-timestamp entry) per source, tracked by its own
   idempotent JSON manifest entry (`"status": "ok"` or `"status": "failed"`).
   A source whose pull failed outright is simply absent as a dict key
   wherever a caller assembles `{source_name: forecast}` -- there is no
   NaN-filled placeholder for it, because nothing was ever written for it to
   begin with. `renormalize_weights` below handles this shape directly: pass
   only the sources actually present as `present_sources`.
2. **The source exists overall but is missing for one particular day.**
   `scripts/run_tier1_regional_baseline.py`'s and
   `scripts/run_tier2_hierarchical_baseline.py`'s own alignment step
   (`xr.align(..., join="outer")`) already produces exactly this shape: a
   source reindexed onto the shared `sample` coordinate gets NaN, not a
   dropped coordinate, for any day it doesn't have -- the same convention
   those scripts already use to detect a missing *day* (`obs.isnull().all(
   dim=["latitude", "longitude"])`). `missing_for_sample` below applies that
   identical convention to a forecast source instead of `obs`.

Both shapes are real and both can occur independently (a source can be
present in the store yet missing one late day, or absent from the store
entirely) -- a caller building `present_sources` for a given day should
check both: source name is a known key AND `missing_for_sample` is False
for that day.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import xarray as xr

# Reasoned per the issue's own stated policy: a "blend" of exactly one
# surviving source isn't a blend at all, so it passes through unweighted
# (weight 1.0) with the fallback explicitly flagged -- mirrors
# weavr.weighting.MIN_TRAIN_SAMPLES's own logic (a degenerate case still
# needs a defined, visible answer, not a divide-by-one that happens to look
# like a real blend). Zero surviving sources has no defined blend at all and
# is a caller error (see `renormalize_weights`), not a fallback case.
MIN_SOURCES_FOR_BLEND = 2


@dataclass
class RenormalizedWeights:
    """The result of redistributing a fitted weight set over the sources
    actually present for one prediction day.

    `weights` covers only `present_sources` and always sums to 1.0 (up to
    floating point). `dropped_sources` names whichever of the original
    weight dict's keys were missing. `is_fallback`/`reason` follow every
    other combiner's own convention: set when only one source survived
    (see `MIN_SOURCES_FOR_BLEND`) or the present sources' original weights
    summed to zero, not merely because a source was dropped -- ordinary
    proportional renormalization with 2+ sources surviving is not a
    fallback, it's the intended path.
    """

    weights: dict[str, float]
    present_sources: tuple[str, ...]
    dropped_sources: tuple[str, ...]
    is_fallback: bool = False
    reason: str | None = None


def renormalize_weights(
    weights: dict[str, float], present_sources: Iterable[str]
) -> RenormalizedWeights:
    """Redistribute an already-fit flat `{source: weight}` dict over
    `present_sources`, proportionally to each present source's own already-
    fit weight -- e.g. weights `{a: 0.2, b: 0.3, c: 0.5}` with `c` missing
    renormalizes to `{a: 0.4, b: 0.6}` (0.2 and 0.3 rescaled to sum to 1).

    Serves `weavr.weighting`/`weavr.regime_weighting` directly: both
    already produce this exact flat-dict shape from
    `fit_region_weights`/`fit_regime_weights`, so a caller blending on a
    given day just filters `present_sources` (per this module's own
    docstring on detecting "missing") and calls this before
    `blend_with_region_weights`/`blend_with_regime_weights`.

    Falls back (see `RenormalizedWeights`) to passing the single surviving
    source through with weight 1.0 when only one of `weights`'s sources is
    present, or to an equal split across present sources when their
    original weights happen to sum to (approximately) zero -- both flagged
    via `is_fallback`/`reason`, the same honest-fallback convention every
    other combiner in this repo already uses, rather than silent
    degradation or a division by zero.

    Raises `ValueError` if none of `weights`'s sources are present --
    there is no source left to redistribute onto, a genuinely unhandled
    case the caller must not reach (e.g. by not invoking a combiner at all
    for a day where every one of its sources is missing).
    """
    original_sources = tuple(weights.keys())
    present = tuple(s for s in original_sources if s in set(present_sources))
    dropped = tuple(s for s in original_sources if s not in present)

    if not present:
        raise ValueError(
            "no source in `weights` is present -- nothing to renormalize onto "
            f"(all of {original_sources} are missing for this day)"
        )

    if len(present) < MIN_SOURCES_FOR_BLEND:
        (only_source,) = present
        return RenormalizedWeights(
            weights={only_source: 1.0},
            present_sources=present,
            dropped_sources=dropped,
            is_fallback=True,
            reason=(
                f"only {len(present)} of {len(original_sources)} source(s) present "
                f"({only_source}); passing it through with weight 1.0 rather than "
                "treating a single source as a genuine blend"
            ),
        )

    total = sum(weights[s] for s in present)
    if total <= 0.0:
        equal = 1.0 / len(present)
        return RenormalizedWeights(
            weights=dict.fromkeys(present, equal),
            present_sources=present,
            dropped_sources=dropped,
            is_fallback=True,
            reason=(
                f"present sources' original weights ({[weights[s] for s in present]}) "
                "summed to zero or less; falling back to an equal split across "
                "present sources"
            ),
        )

    return RenormalizedWeights(
        weights={s: weights[s] / total for s in present},
        present_sources=present,
        dropped_sources=dropped,
        is_fallback=False,
        reason=None,
    )


def missing_for_sample(
    forecast: xr.DataArray, spatial_dims: tuple[str, ...] = ("latitude", "longitude")
) -> xr.DataArray:
    """Per-sample boolean, True where `forecast` is entirely NaN across
    `spatial_dims` -- the real "present in the store but missing/late for
    this particular day" shape (see this module's docstring), matching
    `scripts/run_tier1_regional_baseline.py`'s own
    `obs.isnull().all(dim=["latitude", "longitude"])` convention exactly,
    applied to a forecast source instead of observations.

    Does not detect the other real shape (a source absent as a dict key
    entirely, e.g. its pull failed outright) -- that one has no array to
    inspect in the first place and is checked by the caller not finding the
    key, not by this function.
    """
    dims = [d for d in spatial_dims if d in forecast.dims]
    return forecast.isnull().all(dim=dims)


def present_sources_for_sample(
    forecasts: dict[str, xr.DataArray | None],
    sample,
    sample_dim: str = "sample",
    spatial_dims: tuple[str, ...] = ("latitude", "longitude"),
) -> tuple[str, ...]:
    """The sources in `forecasts` actually usable for one `sample` (day) --
    combines both real "missing" shapes this module's docstring documents:
    a source whose value is `None` (never pulled -- see the caller's own
    dict-assembly step) is excluded, and a source whose array is present
    but entirely NaN across `spatial_dims` for this particular `sample`
    (missing/late for this one day, per `missing_for_sample`) is excluded
    too. Order matches `forecasts`'s own iteration order.
    """
    present = []
    for name, da in forecasts.items():
        if da is None:
            continue
        if sample_dim in da.dims:
            da = da.sel({sample_dim: sample})
        if bool(missing_for_sample(da, spatial_dims=spatial_dims)):
            continue
        present.append(name)
    return tuple(present)
