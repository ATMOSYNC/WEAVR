"""Rain-intensity-bin classification for Phase 4 (Tier 2).

Issue #6's first checkbox: "classify each day's ensemble-mean forecast
into a rain-intensity bin." This bins by the **forecast** value, not the
observation -- a forecast-driven regime label used to select *which
combiner* (EMOS-CSG/BMA, step 4/5) to apply, a genuinely different job from
`weavr.verify.contingency_scores`'s obs-vs-forecast categorical scoring at
the same thresholds. A day/gridpoint's bin is a property of what was
*predicted*, independent of whether that prediction turned out correct.

Bin edges reuse `weavr.verify.IMD_RAIN_THRESHOLDS_MM` (7.5/64.5/115.6/
204.5mm) verbatim -- the same fixed thresholds this project has used since
Phase 1 (`docs/phase-plan.md`), not a new ad-hoc set, so Phase 4's bins and
Tier 0/1's contingency tables always refer to the same real categories.
`docs/phase4-data-and-combiner-scope.md` already found the real per-bin
sample density at these edges (dry/light/moderate fittable at every lead,
heavy only at 24h/48h, extreme never).

Bin naming, checked against IMD's own published terminology rather than
invented: 64.5, 115.6, and 204.5mm are IMD's own official category
boundaries for **Heavy / Very heavy / Extremely heavy rain**, confirmed
against IMD's public rainfall-intensity terminology -- those three upper
bins use IMD's own names verbatim. The lowest threshold here (7.5mm) is
this project's own fixed value (`docs/phase-plan.md`), not one of IMD's
published category boundaries (IMD's own light/moderate split sits at
15.6mm) -- the two bins below 64.5mm are therefore named plainly ("dry",
"light") rather than claiming a precise IMD citation neither bin actually
has.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from weavr.verify import IMD_RAIN_THRESHOLDS_MM

# One more label than len(IMD_RAIN_THRESHOLDS_MM) -- the "extremely_heavy"
# bin is unbounded above. "heavy"/"very_heavy"/"extremely_heavy" are IMD's
# own published category names (64.5/115.6/204.5mm are IMD's real category
# boundaries); "dry"/"light" are plain, non-IMD-cited names for the two
# bins below 64.5mm, per this module's docstring.
RAIN_BIN_LABELS = ("dry", "light", "heavy", "very_heavy", "extremely_heavy")

# The label used for a cell that was NaN in the input (missing forecast
# data) -- never a real bin. `numpy.digitize` does NOT do this itself: it
# silently places NaN into the last bin (checked directly, not assumed --
# `np.digitize([np.nan], [7.5, 64.5, 115.6, 204.5])` returns `[4]`, the
# same index as a genuine extreme-rain forecast), which would silently
# mislabel missing data as "extremely_heavy" rather than excluding it.
MISSING_LABEL = ""


def classify_rain_bin(
    values: xr.DataArray,
    thresholds: tuple[float, ...] = IMD_RAIN_THRESHOLDS_MM,
) -> xr.DataArray:
    """Label each value with the rain-intensity bin its magnitude falls in.

    `values` is any DataArray of rainfall values (a forecast's, per this
    module's docstring -- callers scoring against observations should use
    `weavr.verify.contingency_scores` instead, a different job at the same
    thresholds). Bin edges are `thresholds` (left-closed: a value exactly
    equal to a threshold falls in the *higher* bin, matching
    `verify.contingency_scores`'s own `value >= threshold` event
    convention). Returns a DataArray of string labels from
    `RAIN_BIN_LABELS`, sharing `values`'s dims/coords exactly -- a pure
    value -> label transform, no dimension is added, reduced, or reordered.

    A NaN in `values` (missing forecast data) is labelled `MISSING_LABEL`
    (empty string), never a real bin -- `numpy.digitize` would otherwise
    silently place it in the last (extremely-heavy) bin, see this module's
    docstring.
    """
    if len(thresholds) != len(RAIN_BIN_LABELS) - 1:
        raise ValueError(
            f"thresholds must have {len(RAIN_BIN_LABELS) - 1} edges "
            f"(one fewer than {len(RAIN_BIN_LABELS)} labels), got {len(thresholds)}"
        )

    def _classify(arr: np.ndarray) -> np.ndarray:
        bin_index = np.digitize(arr, thresholds)
        labels = np.asarray(RAIN_BIN_LABELS)[bin_index]
        return np.where(np.isnan(arr), MISSING_LABEL, labels)

    return xr.apply_ufunc(_classify, values).rename("rain_bin")
