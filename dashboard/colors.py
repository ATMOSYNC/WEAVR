"""IMD's real, cited rainfall colour scheme (`docs/phase7-dashboard-scope.md`,
section 3), shared between the blended-map (step 3) and extreme-probability
(step 4) views so both use the same real, sourced breakpoints rather than
each inventing its own palette.

Bin labels and thresholds are reused verbatim from `weavr.rain_bins` /
`weavr.verify` -- not re-derived here -- so this module can't silently
drift from this project's own established rain-intensity categories.
"""

from __future__ import annotations

import numpy as np

from weavr.rain_bins import RAIN_BIN_LABELS
from weavr.verify import IMD_RAIN_THRESHOLDS_MM

# Approximate hex renderings of IMD's real, published 24-hour
# rainfall-warning colours (green < 64.5mm, yellow 64.5-115.5mm, orange
# 115.6-204.4mm, red >= 204.5mm -- docs/phase7-dashboard-scope.md's own
# citation). "dry" (< 7.5mm, this project's own extra split below IMD's
# green band) renders as white/no-fill rather than green, so a genuinely
# dry cell isn't visually confused with light-but-still-raining cells.
RAIN_BIN_COLORS: dict[str, str] = {
    "dry": "#ffffff",
    "light": "#2ecc40",
    "heavy": "#ffdc00",
    "very_heavy": "#ff8c00",
    "extremely_heavy": "#ff0000",
}

if set(RAIN_BIN_COLORS) != set(RAIN_BIN_LABELS):
    raise AssertionError(
        "dashboard.colors.RAIN_BIN_COLORS has drifted from weavr.rain_bins.RAIN_BIN_LABELS"
    )


def rain_bin_index(
    values: np.ndarray, thresholds: tuple[float, ...] = IMD_RAIN_THRESHOLDS_MM
) -> np.ndarray:
    """Bin index (0..len(RAIN_BIN_LABELS)-1) for each value, reusing IMD's
    real thresholds verbatim.

    A NaN input is not expected in this dashboard's committed example
    exports (checked: 0% NaN in both `example_blend_grid.npz` and
    `example_probability_grid.npz`); if present it would land in the last
    bin index per `numpy.digitize`'s own behavior (the same caveat
    `weavr.rain_bins.classify_rain_bin`'s own docstring documents) --
    callers with possibly-missing data should mask before calling this.
    """
    return np.digitize(values, thresholds)


def build_discrete_colorscale(colors: list[str]) -> list[list[object]]:
    """A Plotly `color_continuous_scale`-compatible list of hard colour
    steps (not a smooth gradient) for `len(colors)` equal-width bins."""
    n = len(colors)
    scale: list[list[object]] = []
    for i, color in enumerate(colors):
        scale.append([i / n, color])
        scale.append([(i + 1) / n, color])
    return scale


# A probability (dimensionless, 0-1) has no real IMD-published breakpoint
# of its own -- only the mm-based rain-intensity categories above are
# IMD's real, cited scheme. Rather than inventing new, unstated probability
# breakpoints, this reuses the exact same 4 real IMD colour identities
# (white/green/yellow/orange/red) as a smooth gradient spread evenly across
# [0, 1], so the extreme-probability view stays visually consistent with
# the blended map's own colours without fabricating a second palette.
PROBABILITY_COLORSCALE: list[list[object]] = [
    [0.00, RAIN_BIN_COLORS["dry"]],
    [0.25, RAIN_BIN_COLORS["light"]],
    [0.50, RAIN_BIN_COLORS["heavy"]],
    [0.75, RAIN_BIN_COLORS["very_heavy"]],
    [1.00, RAIN_BIN_COLORS["extremely_heavy"]],
]


def fallback_overlay(is_fallback: np.ndarray) -> np.ndarray:
    """A same-shaped array of `1.0` where `is_fallback` is True and `NaN`
    everywhere else -- for layering a distinct, visible overlay on top of a
    probability heatmap (a second Plotly trace with NaN cells left
    transparent) rather than letting a fallback cell's real ~0 probability
    look identical to a genuinely low-risk, real fitted one.
    """
    return np.where(is_fallback, 1.0, np.nan)
