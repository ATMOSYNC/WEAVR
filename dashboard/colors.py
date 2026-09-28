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
