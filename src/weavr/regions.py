"""Static regional pooling for Phase 3's skill weighting (Tier 1).

Assigns every gridpoint on weavr's locked 0.25 deg India grid
(`src/weavr/grid.py`) to one of Sreekala & Babu's six homogeneous summer
monsoon rainfall zones -- the scheme `docs/phase3-cv-and-regional-scheme.md`
decided on, not Neal et al.'s 30-pattern scheme, which turned out (checked
against the real citation, not assumed from the phase-plan's framing) to be
a *temporal* weather-regime classification, not a spatial partition at all.

Citation: Sreekala, P.P. & Babu, C.A., "Identification and Variability
Analysis of New Homogeneous Summer Monsoon Rainfall Regions Over India by
Using K-Means Clustering Technique," International Journal of Climatology
(Royal Meteorological Society / Wiley), 26 Jan 2025. The paper's six named
zones are West Coast India (WC), Southeast India (SI), West India (WI),
Central India (CI), and Northeast India 1 & 2 (NE1, NE2).

**This is a documented approximation, not the published k-means result.**
The paper's exact gridpoint-level cluster assignment isn't reproducible
from this environment (it needs their original clustering input dataset
and methodology, not just the six zone names) -- per
`docs/phase3-cv-and-regional-scheme.md`'s own decision, each named zone is
approximated here with simple, non-overlapping lat/lon bounds matching its
commonly-understood geographic identity. This is precise enough to pool
gridpoints for weight-fitting (Phase 3's actual need); it is not a claim
that these boundaries match the published clustering pixel-for-pixel.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from weavr.grid import COMMON_LAT, COMMON_LON

# The six Sreekala & Babu zones, in a fixed, documented order.
SREEKALA_BABU_ZONES = ("WC", "SI", "WI", "CI", "NE1", "NE2")

# Approximate boundaries (degrees), chosen to partition the whole common
# grid domain (lat 6.5-38.5, lon 66.5-100.0) into six regions matching each
# zone's broad geographic identity:
#   - West Coast India (WC): the coastal strip south of ~21N, west of ~76E
#     (Kerala, Karnataka coast, Goa, Konkan Maharashtra).
#   - West India (WI): north of ~21N, west of ~76E (Gujarat, Rajasthan,
#     interior/northern Maharashtra, western Madhya Pradesh).
#   - Southeast India (SI): south of ~21N, between ~76E and ~84E (Tamil
#     Nadu, Rayalaseema, coastal Andhra Pradesh).
#   - Central India (CI): north of ~21N, between ~76E and ~84E (Madhya
#     Pradesh, Chhattisgarh, interior Odisha/Maharashtra).
#   - Northeast India 1 (NE1): between ~84E and ~90E (West Bengal, Bihar,
#     Jharkhand, eastern Odisha).
#   - Northeast India 2 (NE2): east of ~90E (Assam and the northeastern
#     hill states).
_LAT_SOUTH_NORTH = 21.0
_LON_WEST_CENTRAL = 76.0
_LON_CENTRAL_EAST = 84.0
_LON_NE1_NE2 = 90.0


def assign_sreekala_babu_zone(
    lat: np.ndarray | float, lon: np.ndarray | float
) -> np.ndarray:
    """Vectorized zone assignment for arbitrary lat/lon arrays (broadcastable).

    The six conditions below exhaustively partition the entire plane (every
    combination of `lon < 76`/`>= 76`, further split by latitude or a finer
    longitude band) -- there is no "no match" case within
    `weavr.grid`'s domain, so every gridpoint gets exactly one label,
    checked directly against the real common grid in
    `tests/test_regions.py`, not assumed from the condition structure alone.
    """
    lat_arr = np.asarray(lat)
    lon_arr = np.asarray(lon)

    is_west = lon_arr < _LON_WEST_CENTRAL
    is_central_band = (lon_arr >= _LON_WEST_CENTRAL) & (lon_arr < _LON_CENTRAL_EAST)
    is_south = lat_arr < _LAT_SOUTH_NORTH

    conditions = [
        is_west & is_south,
        is_west & ~is_south,
        is_central_band & is_south,
        is_central_band & ~is_south,
        (lon_arr >= _LON_CENTRAL_EAST) & (lon_arr < _LON_NE1_NE2),
        (lon_arr >= _LON_NE1_NE2),
    ]
    choices = ["WC", "WI", "SI", "CI", "NE1", "NE2"]
    # default="" would only ever surface a real bug (a gap in the
    # conditions above) -- there is no legitimate unassigned gridpoint.
    return np.select(conditions, choices, default="")


def assign_regions(
    lat_values: np.ndarray = COMMON_LAT,
    lon_values: np.ndarray = COMMON_LON,
) -> xr.DataArray:
    """Region label per gridpoint on the given lat/lon grid (default: the
    real common India grid), as an `xr.DataArray` indexed by
    `(latitude, longitude)`.
    """
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    labels = assign_sreekala_babu_zone(lat_grid, lon_grid)
    return xr.DataArray(
        labels,
        coords={"latitude": lat_values, "longitude": lon_values},
        dims=["latitude", "longitude"],
        name="region",
    )
