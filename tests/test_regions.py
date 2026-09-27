import numpy as np
import pytest

from weavr.grid import COMMON_LAT, COMMON_LON
from weavr.regions import SREEKALA_BABU_ZONES, assign_regions, assign_sreekala_babu_zone


class TestFullCoverageOfRealGrid:
    def test_every_real_gridpoint_gets_a_non_empty_label(self):
        regions = assign_regions(COMMON_LAT, COMMON_LON)

        # default="" in assign_sreekala_babu_zone only surfaces if a real
        # gridpoint falls outside every condition -- a gap, not expected.
        assert not (regions.values == "").any()

    def test_every_label_is_one_of_the_six_named_zones(self):
        regions = assign_regions(COMMON_LAT, COMMON_LON)

        assert set(np.unique(regions.values)) <= set(SREEKALA_BABU_ZONES)

    def test_no_gridpoint_carries_more_than_one_label(self):
        # assign_sreekala_babu_zone returns a single scalar label per
        # gridpoint by construction (np.select picks the first matching
        # condition) -- confirm the output shape has exactly one label per
        # gridpoint, not e.g. an accidental extra dimension.
        regions = assign_regions(COMMON_LAT, COMMON_LON)

        assert regions.dims == ("latitude", "longitude")
        assert regions.shape == (len(COMMON_LAT), len(COMMON_LON))

    def test_exactly_six_regions_are_actually_used_on_the_real_grid(self):
        # Not just "at most six" (the type-level guarantee above) -- the
        # real India domain must actually reach all six zones, not leave
        # one unused because a threshold was placed outside the real grid's
        # range.
        regions = assign_regions(COMMON_LAT, COMMON_LON)

        assert set(np.unique(regions.values)) == set(SREEKALA_BABU_ZONES)


class TestKnownGeographySpotChecks:
    # Six cities, one per zone, each placed clearly away from a threshold
    # boundary so the assignment isn't sensitive to the approximation's
    # exact cutoff -- confirms the scheme actually distinguishes real,
    # well-known Indian regions, not just an internally-consistent partition.
    @pytest.mark.parametrize(
        "city,lat,lon,expected_zone",
        [
            ("Mangalore (Karnataka coast)", 12.9, 74.8, "WC"),
            ("Chennai (Tamil Nadu)", 13.1, 80.3, "SI"),
            ("Jaipur (Rajasthan)", 26.9, 75.8, "WI"),
            ("Nagpur (Maharashtra, interior)", 21.1, 79.1, "CI"),
            ("Kolkata (West Bengal)", 22.6, 88.4, "NE1"),
            ("Guwahati (Assam)", 26.2, 91.7, "NE2"),
        ],
    )
    def test_city_lands_in_expected_zone(self, city, lat, lon, expected_zone):
        assert assign_sreekala_babu_zone(lat, lon) == expected_zone

    def test_a_coastal_and_an_interior_point_land_in_different_zones(self):
        # Mangalore (west coast) vs. Nagpur (central interior) -- the same
        # rough latitude band, but on opposite sides of the coast/interior
        # distinction this scheme's lon_west_central threshold encodes.
        coastal = assign_sreekala_babu_zone(12.9, 74.8)
        interior = assign_sreekala_babu_zone(21.1, 79.1)
        assert coastal != interior
