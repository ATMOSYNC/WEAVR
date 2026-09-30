import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_dashboard_example_grids import (  # noqa: E402
    latest_complete_sample_index,
)


def _source(n_samples: int, present: list[int], *, with_member: bool) -> xr.DataArray:
    """A source with finite data only on `present` sample indices, NaN elsewhere."""
    values = np.full((n_samples, 2, 2), np.nan)
    for index in present:
        values[index] = 1.0
    dims = ("sample", "latitude", "longitude")
    if with_member:
        values = np.stack([values, values], axis=1)
        dims = ("sample", "member", "latitude", "longitude")
    return xr.DataArray(values, dims=dims)


class TestLatestCompleteSampleIndex:
    def test_skips_trailing_samples_missing_from_the_sparser_source(self):
        # Mirrors the real stores: graphcast/hres are daily (122 samples) while
        # the weekly IFS-ENS store covers 18 of them. `sample=-1` picks a step
        # where IFS-ENS is entirely NaN.
        forecasts = {
            "graphcast": _source(10, list(range(10)), with_member=True),
            "ifs_ens": _source(10, [7, 8], with_member=True),
        }

        assert latest_complete_sample_index(forecasts) == 8

    def test_ignores_a_source_with_no_member_dimension(self):
        forecasts = {
            "graphcast": _source(4, [0, 1, 2, 3], with_member=True),
            "hres": _source(4, [0, 1, 2, 3], with_member=False),
        }

        assert latest_complete_sample_index(forecasts) == 3

    def test_returns_last_index_when_every_source_is_daily(self):
        forecasts = {
            "graphcast": _source(5, list(range(5)), with_member=True),
            "hres": _source(5, list(range(5)), with_member=False),
        }

        assert latest_complete_sample_index(forecasts) == 4

    def test_raises_instead_of_returning_a_partial_timestep(self):
        # No shared initialisation: a grid built here would look real but be
        # assembled from missing data, so this must fail loudly.
        forecasts = {
            "graphcast": _source(4, [0, 1], with_member=True),
            "ifs_ens": _source(4, [2, 3], with_member=True),
        }

        with pytest.raises(ValueError, match="no sample on which every forecast source"):
            latest_complete_sample_index(forecasts)

    def test_member_axis_of_all_nan_members_still_counts_as_present(self):
        # An IFS mean over members is what the blend consumes; a member axis
        # that reduces to a finite mean must count as data.
        values = np.full((3, 2, 2, 2), np.nan)
        values[2] = 0.5
        forecasts = {
            "ifs_ens": xr.DataArray(
                values, dims=("sample", "member", "latitude", "longitude")
            )
        }

        assert latest_complete_sample_index(forecasts) == 2
