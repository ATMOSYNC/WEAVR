import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.climatology import climatological_ensemble, climatological_probability

LAT = np.array([10.0, 11.0])
LON = np.array([75.0, 76.0])


def _synthetic_archive(years=(2018, 2019, 2020), value_per_year=None) -> xr.Dataset:
    """A JJAS archive where every cell's value encodes its year.

    That makes year exclusion directly observable: if a year leaks into the
    reference, its marker value shows up in the members.
    """
    frames = []
    times = []
    for year in years:
        season = pd.date_range(f"{year}-06-01", f"{year}-09-30", freq="D")
        marker = value_per_year[year] if value_per_year else float(year)
        frames.append(np.full((len(season), LAT.size, LON.size), marker))
        times.extend(season)
    return xr.Dataset(
        {"rain": (("time", "latitude", "longitude"), np.concatenate(frames))},
        coords={"time": pd.DatetimeIndex(times), "latitude": LAT, "longitude": LON},
    )


class TestClimatologicalEnsemble:
    def test_shape_and_member_count(self):
        archive = _synthetic_archive()
        targets = pd.date_range("2020-07-01", periods=3, freq="D")

        ensemble = climatological_ensemble(archive, targets, window_days=5)

        # 11 days of season x 3 years.
        assert ensemble.sizes == {
            "sample": 3,
            "member": 11 * 3,
            "latitude": 2,
            "longitude": 2,
        }

    def test_excluding_the_test_year_removes_it_from_the_members(self):
        # This is the leak that would invalidate every skill score.
        archive = _synthetic_archive()
        targets = pd.date_range("2020-07-15", periods=2, freq="D")

        ensemble = climatological_ensemble(
            archive, targets, exclude_years=2020, window_days=5
        )

        assert 2020.0 not in set(np.unique(ensemble.values))
        assert set(np.unique(ensemble.values)) == {2018.0, 2019.0}
        assert ensemble.sizes["member"] == 11 * 2

    def test_excluding_several_years_works(self):
        archive = _synthetic_archive()
        targets = pd.DatetimeIndex(["2020-07-15"])

        ensemble = climatological_ensemble(
            archive, targets, exclude_years=[2018, 2020], window_days=3
        )

        assert set(np.unique(ensemble.values)) == {2019.0}

    def test_member_count_is_constant_at_the_season_edges(self):
        # A symmetric window would truncate on 1 June and 30 September,
        # giving a ragged ensemble that is sharper mid-season than at the
        # edges. The window must slide inward instead.
        archive = _synthetic_archive()
        window = 10
        edges = pd.DatetimeIndex(["2020-06-01", "2020-07-20", "2020-09-30"])

        counts = {
            str(d.date()): climatological_ensemble(
                archive, pd.DatetimeIndex([d]), window_days=window
            ).sizes["member"]
            for d in edges
        }

        assert len(set(counts.values())) == 1, counts
        assert set(counts.values()) == {(2 * window + 1) * 3}

    def test_the_window_is_centred_away_from_the_edges(self):
        archive = _synthetic_archive(years=(2019,))
        target = pd.Timestamp("2019-07-20")

        ensemble = climatological_ensemble(
            archive, pd.DatetimeIndex([target]), window_days=5
        )

        assert ensemble.sizes["member"] == 11

    def test_a_season_shorter_than_the_window_returns_everything(self):
        archive = _synthetic_archive(years=(2019,))
        n_days = archive.sizes["time"]

        ensemble = climatological_ensemble(
            archive, pd.DatetimeIndex(["2019-07-01"]), window_days=400
        )

        assert ensemble.sizes["member"] == n_days

    def test_excluding_every_year_raises_rather_than_returning_empty(self):
        archive = _synthetic_archive(years=(2019, 2020))

        with pytest.raises(ValueError, match="removes every day"):
            climatological_ensemble(
                archive, pd.DatetimeIndex(["2020-07-01"]), exclude_years=[2019, 2020]
            )

    def test_accepts_a_dataarray_as_well_as_a_dataset(self):
        archive = _synthetic_archive()
        targets = pd.DatetimeIndex(["2020-07-01"])

        from_dataset = climatological_ensemble(archive, targets, window_days=3)
        from_dataarray = climatological_ensemble(archive["rain"], targets, window_days=3)

        np.testing.assert_array_equal(from_dataset.values, from_dataarray.values)


class TestClimatologicalProbability:
    def test_frequency_matches_the_fraction_of_exceeding_members(self):
        # Two years: one entirely at 100 mm, one entirely at 0 mm. The
        # climatological probability of >= 50 mm must be exactly 0.5.
        archive = _synthetic_archive(
            years=(2019, 2020), value_per_year={2019: 100.0, 2020: 0.0}
        )

        prob = climatological_probability(
            archive, pd.DatetimeIndex(["2020-07-01"]), threshold=50.0, window_days=5
        )

        assert float(prob.mean()) == pytest.approx(0.5)

    def test_excluding_a_year_changes_the_probability(self):
        archive = _synthetic_archive(
            years=(2019, 2020), value_per_year={2019: 100.0, 2020: 0.0}
        )
        targets = pd.DatetimeIndex(["2020-07-01"])

        with_2019_only = climatological_probability(
            archive, targets, threshold=50.0, exclude_years=2020, window_days=5
        )

        assert float(with_2019_only.mean()) == pytest.approx(1.0)

    def test_uses_greater_or_equal_matching_the_contingency_convention(self):
        archive = _synthetic_archive(years=(2019,), value_per_year={2019: 64.5})

        prob = climatological_probability(
            archive, pd.DatetimeIndex(["2019-07-01"]), threshold=64.5, window_days=2
        )

        assert float(prob.mean()) == pytest.approx(1.0)

    def test_probabilities_stay_within_zero_and_one(self):
        rng = np.random.default_rng(0)
        season = pd.date_range("2020-06-01", "2020-09-30", freq="D")
        archive = xr.Dataset(
            {
                "rain": (
                    ("time", "latitude", "longitude"),
                    rng.uniform(0, 200, size=(len(season), 2, 2)),
                )
            },
            coords={"time": season, "latitude": LAT, "longitude": LON},
        )

        prob = climatological_probability(
            archive, season[:10], threshold=64.5, window_days=7
        )

        assert float(prob.min()) >= 0.0
        assert float(prob.max()) <= 1.0
