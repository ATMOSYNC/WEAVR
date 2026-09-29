"""Tests for the step 04 additions to `weavr.verify`.

Kept in their own file rather than appended to tests/test_verify.py so the
Phase 1 metric suite stays readable as the thing it is.
"""

import numpy as np
import pytest
import xarray as xr

from weavr import verify as V


def _field(values: np.ndarray, dims=("sample", "latitude", "longitude")) -> xr.DataArray:
    return xr.DataArray(values, dims=list(dims))


class TestSedi:
    def test_random_forecast_has_no_skill(self):
        rng = np.random.default_rng(0)
        obs = _field(rng.uniform(0, 100, size=(60, 10, 10)))
        noise = _field(rng.uniform(0, 100, size=(60, 10, 10)))

        assert abs(float(V.sedi(noise, obs, 64.5))) < 0.1

    def test_a_nearly_perfect_forecast_approaches_one(self):
        # Not exactly perfect: a perfect forecast has H=1 and F=0, where the
        # formula's logarithms are undefined (see the NaN test below).
        rng = np.random.default_rng(1)
        obs_values = rng.uniform(0, 100, size=(200, 10, 10))
        forecast_values = obs_values.copy()
        # Corrupt 2% of cells so H < 1 and F > 0.
        corrupt = rng.random(obs_values.shape) < 0.02
        forecast_values[corrupt] = rng.uniform(0, 100, size=corrupt.sum())

        result = float(V.sedi(_field(forecast_values), _field(obs_values), 64.5))

        assert result > 0.9

    def test_skilful_beats_random(self):
        rng = np.random.default_rng(2)
        obs_values = rng.uniform(0, 100, size=(200, 8, 8))
        skilful = _field(obs_values + rng.normal(0, 5, size=obs_values.shape))
        useless = _field(rng.uniform(0, 100, size=obs_values.shape))
        obs = _field(obs_values)

        assert float(V.sedi(skilful, obs, 64.5)) > float(V.sedi(useless, obs, 64.5))

    def test_returns_nan_when_the_forecast_never_predicts_the_event(self):
        # H = 0 and F = 0: ln(0) is undefined. This is the common real case
        # at 204.5 mm, where step 01 measured POD = 0 for every source.
        rng = np.random.default_rng(3)
        obs = _field(rng.uniform(0, 300, size=(40, 8, 8)))
        never = obs * 0.0

        assert np.isnan(float(V.sedi(never, obs, 204.5)))

    def test_returns_nan_for_a_perfect_forecast_rather_than_a_wrong_number(self):
        rng = np.random.default_rng(4)
        obs = _field(rng.uniform(0, 200, size=(40, 8, 8)))

        assert np.isnan(float(V.sedi(obs.copy(), obs, 64.5)))

    def test_uses_the_false_alarm_rate_not_the_false_alarm_ratio(self):
        # The two have different denominators. If SEDI accidentally used the
        # ratio (as reported in contingency_scores["far"]), the value would
        # differ; recompute SEDI by hand from the counts and compare.
        rng = np.random.default_rng(5)
        obs_values = rng.uniform(0, 100, size=(100, 8, 8))
        forecast_values = obs_values + rng.normal(0, 20, size=obs_values.shape)
        obs, forecast = _field(obs_values), _field(forecast_values)
        threshold = 64.5

        counts = V.contingency_counts(forecast, obs, thresholds=(threshold,))[threshold]
        hits = float(counts["hits"])
        misses = float(counts["misses"])
        false_alarms = float(counts["false_alarms"])
        correct_negatives = float(counts["correct_negatives"])

        h = hits / (hits + misses)
        f = false_alarms / (false_alarms + correct_negatives)  # RATE, not ratio
        expected = (np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h)) / (
            np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)
        )

        assert float(V.sedi(forecast, obs, threshold)) == pytest.approx(expected)


class TestContingencyCounts:
    def test_counts_sum_to_the_number_of_cells(self):
        rng = np.random.default_rng(6)
        obs = _field(rng.uniform(0, 100, size=(20, 5, 5)))
        forecast = _field(rng.uniform(0, 100, size=(20, 5, 5)))

        counts = V.contingency_counts(forecast, obs, thresholds=(64.5,))[64.5]

        assert sum(float(v) for v in counts.values()) == 20 * 5 * 5

    def test_counts_reproduce_the_reported_pod_and_csi(self):
        rng = np.random.default_rng(7)
        obs = _field(rng.uniform(0, 100, size=(30, 6, 6)))
        forecast = _field(rng.uniform(0, 100, size=(30, 6, 6)))

        counts = V.contingency_counts(forecast, obs, thresholds=(7.5,))[7.5]
        scores = V.contingency_scores(forecast, obs, thresholds=(7.5,))[7.5]
        hits, misses = float(counts["hits"]), float(counts["misses"])
        false_alarms = float(counts["false_alarms"])

        assert hits / (hits + misses) == pytest.approx(float(scores["pod"]))
        assert hits / (hits + misses + false_alarms) == pytest.approx(float(scores["csi"]))


class TestTwcrpsEnsemble:
    @staticmethod
    def _brute_force(members: np.ndarray, y: float, threshold: float) -> float:
        """twCRPS = integral of (F(z) - 1{z >= y})^2 * 1{z >= t} dz.

        Computed directly from the definition by dense numerical
        integration, with no reference to the chaining identity under test.
        """
        grid = np.linspace(threshold, max(members.max(), y) + 50.0, 400_000)
        cdf = (members[None, :] <= grid[:, None]).mean(axis=1)
        indicator = (grid >= y).astype(float)
        return float(np.trapezoid((cdf - indicator) ** 2, grid))

    @pytest.mark.parametrize(
        ("members", "y", "threshold"),
        [
            (np.array([10.0, 20.0, 30.0, 40.0, 50.0]), 35.0, 25.0),
            (np.array([0.0, 0.0, 5.0, 80.0, 120.0]), 90.0, 64.5),
            (np.array([70.0, 75.0, 80.0]), 10.0, 64.5),  # obs below threshold
            (np.array([1.0, 2.0, 3.0]), 100.0, 64.5),  # everything below
        ],
    )
    def test_matches_brute_force_numerical_integration(self, members, y, threshold):
        ensemble = xr.DataArray(members, dims=["member"])
        obs = xr.DataArray(y)

        result = float(V.twcrps_ensemble(ensemble, obs, threshold))
        expected = self._brute_force(members, y, threshold)

        assert result == pytest.approx(expected, abs=1e-3)

    def test_a_threshold_of_zero_reduces_to_plain_crps(self):
        rng = np.random.default_rng(8)
        ensemble = xr.DataArray(
            rng.uniform(0, 100, size=(20, 5)), dims=["sample", "member"]
        )
        obs = xr.DataArray(rng.uniform(0, 100, size=20), dims=["sample"])

        # With non-negative rainfall, max(x, 0) == x, so the weight is 1
        # everywhere and twCRPS must equal CRPS exactly.
        assert float(V.twcrps_ensemble(ensemble, obs, 0.0)) == pytest.approx(
            float(V.crps(ensemble, obs))
        )

    def test_ignores_differences_entirely_below_the_threshold(self):
        # Two forecasts that differ only in the light-rain range must score
        # identically at a heavy-rain threshold -- that is the whole point.
        obs = xr.DataArray(100.0)
        a = xr.DataArray(np.array([1.0, 2.0, 200.0]), dims=["member"])
        b = xr.DataArray(np.array([40.0, 50.0, 200.0]), dims=["member"])

        assert float(V.twcrps_ensemble(a, obs, 64.5)) == pytest.approx(
            float(V.twcrps_ensemble(b, obs, 64.5))
        )
        assert float(V.crps(a, obs)) != pytest.approx(float(V.crps(b, obs)))


class TestCrpsLargeEnsemble:
    """The sorted-ensemble identity must be exactly `crps`, not merely close."""

    @pytest.mark.parametrize("n_members", [2, 3, 17, 50, 200])
    def test_matches_crps_exactly_when_fully_reduced(self, n_members):
        rng = np.random.default_rng(n_members)
        ensemble = xr.DataArray(
            rng.uniform(0, 150, size=(12, 6, n_members)),
            dims=["sample", "cell", "member"],
        )
        obs = xr.DataArray(rng.uniform(0, 150, size=(12, 6)), dims=["sample", "cell"])

        assert float(V.crps_large_ensemble(ensemble, obs)) == pytest.approx(
            float(V.crps(ensemble, obs)), rel=1e-12
        )

    def test_matches_crps_exactly_when_reduced_over_a_subset_of_dims(self):
        rng = np.random.default_rng(0)
        ensemble = xr.DataArray(
            rng.uniform(0, 150, size=(8, 5, 4, 30)),
            dims=["sample", "latitude", "longitude", "member"],
        )
        obs = xr.DataArray(
            rng.uniform(0, 150, size=(8, 5, 4)), dims=["sample", "latitude", "longitude"]
        )

        expected = V.crps(ensemble, obs, dim=["latitude", "longitude"])
        actual = V.crps_large_ensemble(ensemble, obs, dim=["latitude", "longitude"])

        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12)

    def test_dim_none_reduces_over_everything_like_xskillscore_does(self):
        # xskillscore's dim=None means "reduce over all dims", NOT "return
        # the per-cell field". A mismatch here would make this function
        # silently disagree with crps by the spread of the field.
        rng = np.random.default_rng(1)
        ensemble = xr.DataArray(rng.uniform(0, 100, size=(6, 7, 20)), dims=["a", "b", "member"])
        obs = xr.DataArray(rng.uniform(0, 100, size=(6, 7)), dims=["a", "b"])

        assert V.crps_large_ensemble(ensemble, obs).shape == ()

    def test_a_perfect_deterministic_ensemble_scores_zero(self):
        obs = xr.DataArray(np.array([5.0, 20.0]), dims=["sample"])
        ensemble = xr.DataArray(
            np.tile(obs.values[:, None], (1, 10)), dims=["sample", "member"]
        )

        assert float(V.crps_large_ensemble(ensemble, obs)) == pytest.approx(0.0, abs=1e-12)

    def test_handles_the_climatology_sized_ensemble_without_exhausting_memory(self):
        # 434 members is what weavr.climatology produces from a 15-day
        # window over a 14-year archive. The pairwise route needs O(m^2)
        # per cell; this must stay linear.
        rng = np.random.default_rng(2)
        ensemble = xr.DataArray(
            rng.uniform(0, 100, size=(1, 40, 40, 434)),
            dims=["sample", "latitude", "longitude", "member"],
        )
        obs = xr.DataArray(
            rng.uniform(0, 100, size=(1, 40, 40)),
            dims=["sample", "latitude", "longitude"],
        )

        result = V.crps_large_ensemble(ensemble, obs, dim=["latitude", "longitude"])

        assert result.shape == (1,)
        assert np.isfinite(result.values).all()


class TestPitValues:
    def test_a_calibrated_ensemble_gives_uniform_pit(self):
        rng = np.random.default_rng(9)
        # Obs and members drawn from the same distribution -> calibrated.
        members = rng.normal(size=(4000, 20))
        obs = rng.normal(size=4000)
        ensemble = xr.DataArray(members, dims=["sample", "member"])

        pit = V.pit_values(ensemble, xr.DataArray(obs, dims=["sample"]))

        # A uniform sample has mean 0.5 and std 1/sqrt(12) ~ 0.289.
        assert float(pit.mean()) == pytest.approx(0.5, abs=0.02)
        assert float(pit.std()) == pytest.approx(1 / np.sqrt(12), abs=0.02)

    def test_under_dispersion_pushes_pit_to_the_edges(self):
        rng = np.random.default_rng(10)
        # Members far too tightly clustered: the obs keeps landing outside.
        members = rng.normal(scale=0.1, size=(3000, 20))
        obs = rng.normal(scale=1.0, size=3000)
        ensemble = xr.DataArray(members, dims=["sample", "member"])

        pit = V.pit_values(ensemble, xr.DataArray(obs, dims=["sample"])).values

        extreme_fraction = np.mean((pit < 0.05) | (pit > 0.95))
        assert extreme_fraction > 0.5  # U-shaped, far above the calibrated 0.1

    def test_randomization_spreads_ties_instead_of_piling_them_up(self):
        # An all-dry day: every member and the obs are 0. Deterministic
        # ranking would return one identical value for every cell; the
        # randomized PIT must spread them across (0, 1).
        ensemble = xr.DataArray(np.zeros((500, 10)), dims=["sample", "member"])
        obs = xr.DataArray(np.zeros(500), dims=["sample"])

        pit = V.pit_values(ensemble, obs)

        assert float(pit.min()) < 0.1
        assert float(pit.max()) > 0.9
        assert float(pit.mean()) == pytest.approx(0.5, abs=0.05)

    def test_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(11)
        ensemble = xr.DataArray(rng.normal(size=(50, 8)), dims=["sample", "member"])
        obs = xr.DataArray(rng.normal(size=50), dims=["sample"])

        first = V.pit_values(ensemble, obs, seed=3)
        second = V.pit_values(ensemble, obs, seed=3)

        xr.testing.assert_identical(first, second)


class TestPitValuesCsgd:
    def test_wet_observations_give_uniform_pit_when_the_fit_is_right(self):
        # Draw observations from the very CSGD being scored -> calibrated.
        from scipy.stats import gamma as gamma_dist

        mean, std, shift = 8.0, 6.0, -2.0
        shape, scale = mean**2 / std**2, std**2 / mean
        rng = np.random.default_rng(12)
        draws = gamma_dist.rvs(a=shape, scale=scale, size=20000, random_state=rng) + shift
        obs = np.maximum(draws, 0.0)

        pit = V.pit_values_csgd(
            xr.DataArray(np.full(obs.size, mean), dims=["x"]),
            xr.DataArray(np.full(obs.size, std), dims=["x"]),
            shift,
            xr.DataArray(obs, dims=["x"]),
        )

        assert float(pit.mean()) == pytest.approx(0.5, abs=0.02)
        assert float(pit.std()) == pytest.approx(1 / np.sqrt(12), abs=0.02)

    def test_the_point_mass_at_zero_is_randomized_not_piled_up(self):
        # Every observation is dry. Without randomization each would return
        # the same F(0), which would look like severe miscalibration.
        n = 2000
        pit = V.pit_values_csgd(
            xr.DataArray(np.full(n, 5.0), dims=["x"]),
            xr.DataArray(np.full(n, 4.0), dims=["x"]),
            -3.0,
            xr.DataArray(np.zeros(n), dims=["x"]),
        )

        assert float(pit.max()) - float(pit.min()) > 0.1
        assert len(np.unique(pit.values)) > n // 2

    def test_a_larger_observation_never_gets_a_smaller_pit(self):
        obs = xr.DataArray(np.array([0.5, 5.0, 50.0, 200.0]), dims=["x"])
        pit = V.pit_values_csgd(
            xr.DataArray(np.full(4, 8.0), dims=["x"]),
            xr.DataArray(np.full(4, 6.0), dims=["x"]),
            -2.0,
            obs,
        )

        assert (np.diff(pit.values) > 0).all()

    def test_degenerate_parameters_give_nan(self):
        pit = V.pit_values_csgd(
            xr.DataArray(np.array([0.0, 5.0]), dims=["x"]),
            xr.DataArray(np.array([2.0, 0.0]), dims=["x"]),
            -1.0,
            xr.DataArray(np.array([3.0, 3.0]), dims=["x"]),
        )

        assert np.isnan(pit.values).all()


class TestReliabilityTable:
    def test_a_calibrated_forecast_tracks_the_diagonal(self):
        rng = np.random.default_rng(13)
        prob = rng.random(20000)
        event = (rng.random(20000) < prob).astype(float)

        table = V.reliability_table(
            xr.DataArray(prob, dims=["x"]), xr.DataArray(event, dims=["x"])
        )

        occupied = table["count"].values > 0
        np.testing.assert_allclose(
            table["observed_frequency"].values[occupied],
            table["forecast_probability"].values[occupied],
            atol=0.03,
        )

    def test_an_over_forecasting_system_sits_below_the_diagonal(self):
        rng = np.random.default_rng(14)
        prob = rng.random(20000)
        event = (rng.random(20000) < prob * 0.5).astype(float)  # events half as often

        table = V.reliability_table(
            xr.DataArray(prob, dims=["x"]), xr.DataArray(event, dims=["x"])
        )

        high = table["forecast_probability"].values > 0.5
        assert (
            table["observed_frequency"].values[high]
            < table["forecast_probability"].values[high]
        ).all()

    def test_counts_sum_to_the_sample_size(self):
        rng = np.random.default_rng(15)
        prob = rng.random(500)
        event = (rng.random(500) < 0.5).astype(float)

        table = V.reliability_table(
            xr.DataArray(prob, dims=["x"]), xr.DataArray(event, dims=["x"])
        )

        assert table["count"].sum() == 500

    def test_probability_of_exactly_one_lands_in_the_top_bin(self):
        table = V.reliability_table(
            xr.DataArray(np.array([1.0, 1.0]), dims=["x"]),
            xr.DataArray(np.array([1.0, 0.0]), dims=["x"]),
            n_bins=10,
        )

        assert table["count"].values[-1] == 2
        assert table["observed_frequency"].values[-1] == pytest.approx(0.5)

    def test_empty_bins_are_nan_not_zero(self):
        # An unoccupied bin carries no evidence; 0.0 would claim it does.
        table = V.reliability_table(
            xr.DataArray(np.array([0.95, 0.96]), dims=["x"]),
            xr.DataArray(np.array([1.0, 1.0]), dims=["x"]),
            n_bins=10,
        )

        assert table["count"].values[0] == 0
        assert np.isnan(table["observed_frequency"].values[0])

    def test_rejects_zero_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            V.reliability_table(
                xr.DataArray([0.5], dims=["x"]), xr.DataArray([1.0], dims=["x"]), n_bins=0
            )


class TestSkillScores:
    def test_brier_skill_score_signs(self):
        assert float(V.brier_skill_score(0.1, 0.2)) == pytest.approx(0.5)
        assert float(V.brier_skill_score(0.2, 0.2)) == pytest.approx(0.0)
        assert float(V.brier_skill_score(0.4, 0.2)) == pytest.approx(-1.0)

    def test_crpss_signs(self):
        assert float(V.crpss(2.0, 4.0)) == pytest.approx(0.5)
        assert float(V.crpss(4.0, 4.0)) == pytest.approx(0.0)

    def test_a_zero_reference_gives_nan_not_infinity(self):
        assert np.isnan(float(V.brier_skill_score(0.1, 0.0)))
        assert np.isnan(float(V.crpss(1.0, 0.0)))
