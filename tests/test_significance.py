import numpy as np
import pytest

from weavr.significance import (
    block_bootstrap_indices,
    bootstrap_is_degenerate,
    diebold_mariano,
    lag1_autocorrelation,
    paired_difference_ci,
)


class TestDegenerateBootstrap:
    """The block bootstrap must refuse to invent an interval it cannot have.

    A moving block of length L over n days has n - L + 1 start positions.
    When L >= n there is exactly one, so every replicate is the original
    series and the interval has ZERO width -- which would read as "excludes
    0" for any non-zero difference. On the pre-step-07 store (3-4 test days,
    7-day block) that made 57% of scorecard comparisons look significant.
    """

    @pytest.mark.parametrize(
        ("n_days", "block_days", "expected"),
        [(4, 7, True), (7, 7, True), (8, 7, False), (30, 7, False), (2, 2, True)],
    )
    def test_detects_when_there_is_only_one_possible_resample(
        self, n_days, block_days, expected
    ):
        assert bootstrap_is_degenerate(n_days, block_days) is expected

    def test_a_too_short_series_gives_nan_bounds_and_is_not_significant(self):
        # Four days, 7-day block, and a large consistent difference: the
        # naive implementation returned ci_lo == ci_hi == -10 and called it
        # significant.
        a = np.array([10.0, 11.0, 12.0, 13.0])
        b = np.array([20.0, 21.0, 22.0, 23.0])

        result = paired_difference_ci(a, b, block_days=7, n_resamples=1000)

        assert result.estimate == pytest.approx(-10.0)
        assert result.degenerate
        assert np.isnan(result.ci_lo) and np.isnan(result.ci_hi)
        assert not result.significant

    def test_the_same_data_with_enough_days_does_produce_an_interval(self):
        rng = np.random.default_rng(0)
        b = rng.uniform(20.0, 21.0, size=40)
        a = b - 10.0

        result = paired_difference_ci(a, b, block_days=7, n_resamples=500)

        assert not result.degenerate
        assert np.isfinite(result.ci_lo) and np.isfinite(result.ci_hi)
        assert result.significant

    def test_a_zero_width_interval_from_nan_bounds_is_never_significant(self):
        from weavr.significance import DifferenceCI

        nan = float("nan")
        assert not DifferenceCI(-5.0, nan, nan, 4, 4, 1000, degenerate=True).significant


class TestBlockBootstrapIndices:
    def test_shape_is_resamples_by_days(self):
        idx = block_bootstrap_indices(n_days=20, block_days=5, n_resamples=100, seed=0)
        assert idx.shape == (100, 20)

    def test_every_index_is_a_real_day(self):
        idx = block_bootstrap_indices(n_days=13, block_days=4, n_resamples=50, seed=1)
        assert idx.min() >= 0
        assert idx.max() <= 12

    def test_blocks_are_contiguous_in_time(self):
        # Within each block of `block_days`, consecutive indices must differ
        # by exactly 1 -- that contiguity is the whole point of a block
        # bootstrap, and an i.i.d. bootstrap would fail this.
        block_days = 5
        idx = block_bootstrap_indices(n_days=20, block_days=block_days, n_resamples=20, seed=2)
        for row in idx:
            for start in range(0, 20 - block_days, block_days):
                block = row[start : start + block_days]
                np.testing.assert_array_equal(np.diff(block), np.ones(block_days - 1))

    def test_blocks_never_wrap_past_the_end(self):
        # A wrapping implementation would join the last day to the first,
        # inventing a September-to-June transition. Every block must stay
        # inside the series, so no block may contain a decreasing step.
        block_days = 7
        idx = block_bootstrap_indices(n_days=30, block_days=block_days, n_resamples=200, seed=3)
        for row in idx:
            for start in range(0, 30 - block_days, block_days):
                block = row[start : start + block_days]
                assert (np.diff(block) > 0).all()

    def test_same_seed_gives_identical_indices(self):
        a = block_bootstrap_indices(20, 5, 50, seed=7)
        b = block_bootstrap_indices(20, 5, 50, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_indices(self):
        a = block_bootstrap_indices(20, 5, 50, seed=7)
        b = block_bootstrap_indices(20, 5, 50, seed=8)
        assert not np.array_equal(a, b)

    def test_block_longer_than_series_degenerates_to_the_whole_series(self):
        # Today's weekly store has 3-4 test days against a 7-day block.
        idx = block_bootstrap_indices(n_days=4, block_days=7, n_resamples=10, seed=0)
        assert idx.shape == (10, 4)
        for row in idx:
            np.testing.assert_array_equal(row, np.arange(4))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_days": 0}, "n_days"),
            ({"n_days": 10, "block_days": 0}, "block_days"),
            ({"n_days": 10, "n_resamples": 0}, "n_resamples"),
        ],
    )
    def test_rejects_degenerate_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            block_bootstrap_indices(**kwargs)


class TestPairedDifferenceCI:
    def test_identical_losses_give_a_zero_interval_containing_zero(self):
        rng = np.random.default_rng(0)
        loss = rng.uniform(1.0, 10.0, size=40)

        result = paired_difference_ci(loss, loss.copy(), block_days=7, n_resamples=500)

        assert result.estimate == pytest.approx(0.0)
        assert result.ci_lo <= 0.0 <= result.ci_hi
        assert not result.significant

    def test_a_consistent_winner_gives_an_interval_excluding_zero(self):
        # A is better than B on every single day, by a wide margin.
        rng = np.random.default_rng(1)
        loss_b = rng.uniform(5.0, 6.0, size=60)
        loss_a = loss_b - 2.0

        result = paired_difference_ci(loss_a, loss_b, block_days=7, n_resamples=500)

        assert result.estimate == pytest.approx(-2.0)
        assert result.ci_hi < 0.0
        assert result.significant

    def test_a_tiny_difference_in_noise_is_not_significant(self):
        rng = np.random.default_rng(2)
        loss_a = rng.normal(5.0, 3.0, size=40)
        loss_b = loss_a + rng.normal(0.01, 3.0, size=40)

        result = paired_difference_ci(loss_a, loss_b, block_days=7, n_resamples=500)

        assert not result.significant

    def test_pairing_is_what_makes_the_interval_narrow(self):
        # A and B share a large day-to-day signal and differ by a small
        # constant. Paired resampling cancels the shared part; if the two
        # series were resampled independently the interval would be far
        # wider and this difference would vanish into it.
        rng = np.random.default_rng(3)
        hard_days = rng.uniform(0.0, 50.0, size=60)
        loss_a = hard_days
        loss_b = hard_days + 0.5

        result = paired_difference_ci(loss_a, loss_b, block_days=7, n_resamples=500)

        assert result.estimate == pytest.approx(-0.5)
        assert result.ci_hi < 0.0
        assert (result.ci_hi - result.ci_lo) < 1.0

    def test_rmse_mode_takes_sqrt_of_mean_mse_not_mean_of_daily_rmse(self):
        # Daily MSEs of 1 and 9: RMSE of the period is sqrt(mean([1,9]))
        # = sqrt(5) = 2.236, NOT mean([1, 3]) = 2. Against a perfect
        # forecast the difference must be sqrt(5), which distinguishes the
        # two aggregations unambiguously.
        mse_a = np.array([1.0, 9.0] * 20)
        mse_b = np.zeros(40)

        result = paired_difference_ci(mse_a, mse_b, aggregate="rmse", n_resamples=200)

        assert result.estimate == pytest.approx(np.sqrt(5.0))
        assert result.estimate != pytest.approx(2.0)

    def test_same_seed_gives_identical_interval(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=30), rng.normal(size=30)

        first = paired_difference_ci(a, b, seed=11, n_resamples=200)
        second = paired_difference_ci(a, b, seed=11, n_resamples=200)

        assert first == second

    def test_drops_days_missing_in_either_series_keeping_the_pairing(self):
        a = np.array([1.0, 2.0, np.nan, 4.0])
        b = np.array([1.0, np.nan, 3.0, 4.0])

        result = paired_difference_ci(a, b, n_resamples=50)

        # Only days 0 and 3 are finite in both.
        assert result.n_days == 2
        assert result.estimate == pytest.approx(0.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            paired_difference_ci(np.zeros(5), np.zeros(6))

    def test_rejects_series_with_no_overlapping_finite_days(self):
        a = np.array([1.0, np.nan])
        b = np.array([np.nan, 2.0])
        with pytest.raises(ValueError, match="nothing to compare"):
            paired_difference_ci(a, b)

    def test_rejects_an_unknown_aggregate(self):
        with pytest.raises(ValueError, match="aggregate must be"):
            paired_difference_ci(np.ones(5), np.ones(5), aggregate="median")  # type: ignore[arg-type]


class TestDieboldMariano:
    def test_identical_losses_give_p_of_one(self):
        loss = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = diebold_mariano(loss, loss.copy())

        assert result.statistic == 0.0
        assert result.p_value == pytest.approx(1.0)

    def test_a_clear_consistent_winner_is_significant(self):
        rng = np.random.default_rng(5)
        loss_b = rng.normal(10.0, 1.0, size=200)
        loss_a = loss_b - 1.0

        result = diebold_mariano(loss_a, loss_b)

        assert result.statistic < 0.0
        assert result.p_value < 0.01

    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(6)
        loss_a = rng.normal(5.0, 1.0, size=200)
        loss_b = rng.normal(5.0, 1.0, size=200)

        result = diebold_mariano(loss_a, loss_b)

        assert result.p_value > 0.05

    def test_horizon_one_matches_the_plain_sample_variance(self):
        # At horizon=1 the HAC sum is empty, so the statistic must equal
        # mean(d) / sqrt(var(d, ddof=0) / n) exactly.
        rng = np.random.default_rng(7)
        a = rng.normal(size=50)
        b = rng.normal(size=50)
        d = a - b
        expected = d.mean() / np.sqrt(d.var(ddof=0) / d.size)

        assert diebold_mariano(a, b, horizon=1).statistic == pytest.approx(expected)

    def test_a_longer_horizon_widens_the_variance_for_correlated_differences(self):
        # Positively autocorrelated differences: accounting for that
        # dependence must make the test MORE conservative, not less.
        rng = np.random.default_rng(8)
        noise = rng.normal(size=300)
        d = np.convolve(noise, np.ones(10) / 10.0, mode="same") + 0.3
        a, b = d, np.zeros_like(d)

        h1 = diebold_mariano(a, b, horizon=1)
        h10 = diebold_mariano(a, b, horizon=10)

        assert abs(h10.statistic) < abs(h1.statistic)
        assert h10.p_value > h1.p_value

    def test_constant_nonzero_difference_is_maximally_significant(self):
        a = np.full(10, 1.0)
        b = np.zeros(10)

        result = diebold_mariano(a, b)

        assert result.statistic == np.inf
        assert result.p_value == 0.0

    def test_rejects_too_few_days(self):
        with pytest.raises(ValueError, match="at least 2"):
            diebold_mariano(np.array([1.0]), np.array([2.0]))

    def test_rejects_a_zero_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            diebold_mariano(np.zeros(5), np.ones(5), horizon=0)


class TestLag1Autocorrelation:
    def test_independent_noise_is_near_zero(self):
        rng = np.random.default_rng(9)
        assert abs(lag1_autocorrelation(rng.normal(size=5000))) < 0.05

    def test_a_strongly_persistent_series_is_near_one(self):
        # A slowly varying series, like rainfall through a monsoon spell.
        x = np.sin(np.linspace(0, 2 * np.pi, 500))
        assert lag1_autocorrelation(x) > 0.9

    def test_an_alternating_series_is_near_minus_one(self):
        x = np.array([1.0, -1.0] * 100)
        assert lag1_autocorrelation(x) == pytest.approx(-1.0, abs=0.02)

    def test_a_constant_series_is_nan_not_zero(self):
        # Zero would falsely suggest independence was measured.
        assert np.isnan(lag1_autocorrelation(np.full(10, 3.0)))

    def test_too_short_a_series_is_nan(self):
        assert np.isnan(lag1_autocorrelation(np.array([1.0, 2.0])))

    def test_ignores_missing_days(self):
        x = np.array([1.0, -1.0, np.nan, 1.0, -1.0, 1.0, -1.0])
        assert np.isfinite(lag1_autocorrelation(x))
