import numpy as np
import pytest
import xarray as xr

from weavr.independence import (
    effective_number_of_models,
    error_correlation_matrix,
    participation_ratio,
)


def _error_field(values: np.ndarray) -> xr.DataArray:
    """Wrap a (sample, lat, lon) array in the shape the diagnostic expects."""
    n_sample, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        coords={
            "sample": np.arange(n_sample),
            "latitude": np.arange(n_lat, dtype=float),
            "longitude": np.arange(n_lon, dtype=float),
        },
        dims=["sample", "latitude", "longitude"],
    )


DIMS = ("sample", "latitude", "longitude")


class TestErrorCorrelationMatrix:
    def test_identical_errors_correlate_at_one(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(20, 5, 6))
        errors = {"a": _error_field(values), "b": _error_field(values.copy())}

        names, matrix = error_correlation_matrix(errors, DIMS)

        assert names == ["a", "b"]
        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == pytest.approx(1.0)

    def test_independent_errors_correlate_near_zero(self):
        rng = np.random.default_rng(1)
        errors = {
            "a": _error_field(rng.normal(size=(50, 10, 10))),
            "b": _error_field(rng.normal(size=(50, 10, 10))),
        }

        _, matrix = error_correlation_matrix(errors, DIMS)

        assert abs(matrix[0, 1]) < 0.05

    def test_recovers_a_known_two_source_correlation(self):
        # b = rho * a + sqrt(1 - rho^2) * independent noise has corr(a, b) =
        # rho exactly in expectation; with 5000 cells the sample estimate is
        # within a few thousandths.
        rho = 0.7
        rng = np.random.default_rng(2)
        a = rng.normal(size=(50, 10, 10))
        noise = rng.normal(size=(50, 10, 10))
        b = rho * a + np.sqrt(1 - rho**2) * noise

        _, matrix = error_correlation_matrix(
            {"a": _error_field(a), "b": _error_field(b)}, DIMS
        )

        assert matrix[0, 1] == pytest.approx(rho, abs=0.02)

    def test_uses_only_cells_finite_in_every_source(self):
        # Source b is NaN wherever a would have dragged the correlation:
        # those cells must be excluded from BOTH series, not just from b's.
        rng = np.random.default_rng(3)
        a = rng.normal(size=(20, 5, 5))
        b = a.copy()
        b[0] = np.nan
        a_with_garbage = a.copy()
        a_with_garbage[0] = 1e6  # only reachable if the mask is not shared

        _, matrix = error_correlation_matrix(
            {"a": _error_field(a_with_garbage), "b": _error_field(b)}, DIMS
        )

        assert matrix[0, 1] == pytest.approx(1.0)

    def test_rejects_a_single_source(self):
        with pytest.raises(ValueError, match="at least 2 sources"):
            error_correlation_matrix({"a": _error_field(np.zeros((2, 2, 2)))}, DIMS)

    def test_rejects_too_few_common_finite_cells(self):
        a = np.full((2, 2, 2), np.nan)
        a.flat[0] = 1.0
        b = a.copy()
        with pytest.raises(ValueError, match="finite in all"):
            error_correlation_matrix({"a": _error_field(a), "b": _error_field(b)}, DIMS)


class TestEffectiveNumberOfModels:
    def test_uncorrelated_sources_are_worth_all_of_them(self):
        assert effective_number_of_models(np.eye(3)) == pytest.approx(3.0)

    def test_identical_sources_are_worth_one(self):
        assert effective_number_of_models(np.ones((4, 4))) == pytest.approx(1.0)

    def test_matches_the_closed_form_for_a_known_correlation(self):
        rho = 0.8
        corr = np.array([[1.0, rho], [rho, 1.0]])
        assert effective_number_of_models(corr) == pytest.approx(2 / (1 + rho))

    def test_strong_anticorrelation_is_capped_at_n_rather_than_exploding(self):
        # 1 + (N-1)*mean_off_diag <= 0 here; the variance-reduction reading
        # of the formula no longer holds, so N is the honest ceiling.
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert effective_number_of_models(corr) == pytest.approx(2.0)

    def test_rejects_a_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            effective_number_of_models(np.ones((2, 3)))


class TestParticipationRatio:
    def test_uncorrelated_sources_give_n(self):
        assert participation_ratio(np.eye(5)) == pytest.approx(5.0)

    def test_identical_sources_give_one(self):
        assert participation_ratio(np.ones((3, 3))) == pytest.approx(1.0)

    def test_separates_shapes_that_the_mean_off_diagonal_cannot(self):
        # Two dependence structures with the SAME mean off-diagonal (0.3),
        # so effective_number_of_models cannot tell them apart at all:
        #   concentrated -- a and b nearly duplicate, c independent
        #   spread       -- all three mildly correlated
        concentrated = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
        spread = np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]])

        assert effective_number_of_models(concentrated) == pytest.approx(
            effective_number_of_models(spread)
        )
        # The participation ratio does distinguish them, and ranks the
        # concentrated case as the less diverse one. This is why both
        # summaries are reported rather than just the simpler formula.
        assert participation_ratio(concentrated) < participation_ratio(spread)
        assert participation_ratio(concentrated) == pytest.approx(9 / 4.62, rel=1e-6)
        assert participation_ratio(spread) == pytest.approx(9 / 3.54, rel=1e-6)

    def test_rejects_a_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            participation_ratio(np.ones((3, 2)))
