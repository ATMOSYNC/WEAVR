import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.splits import (
    InsufficientTimeBlocksError,
    iter_evaluation_folds,
    leave_one_year_out,
    seasonal_block_split,
)


def _dataset_with_times(times) -> xr.Dataset:
    times = pd.DatetimeIndex(times)
    da = xr.DataArray(np.arange(len(times), dtype=float), coords={"time": times}, dims=["time"])
    return xr.Dataset({"value": da})


class TestLeaveOneYearOut:
    def test_every_timestamp_appears_in_exactly_one_test_fold(self):
        times = pd.date_range("2018-06-01", "2021-09-30", freq="17D")
        ds = _dataset_with_times(times)

        test_masks = [test_mask for _, test_mask in leave_one_year_out(ds)]
        n_years = len(np.unique(times.year))
        assert len(test_masks) == n_years

        coverage = np.zeros(len(times), dtype=int)
        for test_mask in test_masks:
            coverage += test_mask.astype(int)
        assert np.all(coverage == 1)

    def test_train_and_test_are_disjoint_and_complete_each_fold(self):
        times = pd.date_range("2018-01-01", "2020-12-31", freq="30D")
        ds = _dataset_with_times(times)

        for train_mask, test_mask in leave_one_year_out(ds):
            assert not np.any(train_mask & test_mask)
            assert np.all(train_mask | test_mask)
            assert test_mask.sum() > 0
            assert train_mask.sum() > 0

    def test_held_out_year_matches_test_mask(self):
        times = pd.date_range("2018-01-01", "2020-12-31", freq="30D")
        ds = _dataset_with_times(times)

        folds = list(leave_one_year_out(ds))
        held_out_years = [pd.DatetimeIndex(times[test]).year.unique().tolist() for _, test in folds]
        assert held_out_years == [[2018], [2019], [2020]]

    def test_single_year_input_raises(self):
        times = pd.date_range("2020-06-01", "2020-09-30", freq="D")
        ds = _dataset_with_times(times)

        with pytest.raises(InsufficientTimeBlocksError, match="at least 2 distinct years"):
            list(leave_one_year_out(ds))

    def test_accepts_plain_datetime_index_not_just_dataset(self):
        times = pd.date_range("2019-01-01", "2020-12-31", freq="60D")
        folds = list(leave_one_year_out(times))
        assert len(folds) == 2


class TestSeasonalBlockSplit:
    def test_trailing_split_boundary_does_not_interleave(self):
        times = pd.date_range("2020-06-01", "2020-09-30", freq="D")
        ds = _dataset_with_times(times)

        train_mask, test_mask = seasonal_block_split(ds, test_fraction=0.2)

        assert not np.any(train_mask & test_mask)
        assert np.all(train_mask | test_mask)
        train_times = pd.DatetimeIndex(times[train_mask])
        test_times = pd.DatetimeIndex(times[test_mask])
        # every train timestamp strictly precedes every test timestamp --
        # the boundary is a single clean cut, not interleaved chunks.
        assert train_times.max() < test_times.min()

    def test_leading_split_reverses_which_side_is_held_out(self):
        times = pd.date_range("2020-06-01", "2020-09-30", freq="D")
        ds = _dataset_with_times(times)

        train_mask, test_mask = seasonal_block_split(ds, test_fraction=0.2, position="leading")

        train_times = pd.DatetimeIndex(times[train_mask])
        test_times = pd.DatetimeIndex(times[test_mask])
        assert test_times.max() < train_times.min()

    def test_respects_requested_fraction_approximately(self):
        times = pd.date_range("2020-06-01", "2020-09-30", freq="D")
        ds = _dataset_with_times(times)

        _, test_mask = seasonal_block_split(ds, test_fraction=0.25)

        assert test_mask.sum() == pytest.approx(len(times) * 0.25, abs=1)

    def test_missing_days_do_not_shift_the_boundary(self):
        # A gap in the middle of the season (e.g. IMD's documented missing
        # days) -- the boundary is computed from timestamps actually
        # present, not an assumed contiguous calendar range.
        times = pd.date_range("2020-06-01", "2020-09-30", freq="D").delete([10, 11, 12])
        ds = _dataset_with_times(times)

        train_mask, test_mask = seasonal_block_split(ds, test_fraction=0.2)

        train_times = pd.DatetimeIndex(times[train_mask])
        test_times = pd.DatetimeIndex(times[test_mask])
        assert train_times.max() < test_times.min()
        assert len(train_times) + len(test_times) == len(times)

    def test_single_timestamp_raises(self):
        ds = _dataset_with_times(pd.date_range("2020-06-01", periods=1))

        with pytest.raises(InsufficientTimeBlocksError):
            seasonal_block_split(ds)

    def test_extreme_fraction_raises_rather_than_returning_empty_side(self):
        times = pd.date_range("2020-06-01", "2020-06-05", freq="D")  # 5 days
        ds = _dataset_with_times(times)

        with pytest.raises(InsufficientTimeBlocksError, match="empty train or test block"):
            seasonal_block_split(ds, test_fraction=0.01)

    def test_invalid_test_fraction_raises_value_error(self):
        ds = _dataset_with_times(pd.date_range("2020-06-01", periods=10))

        with pytest.raises(ValueError, match="test_fraction"):
            seasonal_block_split(ds, test_fraction=1.5)

    def test_invalid_position_raises_value_error(self):
        ds = _dataset_with_times(pd.date_range("2020-06-01", periods=10))

        with pytest.raises(ValueError, match="position"):
            seasonal_block_split(ds, position="sideways")  # type: ignore[arg-type]


class TestIterEvaluationFolds:
    def test_two_synthetic_years_yields_two_folds_with_year_labels(self):
        times = pd.date_range("2018-06-01", "2018-09-30", freq="10D").union(
            pd.date_range("2020-06-01", "2020-09-30", freq="10D")
        )
        ds = _dataset_with_times(times)

        folds = list(iter_evaluation_folds(ds))
        assert len(folds) == 2

        labels = [label for _, _, label in folds]
        assert labels == ["2018", "2020"]

        # Check train and test masks for fold 2018
        train_mask_18, test_mask_18, label_18 = folds[0]
        assert label_18 == "2018"
        assert not np.any(train_mask_18 & test_mask_18)
        assert np.all(train_mask_18 | test_mask_18)
        assert np.all(pd.DatetimeIndex(times[test_mask_18]).year == 2018)
        assert np.all(pd.DatetimeIndex(times[train_mask_18]).year == 2020)

        # Check train and test masks for fold 2020
        train_mask_20, test_mask_20, label_20 = folds[1]
        assert label_20 == "2020"
        assert np.all(pd.DatetimeIndex(times[test_mask_20]).year == 2020)
        assert np.all(pd.DatetimeIndex(times[train_mask_20]).year == 2018)

    def test_single_year_yields_one_labelled_block_split(self):
        times = pd.date_range("2020-06-01", "2020-09-30", freq="D")
        ds = _dataset_with_times(times)

        folds = list(iter_evaluation_folds(ds, test_fraction=0.25))
        assert len(folds) == 1

        train_mask, test_mask, label = folds[0]
        assert label == "seasonal_block_split"
        assert not np.any(train_mask & test_mask)
        assert np.all(train_mask | test_mask)
        assert test_mask.sum() == pytest.approx(len(times) * 0.25, abs=1)

