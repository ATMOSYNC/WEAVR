import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier2b_combined import (
    combine_predictive_quantiles,
    compute_brier_score,
    score_quantiles_crps,
)


class TestCombinePredictiveQuantiles:
    def test_combines_csgd_and_bma_samples(self):
        mean = np.array([2.0, 5.0])
        std = np.array([1.5, 2.5])
        shift = np.array([-0.5, -1.0])

        rng = np.random.default_rng(42)
        bma_samples = rng.uniform(0, 10, size=(2, 200))
        levels = np.array([0.25, 0.50, 0.75])

        q_csgd, q_bma, q_avg = combine_predictive_quantiles(
            mean, std, shift, bma_samples, levels=levels, weight=0.5
        )

        assert q_csgd.shape == (2, 3)
        assert q_bma.shape == (2, 3)
        assert q_avg.shape == (2, 3)
        assert np.all(q_avg >= 0.0)
        assert np.all(np.diff(q_avg, axis=-1) >= 0.0)


class TestComputeBrierScore:
    def test_perfect_forecast(self):
        prob = np.array([0.0, 1.0, 1.0])
        obs = np.array([2.0, 15.0, 20.0])
        # threshold 10.0 -> event is [0, 1, 1]
        assert compute_brier_score(prob, obs, threshold=10.0) == pytest.approx(0.0)

    def test_worst_forecast(self):
        prob = np.array([1.0, 0.0])
        obs = np.array([0.0, 50.0])
        # threshold 10.0 -> event is [0, 1]
        assert compute_brier_score(prob, obs, threshold=10.0) == pytest.approx(1.0)


class TestScoreQuantilesCrps:
    def test_computes_mean_crps(self):
        quantiles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        levels = np.array([0.25, 0.50, 0.75])
        obs = np.array([2.0, 5.0])

        score = score_quantiles_crps(quantiles, levels, obs)
        assert score >= 0.0
