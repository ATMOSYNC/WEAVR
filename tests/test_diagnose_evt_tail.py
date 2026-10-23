from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_evt_tail import (  # noqa: E402
    diagnose_threshold,
    exceedances_by_region,
    moving_block_day_indices,
)


def test_block_indices_cover_the_season_length_in_contiguous_runs() -> None:
    idx = moving_block_day_indices(122, 7, np.random.default_rng(0))
    assert idx.shape == (122,)
    assert idx.min() >= 0 and idx.max() < 122
    # Within a block, days are consecutive.
    assert np.all(np.diff(idx[:7]) == 1)


def test_block_length_is_capped_at_the_season() -> None:
    idx = moving_block_day_indices(3, 7, np.random.default_rng(0))
    assert np.array_equal(np.sort(idx), np.arange(3))


def test_exceedances_are_grouped_by_region_and_strictly_above_u() -> None:
    rain = np.array([[[70.0, 64.5], [10.0, 90.0]]])
    regions = np.array([["WC", "WC"], ["SI", "SI"]])
    out = exceedances_by_region(rain, regions, 64.5)
    assert set(out) == {"WC", "SI"}
    assert np.array_equal(out["WC"], [70.0])
    assert np.array_equal(out["SI"], [90.0])


def test_recovers_a_known_shape_with_both_intervals() -> None:
    rng = np.random.default_rng(3)
    true_xi, scale, u = 0.1, 20.0, 64.5
    n_days, n_cells = 120, 60
    # Independent cells and days, every value a GPD exceedance above u.
    z = scale / true_xi * (rng.uniform(size=(n_days, 2, n_cells)) ** -true_xi - 1.0)
    rain = u + z
    regions = np.array([["WC"] * n_cells, ["CI"] * n_cells])

    row = diagnose_threshold(rain, regions, u, n_resamples=60, block_days=7, rng=rng)

    assert row["n_exceedances"] == n_days * 2 * n_cells
    assert row["xi_ci_lo_information"] < true_xi < row["xi_ci_hi_information"]
    assert row["xi_ci_lo_block_bootstrap"] < true_xi < row["xi_ci_hi_block_bootstrap"]
    # With no dependence to carry, the two intervals agree to within resampling noise.
    assert 0.6 < row["dependence_inflation"] < 1.6
    assert row["n_resamples_fitted"] == 60
