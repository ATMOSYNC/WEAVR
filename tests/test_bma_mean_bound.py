"""The BMA mean regressor must be bounded by its own training range.

One (bin, region) per LOYO fold diverged on the two-season daily base --
`heavy`xSI on 2018 (CRPS 145mm, bias +277mm) and `heavy`xNE1 on 2020 (CRPS
154mm, bias +197mm) -- while the other five regions in the same bin and lead
scored 17-40mm. Observed daily accumulations top out near 400mm, so the RMSE
of 667-748mm in those rows was not a forecast error.

The cause is the only unbounded quantity in the fit: `gamma_mean_intercept`
and `gamma_mean_slope` come from `np.linalg.lstsq` unconstrained, while `p0`
and the variance are both clipped. A near-degenerate `forecast_ct` inside one
bin-region training cell makes that regression ill-conditioned, and because
the density uses `mean_ct**3` as the gamma scale, the predictive mean blows
up.
"""

from __future__ import annotations

import numpy as np
import pytest

from weavr.bma import (
    _component_predictive_params,
    _fit_component,
)


def _degenerate_training_cell(n: int = 400) -> tuple[np.ndarray, ...]:
    """A bin-region training cell whose forecast mean barely varies.

    Nearly-constant `forecast_ct` is what makes the two-column least-squares
    design ill-conditioned; the wet observations are large so the mean
    regressor, not the variance regressor, is the one that runs away.
    """
    rng = np.random.default_rng(0)
    forecast_mean = 60.0 + rng.normal(0.0, 0.05, size=n)  # spread ~0.05 mm
    forecast_spread = 12.0 + rng.normal(0.0, 0.5, size=n)
    obs = 70.0 + rng.normal(0.0, 8.0, size=n)
    obs = np.clip(obs, 0.1, None)
    return forecast_mean, forecast_spread, obs


def test_component_caps_its_predicted_mean_at_the_training_maximum() -> None:
    forecast_mean, forecast_spread, obs = _degenerate_training_cell()
    component = _fit_component("ifs_ens", forecast_mean, forecast_spread, obs)

    assert component.max_mean_ct == pytest.approx(np.cbrt(obs).max())

    # A test forecast well outside the training range: previously this is
    # where mean_ct ran away and the predictive mean reached hundreds of mm.
    far_forecast = np.array([500.0])
    far_spread = np.array([20.0])
    _, mean_ct, _ = _component_predictive_params(
        component, far_forecast, far_spread
    )
    assert np.all(np.isfinite(mean_ct))
    assert mean_ct[0] <= component.max_mean_ct + 1e-9
    # And the implied mean is the cube of that, so still a plausible rainfall
    # rather than the 667-748mm RMSE the unbounded fit produced.
    assert mean_ct[0] ** 3 < 500.0


def test_cap_does_not_disturb_a_well_conditioned_fit() -> None:
    """A correctly conditioned regression already predicts inside the cap, so
    the bound must be inert -- otherwise this is a silent behaviour change."""
    rng = np.random.default_rng(1)
    n = 600
    truth = rng.gamma(shape=1.5, scale=25.0, size=n)
    forecast_mean = np.clip(truth + rng.normal(0, 8, n), 0.1, None)
    forecast_spread = 10.0 + rng.normal(0, 2, n)
    obs = np.clip(truth + rng.normal(0, 6, n), 0.1, None)

    component = _fit_component("graphcast", forecast_mean, forecast_spread, obs)
    inside = np.array([float(np.median(forecast_mean))])
    _, mean_ct, _ = _component_predictive_params(component, inside, np.array([12.0]))

    # Compare against the unclipped regression on the same point.
    raw = component.gamma_mean_intercept + component.gamma_mean_slope * np.cbrt(inside)
    assert mean_ct[0] == pytest.approx(float(raw[0]), rel=1e-9)


def test_cap_never_allows_a_negative_mean() -> None:
    forecast_mean, forecast_spread, obs = _degenerate_training_cell()
    component = _fit_component("ifs_ens", forecast_mean, forecast_spread, obs)
    # A forecast of zero is the extreme low end.
    _, mean_ct, _ = _component_predictive_params(
        component, np.array([0.0]), np.array([20.0])
    )
    assert mean_ct[0] > 0.0
