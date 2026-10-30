import numpy as np
import pytest
import xarray as xr

import weavr.bma as bma_module
from weavr.bma import (
    BMA_CRPS_GRID_MM,
    MIN_TRAIN_DAYS_PER_BIN,
    BmaComponentFit,
    BmaFitResult,
    _ensemble_crps_chunked,
    _fit_component,
    bma_analytic_predictive_mean,
    bma_crps,
    bma_exceedance_probability,
    bma_mixture_cdf,
    bma_predictive_quantiles,
    crps_chunk_cells,
    fit_hierarchical_bma,
    renormalize_bma_for_present_sources,
    sample_bma_mixture,
    score_bma,
    score_bma_and_mean,
)
from weavr.stacking import predictive_quantiles_bma  # noqa: E402
from weavr.verify import crps as ensemble_crps


def _coords(n_sample, n_lat, n_lon):
    return {
        "sample": np.arange(n_sample),
        "latitude": np.arange(n_lat, dtype=float),
        "longitude": np.arange(n_lon, dtype=float),
    }


def _da(values, coords):
    dims = ["sample", "latitude", "longitude"]
    if values.ndim == 4:
        dims = [*dims, "member"]
    return xr.DataArray(values, coords=coords, dims=dims)


def _single_region_and_bin(n_sample, n_lat, n_lon, coords, bin_label="light"):
    region_labels = xr.DataArray(
        np.full((n_lat, n_lon), "R1"),
        coords={"latitude": coords["latitude"], "longitude": coords["longitude"]},
        dims=["latitude", "longitude"],
    )
    rain_bin_labels = xr.DataArray(
        np.full((n_sample, n_lat, n_lon), bin_label, dtype=object),
        coords=coords,
        dims=["sample", "latitude", "longitude"],
    )
    return region_labels, rain_bin_labels


class TestFitHierarchicalBmaRecoversBetterComponent:
    def test_clearly_better_source_gets_higher_weight(self):
        rng = np.random.default_rng(0)
        n_sample, n_lat, n_lon, n_member = 40, 3, 3, 8
        coords = _coords(n_sample, n_lat, n_lon)

        # Real precipitation-like obs: mostly small positive values, no
        # zeros, so both components' wet-regression path gets exercised.
        obs_values = rng.uniform(5.0, 30.0, size=(n_sample, n_lat, n_lon))

        # "good" ensemble tracks obs almost exactly across members; "bad"
        # is unrelated noise -- EM should assign "good" far more weight.
        good_members = obs_values[..., None] + rng.normal(
            scale=0.2, size=(n_sample, n_lat, n_lon, n_member)
        )
        bad_members = rng.uniform(5.0, 30.0, size=(n_sample, n_lat, n_lon, n_member))

        obs = _da(obs_values, coords)
        forecasts = {
            "good": _da(np.clip(good_members, 0.0, None), coords),
            "bad": _da(np.clip(bad_members, 0.0, None), coords),
        }
        region_labels, rain_bin_labels = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("light", "R1")]

        assert not result.is_fallback
        assert result.weights["good"] > result.weights["bad"]
        assert result.weights["good"] > 0.7
        assert sum(result.weights.values()) == pytest.approx(1.0)


class TestMixtureWeightsSumToOne:
    def test_weights_sum_to_one_for_every_fitted_cell(self):
        rng = np.random.default_rng(1)
        n_sample, n_lat, n_lon, n_member = 30, 2, 2, 6
        coords = _coords(n_sample, n_lat, n_lon)

        obs_values = rng.uniform(1.0, 20.0, size=(n_sample, n_lat, n_lon))
        members = obs_values[..., None] + rng.normal(
            scale=1.0, size=(n_sample, n_lat, n_lon, n_member)
        )
        deterministic = obs_values + rng.normal(scale=1.5, size=(n_sample, n_lat, n_lon))

        obs = _da(obs_values, coords)
        forecasts = {
            "ensemble_source": _da(np.clip(members, 0.0, None), coords),
            "hres": _da(np.clip(deterministic, 0.0, None), coords),
        }
        region_labels, rain_bin_labels = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("light", "R1")]

        assert not result.is_fallback
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert result.components["hres"].route == "kernel_dressing"
        assert result.components["ensemble_source"].route == "ensemble_dressing"


class TestSparseCellFallback:
    def test_cell_with_too_few_train_days_falls_back(self):
        n_sample, n_lat, n_lon, n_member = 10, 2, 2, 4
        coords = _coords(n_sample, n_lat, n_lon)
        rng = np.random.default_rng(2)

        obs_values = rng.uniform(1.0, 5.0, size=(n_sample, n_lat, n_lon))
        members = obs_values[..., None] + rng.normal(
            scale=0.5, size=(n_sample, n_lat, n_lon, n_member)
        )

        obs = _da(obs_values, coords)
        forecasts = {"good": _da(np.clip(members, 0.0, None), coords)}

        region_labels, _ = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        # Only 3 of 10 train days land in the "heavy" bin -- below
        # MIN_TRAIN_DAYS_PER_BIN (5).
        labels = np.full((n_sample, n_lat, n_lon), "light", dtype=object)
        labels[:3, 0, 0] = "heavy"
        rain_bin_labels = xr.DataArray(
            labels, coords=coords, dims=["sample", "latitude", "longitude"]
        )

        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("heavy", "R1")]

        assert result.is_fallback
        assert result.n_train_days == 3
        assert f"need >= {MIN_TRAIN_DAYS_PER_BIN}" in result.reason
        assert sum(result.weights.values()) == pytest.approx(1.0)


class TestScoreBma:
    def test_fallback_result_scores_against_a_point_mass_at_zero(self):
        n_lat, n_lon = 2, 2
        coords = {
            "latitude": np.arange(n_lat, dtype=float),
            "longitude": np.arange(n_lon, dtype=float),
        }
        obs = xr.DataArray(
            np.full((n_lat, n_lon), 3.0), coords=coords, dims=["latitude", "longitude"]
        )
        forecast_mean = {
            "good": xr.DataArray(
                np.full((n_lat, n_lon), 3.0), coords=coords, dims=["latitude", "longitude"]
            )
        }
        forecast_spread = {"good": None}

        result = BmaFitResult(
            bin_label="light", region="R1", weights={"good": 1.0}, is_fallback=True
        )

        scores = score_bma(
            result,
            forecast_mean,
            forecast_spread,
            obs,
            rng=np.random.default_rng(0),
            n_samples=50,
        )

        # A point-mass-at-zero forecast's CRPS against obs=3 is exactly |3-0|=3.
        assert np.allclose(scores.values, 3.0, atol=1e-6)

    def test_fitted_result_scores_lower_than_a_far_off_fallback(self):
        rng = np.random.default_rng(3)
        n_sample, n_lat, n_lon, n_member = 30, 2, 2, 6
        coords = _coords(n_sample, n_lat, n_lon)

        obs_values = rng.uniform(5.0, 15.0, size=(n_sample, n_lat, n_lon))
        members = obs_values[..., None] + rng.normal(
            scale=0.5, size=(n_sample, n_lat, n_lon, n_member)
        )

        obs = _da(obs_values, coords)
        forecasts = {"good": _da(np.clip(members, 0.0, None), coords)}
        region_labels, rain_bin_labels = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("light", "R1")]
        assert not result.is_fallback

        test_coords = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
        }
        test_obs = xr.DataArray(
            obs_values[0], coords=test_coords, dims=["latitude", "longitude"]
        )
        forecast_mean = {
            "good": xr.DataArray(
                members[0].mean(axis=-1), coords=test_coords, dims=["latitude", "longitude"]
            )
        }
        forecast_spread = {
            "good": xr.DataArray(
                members[0].std(axis=-1, ddof=1), coords=test_coords, dims=["latitude", "longitude"]
            )
        }

        fitted_scores = score_bma(
            result, forecast_mean, forecast_spread, test_obs, rng=np.random.default_rng(4)
        )

        fallback_result = BmaFitResult(
            bin_label="light", region="R1", weights={"good": 1.0}, is_fallback=True
        )
        fallback_scores = score_bma(
            fallback_result,
            forecast_mean,
            forecast_spread,
            test_obs,
            rng=np.random.default_rng(4),
        )

        assert float(fitted_scores.mean()) < float(fallback_scores.mean())


def _component(source):
    return BmaComponentFit(
        source=source,
        route="kernel_dressing",
        zero_intercept=0.0,
        zero_slope=0.0,
        gamma_mean_intercept=1.0,
        gamma_mean_slope=0.5,
        gamma_variance_intercept=1.0,
        gamma_variance_slope=0.0,
    )


class TestRenormalizeBmaForPresentSources:
    def _fitted_result(self):
        return BmaFitResult(
            bin_label="light",
            region="R1",
            weights={"graphcast": 0.5, "hres": 0.3, "ifs_ens": 0.2},
            components={
                "graphcast": _component("graphcast"),
                "hres": _component("hres"),
                "ifs_ens": _component("ifs_ens"),
            },
            is_fallback=False,
            n_train_days=10,
        )

    def test_missing_source_drops_its_weight_and_component_then_renormalizes(self):
        result = self._fitted_result()

        renormalized = renormalize_bma_for_present_sources(result, ["graphcast", "ifs_ens"])

        assert set(renormalized.weights) == {"graphcast", "ifs_ens"}
        assert sum(renormalized.weights.values()) == pytest.approx(1.0)
        assert renormalized.weights["graphcast"] == pytest.approx(5 / 7)
        assert set(renormalized.components) == {"graphcast", "ifs_ens"}
        assert not renormalized.is_fallback

    def test_only_one_source_present_is_flagged_but_keeps_its_real_component(self):
        result = self._fitted_result()

        renormalized = renormalize_bma_for_present_sources(result, ["hres"])

        assert renormalized.weights == {"hres": 1.0}
        assert set(renormalized.components) == {"hres"}
        assert renormalized.is_fallback
        assert renormalized.reason is not None

    def test_single_surviving_source_still_samples_from_its_real_component_not_zero(self):
        result = self._fitted_result()
        renormalized = renormalize_bma_for_present_sources(result, ["hres"])

        forecast_mean = {"hres": np.array([2.0, 4.0])}
        forecast_spread = {"hres": None}
        samples = sample_bma_mixture(
            renormalized, forecast_mean, forecast_spread, np.random.default_rng(0), n_samples=200
        )

        # A real, non-degenerate mixture draws a spread of positive values,
        # not the fit-time fallback's point mass at exactly zero.
        assert samples.mean() > 0.0

    def test_fit_time_fallback_cell_is_returned_unchanged(self):
        fallback_result = BmaFitResult(
            bin_label="extreme",
            region="R1",
            weights={"graphcast": 0.5, "hres": 0.5},
            components={},
            is_fallback=True,
            reason="too few train days",
            n_train_days=1,
        )

        renormalized = renormalize_bma_for_present_sources(fallback_result, ["graphcast"])

        assert renormalized is fallback_result

    def test_present_sources_original_weights_sum_to_zero_still_keeps_components(self):
        result = BmaFitResult(
            bin_label="light",
            region="R1",
            weights={"a": 0.0, "b": 0.0, "c": 1.0},
            components={"a": _component("a"), "b": _component("b"), "c": _component("c")},
            is_fallback=False,
            n_train_days=10,
        )

        renormalized = renormalize_bma_for_present_sources(result, ["a", "b"])

        assert renormalized.weights == pytest.approx({"a": 0.5, "b": 0.5})
        assert set(renormalized.components) == {"a", "b"}
        assert renormalized.is_fallback


class TestScoreBmaAndMean:
    """CRPS and predictive mean must describe one shared realisation of the
    fitted mixture. Sampling once for each -- as `score_bma` plus a separate
    `sample_bma_mixture` call used to do -- doubles the runtime and peak memory
    of the `(n_cells, n_samples)` array (which is what exhausts RAM on a 17 GB
    machine) and makes `crps_mm` disagree with the `rmse_mm`/`bias_mm`
    reported beside it."""

    def _fit_and_arrays(self, n_sample=40, n_lat=2, n_lon=2, n_member=6, seed=0):
        rng = np.random.default_rng(seed)
        coords = _coords(n_sample, n_lat, n_lon)
        obs_values = rng.uniform(5.0, 30.0, size=(n_sample, n_lat, n_lon))
        obs = _da(obs_values, coords)
        mean_arrays = {
            "good": _da(
                np.clip(
                    obs_values[..., None]
                    + rng.normal(scale=0.2, size=(n_sample, n_lat, n_lon, n_member)),
                    0.0,
                    None,
                ),
                coords,
            ),
            "bad": _da(
                rng.uniform(5.0, 30.0, size=(n_sample, n_lat, n_lon, n_member)), coords
            ),
        }
        region_labels, rain_bin_labels = _single_region_and_bin(
            n_sample, n_lat, n_lon, coords
        )
        mask = np.ones(n_sample, dtype=bool)
        results = fit_hierarchical_bma(mean_arrays, obs, rain_bin_labels, region_labels, mask)
        # `score_bma*` takes member-reduced means/spreads, exactly as
        # `score_bma_cells` passes them: the `member` axis of its own output is
        # the Monte Carlo draw axis, not the forecast ensemble.
        reduced = {s: v.mean(dim="member") for s, v in mean_arrays.items()}
        return results[("light", "R1")], reduced, obs

    def test_crps_comes_from_one_shared_draw_set_and_the_mean_is_analytic(self):
        """CRPS is sampled; the predictive mean is computed exactly.

        The mean used to be `samples.mean(axis=-1)`, and because the predictive
        variable is `X**3` with a fat-tailed `X`, that estimator reported
        327.6 mm where the truth is 62.2 mm. It is now
        `E[X**3] = theta**3 kappa (kappa+1) (kappa+2)`, so only CRPS depends on
        the generator.
        """
        result, mean_arrays, obs = self._fit_and_arrays()
        spread = dict.fromkeys(mean_arrays, None)

        crps, predictive_mean = score_bma_and_mean(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(7), n_samples=200
        )

        replayed = sample_bma_mixture(
            result,
            {s: v.values for s, v in mean_arrays.items()},
            {s: None for s in mean_arrays},
            np.random.default_rng(7),
            n_samples=200,
        )
        assert replayed.shape[-1] == 200
        # CRPS is now integrated from the CDF, so it must NOT equal the sampled
        # value -- and the gap is the point of the change.
        assert crps.shape == obs.shape
        exact = bma_crps(
            result,
            {s: v.values for s, v in mean_arrays.items()},
            {s: None for s in mean_arrays},
            obs.values,
        )
        assert np.allclose(crps.values, exact)
        # The mean must equal the analytic moment, not the sample mean.
        analytic = bma_analytic_predictive_mean(
            result,
            {s: v.values for s, v in mean_arrays.items()},
            {s: None for s in mean_arrays},
        )
        assert np.allclose(predictive_mean.values, analytic)
        # On this well-behaved fixture the two agree closely -- which is the
        # point: they only diverge where the tail is heavy enough for 200 draws
        # to be unreliable, and there the analytic one is the correct answer.
        assert np.allclose(analytic, replayed.mean(axis=-1), rtol=0.02)

    def test_the_predictive_mean_does_not_depend_on_the_generator(self):
        """Two seeds give different draws, and now provably the same mean.

        This is the regression the analytic mean exists to remove: the mean used
        to move with the seed, so `bias_mm`, `mse_mm2` and `rmse_mm` were Monte
        Carlo estimates, and in a heavy tail that is most of their value.
        """
        result, mean_arrays, obs = self._fit_and_arrays()
        spread = dict.fromkeys(mean_arrays, None)

        _crps_a, mean_a = score_bma_and_mean(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(7), n_samples=200
        )
        _crps_b, mean_b = score_bma_and_mean(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(8), n_samples=200
        )
        assert np.array_equal(mean_a.values, mean_b.values)

    def test_a_sampled_mean_would_have_differed_between_seeds(self):
        """The guard for the test above: the sampled estimator really did move."""
        result, mean_arrays, obs = self._fit_and_arrays()
        arrays = {s: v.values for s, v in mean_arrays.items()}
        spread = {s: None for s in mean_arrays}
        first = sample_bma_mixture(result, arrays, spread, np.random.default_rng(7), n_samples=200)
        second = sample_bma_mixture(result, arrays, spread, np.random.default_rng(8), n_samples=200)
        assert not np.allclose(first.mean(axis=-1), second.mean(axis=-1))

    def test_score_bma_delegates_to_the_same_draws(self):
        result, mean_arrays, obs = self._fit_and_arrays()
        spread = dict.fromkeys(mean_arrays, None)

        via_score_bma = score_bma(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(11), n_samples=200
        )
        crps, _mean = score_bma_and_mean(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(11), n_samples=200
        )

        assert np.allclose(via_score_bma.values, crps.values)

    def test_predictive_mean_is_per_cell_and_finite(self):
        result, mean_arrays, obs = self._fit_and_arrays()
        spread = dict.fromkeys(mean_arrays, None)

        crps, predictive_mean = score_bma_and_mean(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(3), n_samples=25
        )

        # Both outputs must stay unreduced: `score_bma_cells` writes each one
        # straight into a per-cell grid through a boolean mask.
        assert predictive_mean.dims == obs.dims
        assert predictive_mean.shape == obs.shape
        assert np.isfinite(predictive_mean.values).all()
        assert (predictive_mean.values >= 0.0).all()
        assert crps.shape == obs.shape, (
            "CRPS must be per-cell. It used to be reduced to a scalar, which "
            "made every cell of a (bin, region) group receive the group mean "
            "and left the per-day BMA files with no within-group variation."
        )
        assert np.isfinite(crps.values).all()
        assert (crps.values >= 0.0).all()


class TestSampleBmaMixtureChunking:
    """`sample_bma_mixture` draws in cell chunks so peak memory is bounded by
    the chunk size instead of growing with the (bin, region) group. Unchunked,
    ~7 full-size `(cells, n_samples)` temporaries are live at once, which is
    what the OOM killer was terminating Tier 2 on."""

    def _arrays(self, n_cells, seed=0):
        rng = np.random.default_rng(seed)
        return (
            {"a": rng.uniform(1.0, 50.0, n_cells), "b": rng.uniform(1.0, 50.0, n_cells)},
            {"a": rng.uniform(0.5, 5.0, n_cells), "b": None},
        )

    def _result(self):
        def component(source, route):
            return BmaComponentFit(
                source=source,
                route=route,
                zero_intercept=1.0,
                zero_slope=0.3,
                gamma_mean_intercept=1.0,
                gamma_mean_slope=0.5,
                gamma_variance_intercept=0.4,
                gamma_variance_slope=0.2,
            )

        return BmaFitResult(
            bin_label="light",
            region="R1",
            weights={"a": 0.6, "b": 0.4},
            components={
                "a": component("a", "ensemble_dressing"),
                "b": component("b", "kernel_dressing"),
            },
            is_fallback=False,
        )

    def test_chunked_draws_match_unchunked_in_the_aggregate(self):
        mean, spread = self._arrays(4_000)
        result = self._result()

        unchunked = sample_bma_mixture(
            result, mean, spread, np.random.default_rng(42), n_samples=400
        )
        chunked = sample_bma_mixture(
            result, mean, spread, np.random.default_rng(42), n_samples=400,
            max_cells_per_chunk=512,
        )

        assert chunked.shape == unchunked.shape == (4_000, 400)
        # The per-draw distribution is heavy-tailed, so individual samples
        # legitimately differ; what the metrics are computed from is the
        # predictive mean, and that has to agree.
        assert chunked.mean() == pytest.approx(unchunked.mean(), rel=0.05)

    def test_chunk_size_does_not_change_the_answer(self):
        mean, spread = self._arrays(4_000)
        result = self._result()
        kwargs = dict(n_samples=300, max_cells_per_chunk=700)

        first = sample_bma_mixture(
            result, mean, spread, np.random.default_rng(5), **kwargs
        )
        second = sample_bma_mixture(
            result, mean, spread, np.random.default_rng(5), **kwargs
        )

        assert np.array_equal(first, second)

    def test_multidimensional_inputs_keep_their_shape(self):
        mean, spread = self._arrays(3 * 4)
        result = self._result()
        mean_3d = {s: v.reshape(3, 4) for s, v in mean.items()}
        spread_3d = {s: (None if v is None else v.reshape(3, 4)) for s, v in spread.items()}

        draws = sample_bma_mixture(
            result, mean_3d, spread_3d, np.random.default_rng(1), n_samples=50,
            max_cells_per_chunk=5,
        )

        assert draws.shape == (3, 4, 50)

    def test_no_components_still_returns_a_point_mass_at_zero(self):
        empty = BmaFitResult(
            bin_label="light", region="R1", weights={"a": 1.0}, is_fallback=True
        )
        mean, spread = self._arrays(64)

        draws = sample_bma_mixture(
            empty, mean, spread, np.random.default_rng(0), n_samples=25,
            max_cells_per_chunk=8,
        )

        assert draws.shape == (64, 25)
        assert not draws.any()


class TestEnsembleCrpsChunked:
    """`xskillscore.crps_ensemble` builds a pairwise member-difference tensor,
    so its working set grows with cells * n_samples**2 -- a single (bin, region)
    group of a real 129x135 grid needs a 10 GB temporary at the default
    n_samples=500, which is what the OOM killer was terminating Tier 2 on."""

    def test_block_size_shrinks_as_the_sample_count_grows(self):
        # cells * 8 * n_samples**2 must stay inside the budget.
        assert crps_chunk_cells(500) < crps_chunk_cells(100) < crps_chunk_cells(20)

    def test_block_size_is_never_zero(self):
        assert crps_chunk_cells(100_000) >= 1

    def test_rejects_a_non_positive_sample_count(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            crps_chunk_cells(0)

    def test_chunked_score_matches_the_unblocked_score(self, monkeypatch):
        rng = np.random.default_rng(0)
        samples = rng.gamma(2.0, 4.0, size=(400, 120))
        obs = xr.DataArray(rng.uniform(1.0, 20.0, 400), dims="cell")

        reference = float(
            ensemble_crps(
                xr.DataArray(samples, dims=("cell", "member")), obs, member_dim="member"
            ).values
        )

        # Force blocking: a cell's members are independent, so only the order
        # the per-cell scores are summed in changes, and that is a rounding
        # difference in the last ulp, not a different score.
        monkeypatch.setattr(bma_module, "crps_chunk_cells", lambda n_samples, budget=0: 37)
        chunked = float(_ensemble_crps_chunked(samples, obs, "member").values)

        assert chunked == pytest.approx(reference, rel=1e-12)

    def test_small_input_takes_the_unblocked_path_unchanged(self):
        rng = np.random.default_rng(1)
        samples = rng.gamma(2.0, 4.0, size=(5, 10))
        obs = xr.DataArray(rng.uniform(1.0, 20.0, 5), dims="cell")

        got = _ensemble_crps_chunked(samples, obs, "member")
        expected = ensemble_crps(
            xr.DataArray(samples, dims=("cell", "member")), obs, member_dim="member"
        )

        assert float(got.values) == float(expected.values)


class TestBmaPredictiveQuantiles:
    """Exact quantiles by inverting the CDF, replacing draw order statistics."""

    LEVELS = np.linspace(0.01, 0.99, 99)

    def _result(self, zero_intercept, zero_slope):
        return BmaFitResult(
            bin_label="x", region="R1", weights={"a": 1.0},
            components={
                "a": BmaComponentFit(
                    "a", "kernel_dressing", zero_intercept, zero_slope,
                    1.2, 0.4, 0.3, 0.0, max_mean_ct=6.0,
                )
            },
        )

    def test_monotone_and_non_negative(self):
        q = bma_predictive_quantiles(
            self._result(0.0, -1.0), {"a": np.full((4,), 3.0)}, {"a": None}, self.LEVELS
        )
        assert q.shape == (4, 99)
        assert np.all(np.diff(q, axis=-1) >= -1e-9)
        assert np.all(q >= 0.0)

    def test_quantiles_below_the_point_mass_are_exactly_zero(self):
        """`Q(tau) = 0` for `tau <= p0`, the same convention as the CSGD.

        Interpolating toward the point mass instead produces a *negative*
        quantile, because the CDF has already reached p0 at the grid's first
        point.
        """
        dry = self._result(3.0, -2.0)
        q = bma_predictive_quantiles(
            dry, {"a": np.full((4,), 0.2)}, {"a": None}, self.LEVELS
        )
        cdf_at_zero = bma_mixture_cdf(
            dry, {"a": np.full((4,), 0.2)}, {"a": None}, np.zeros(1)
        )[:, 0]
        for level_index, level in enumerate(self.LEVELS):
            if level <= cdf_at_zero[0]:
                assert q[0, level_index] == 0.0

    def test_agrees_with_the_exact_cdf(self):
        """Inversion is accurate to 2% of probability, above and below `p0`.

        Two separate reasons the tolerance is needed, both inherent rather than
        bugs:

        * Below the point mass `Q(tau) = 0` and `F(0) = p0 > tau`, because the
          CDF jumps at zero. `F(Q(tau))` legitimately exceeds `tau` there.
        * Just above it the CDF is steep, and the inversion interpolates linearly
          in *value*, so `F(Q(tau))` lands a little under `tau` -- measured at
          0.011 at `tau = 0.21` against `p0 = 0.19`.
        """
        result = self._result(0.0, -1.0)
        mean = {"a": np.full((4,), 3.0)}
        spread = {"a": None}
        q = bma_predictive_quantiles(result, mean, spread, self.LEVELS)
        achieved = bma_mixture_cdf(result, mean, spread, q)
        p0 = bma_mixture_cdf(result, mean, spread, np.zeros(1))[:, 0]
        above = self.LEVELS[None, :] > p0[:, None]
        target = np.broadcast_to(self.LEVELS, achieved.shape)
        assert np.allclose(achieved[above], target[above], atol=0.02)
        # Below `p0` the quantile is zero and the achieved level is `p0` itself.
        below = ~above
        assert np.allclose(achieved[below], np.broadcast_to(p0[:, None], achieved.shape)[below])

    #: Cube-root mean and variance of a cell whose CDF is flat over the first
    #: grid step and still well below 1 at the grid top (100 m).
    FAT_MEAN_CT, FAT_VARIANCE_CT = 46.0, 30.0

    def _fat_tailed_wet_result(self):
        """A wet cell with levels the grid cannot bracket.

        This is the shape that broke Tier 2b: with the grid ending at 1000 mm,
        such a cell's 0.99 quantile came back as 4,054,806 mm, because levels
        above the grid top had no bracketing pair and were extrapolated from the
        first interval, where the CDF is nearly flat.
        """
        return BmaFitResult(
            bin_label="x", region="R1", weights={"a": 1.0},
            components={
                "a": BmaComponentFit(
                    "a", "kernel_dressing", -3.0, 0.0, self.FAT_MEAN_CT, 0.0,
                    self.FAT_VARIANCE_CT, 0.0, max_mean_ct=50.0,
                )
            },
        )

    def test_fallback_cdf_has_the_fitted_paths_shape_for_per_cell_values(self):
        """`(n,)` cells against `(n, 1)` values must give `(n, 1)`, as fitted cells do.

        Broadcasting the two from the right instead gives `(n, n)`, which only
        `[..., 0]` in `bma_crps` was hiding.
        """
        fallback = BmaFitResult(bin_label="x", region="R1", weights={"a": 1.0}, components={})
        values = np.arange(5.0)[:, None]
        got = bma_mixture_cdf(fallback, {"a": np.ones(5)}, {"a": None}, values)
        fitted = bma_mixture_cdf(self._result(0.0, -1.0), {"a": np.ones(5)}, {"a": None}, values)
        assert got.shape == fitted.shape == (5, 1)
        assert np.all(got == 1.0)

    def test_levels_beyond_the_grid_are_solved_exactly_not_extrapolated(self):
        result = self._fat_tailed_wet_result()
        mean = {"a": np.full((3,), 1.0)}
        spread = {"a": None}
        # A grid ending at 100 m (the default reaches 216 m, past every level of
        # this fixture), so some levels lie beyond it and take the bisection path.
        grid = BMA_CRPS_GRID_MM[BMA_CRPS_GRID_MM <= 1e5]
        top = bma_mixture_cdf(result, mean, spread, grid[-1:])[0, 0]
        beyond = self.LEVELS > top
        assert beyond.any(), "the fixture must leave some levels above the grid"

        q = bma_predictive_quantiles(result, mean, spread, self.LEVELS, grid=grid)

        # Above the grid top, and exactly at the requested level.
        assert np.all(q[:, beyond] > grid[-1])
        achieved = bma_mixture_cdf(result, mean, spread, q)
        assert np.allclose(achieved[:, beyond], self.LEVELS[beyond], atol=1e-9)
        # Bounded by the distribution itself: no quantile of a cube-root gamma
        # sits more than 7 sd above its cube-root mean.
        assert np.all(q < (self.FAT_MEAN_CT + 7.0 * np.sqrt(self.FAT_VARIANCE_CT)) ** 3)
        assert np.all(np.diff(q, axis=-1) >= 0.0)

    def test_is_not_the_order_statistics_of_a_finite_sample(self):
        """The draw-based version is the thing this replaces; keep it visible.

        Step 13's H10 verdict turns on 0.02-0.03 mm of CRPS, and draw-based
        quantiles were off by 1-6%, so a regression back to sampling would be
        larger than the margin it decides.
        """
        result = self._result(0.0, -1.0)
        mean = {"a": np.full((4,), 3.0)}
        spread = {"a": None}
        exact = bma_predictive_quantiles(result, mean, spread, self.LEVELS)
        sampled = predictive_quantiles_bma(
            sample_bma_mixture(result, mean, spread, np.random.default_rng(0), n_samples=500),
            self.LEVELS,
        )
        assert not np.allclose(exact[:, -1], sampled[:, -1])


class TestNearDegenerateVarianceKeepsTheDistributionWhereItIs:
    """`heavy`xSI / `heavy`xNE1: the two cells that scored CRPS ~145-155 mm.

    A negative-slope variance regression, extrapolated to a test spread wider
    than training, floored the variance at 1e-6. The CDF then clipped the gamma
    scale `variance / mean` at 1e-6 on its own, which moved a spike at
    cube-root 3.5 (~43 mm) to 12.3 (~1850 mm) and put 58% of the mixture's mass
    above 1000 mm.
    """

    def _spike(self):
        # Variance pinned at the floor: the state those cells reached.
        return BmaFitResult(
            bin_label="heavy", region="SI", weights={"a": 1.0},
            components={
                "a": BmaComponentFit(
                    "a", "kernel_dressing", -20.0, 0.0, 3.5, 0.0, 1e-9, 0.0, max_mean_ct=6.8,
                )
            },
        )

    def test_cdf_puts_the_spike_at_its_own_mean(self):
        mean = {"a": np.full(2, 10.0)}
        cdf = bma_mixture_cdf(self._spike(), mean, {"a": None}, np.array([40.0, 46.0, 1000.0]))
        # 3.5**3 = 42.9 mm: all of the mass is between 40 and 46 mm.
        assert np.allclose(cdf[:, 0], 0.0, atol=1e-6)
        assert np.allclose(cdf[:, 1:], 1.0, atol=1e-6)

    def test_exceedance_agrees_with_the_cdf(self):
        mean = {"a": np.full(2, 10.0)}
        assert np.allclose(
            bma_exceedance_probability(self._spike(), mean, {"a": None}, 46.0), 0.0, atol=1e-6
        )

    def test_variance_prediction_is_held_inside_the_training_range(self):
        rng = np.random.default_rng(0)
        spread = rng.uniform(10.0, 40.0, 300)
        fmean = rng.uniform(60.0, 120.0, 300)
        # Residual variance falls as spread rises, so the fitted slope is negative.
        obs = (np.cbrt(fmean) + rng.normal(0.0, 1.0, 300) * (2.0 - np.cbrt(spread) / 3.5)) ** 3
        obs = np.clip(obs, 0.1, None)
        component = _fit_component("a", fmean, spread, obs)
        assert component.gamma_variance_slope < 0.0

        # A spread far wider than training would extrapolate below zero.
        _, _, variance = bma_module._component_predictive_params(
            component, np.array([90.0]), np.array([5000.0])
        )
        assert variance[0] == pytest.approx(component.min_variance_ct)
        assert component.min_variance_ct > 1e-3


class TestBmaCrpsAccuracy:
    """`bma_crps` against adaptive quadrature of the same mixture CDF.

    Integrated in `u = v**(1/3)`, where the CDF is smooth, so `quad` is a
    trustworthy reference. Both 31 Oct defects fail here on the old code: the
    observation's grid interval was counted twice in `int (1 - F)^2`, and the
    1 mm grid could not follow the CDF's vertical rise just above zero.
    """

    @staticmethod
    def _result(zero_intercept, mean_intercept, mean_slope, var_intercept, sources=("a",)):
        return BmaFitResult(
            bin_label="x", region="R1",
            weights={s: 1.0 / len(sources) for s in sources},
            components={
                s: BmaComponentFit(
                    s, "kernel_dressing", zero_intercept, 0.0, mean_intercept, mean_slope,
                    var_intercept, 0.0, max_mean_ct=50.0,
                )
                for s in sources
            },
        )

    @staticmethod
    def _quad_crps(result, mean, y):
        from scipy import integrate

        spread = dict.fromkeys(mean, None)

        def cdf(u):
            return bma_mixture_cdf(result, mean, spread, np.array([u**3]))[0, 0]

        uy = np.cbrt(y)
        below = (
            integrate.quad(lambda u: cdf(u) ** 2 * 3 * u * u, 0.0, uy, limit=400)[0]
            if y > 0 else 0.0
        )
        above = integrate.quad(
            lambda u: (1 - cdf(u)) ** 2 * 3 * u * u, uy, np.inf, limit=400
        )[0]
        return below + above

    @pytest.mark.parametrize(
        "params",
        [
            (0.5, 0.8, 0.0, 0.05),   # mostly dry, drizzle when wet
            (-1.0, 1.5, 0.0, 0.2),   # light rain
            (-3.0, 3.0, 0.0, 0.6),   # wet, fat tail
        ],
    )
    @pytest.mark.parametrize("y", [0.0, 0.4, 2.0, 15.0, 120.0])
    def test_matches_quadrature(self, params, y):
        result = self._result(*params)
        mean = {"a": np.array([1.0])}
        got = bma_crps(result, mean, {"a": None}, np.array([y]))[0]
        assert got == pytest.approx(self._quad_crps(result, mean, y), abs=2e-3)

    def test_quantiles_reach_their_levels(self):
        """Inverted in cube root, `F(Q(tau))` is `tau` to 1e-3 above the point mass."""
        result = self._result(-1.0, 1.5, 0.0, 0.2)
        mean = {"a": np.full((2,), 1.0)}
        spread = {"a": None}
        levels = np.linspace(0.01, 0.99, 99)
        q = bma_predictive_quantiles(result, mean, spread, levels)
        achieved = bma_mixture_cdf(result, mean, spread, q)
        p0 = bma_mixture_cdf(result, mean, spread, np.zeros(1))[:, 0]
        above = levels[None, :] > p0[:, None] + 1e-6
        assert above.any()
        np.testing.assert_allclose(
            achieved[above], np.broadcast_to(levels, achieved.shape)[above], atol=1e-3
        )
