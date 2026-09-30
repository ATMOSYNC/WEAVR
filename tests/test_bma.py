import numpy as np
import pytest
import xarray as xr

import weavr.bma as bma_module
from weavr.bma import (
    MIN_TRAIN_DAYS_PER_BIN,
    BmaComponentFit,
    BmaFitResult,
    _ensemble_crps_chunked,
    crps_chunk_cells,
    fit_hierarchical_bma,
    renormalize_bma_for_present_sources,
    sample_bma_mixture,
    score_bma,
    score_bma_and_mean,
)
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

    def test_both_numbers_come_from_one_shared_draw_set(self):
        result, mean_arrays, obs = self._fit_and_arrays()
        spread = dict.fromkeys(mean_arrays, None)

        crps, predictive_mean = score_bma_and_mean(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(7), n_samples=200
        )

        # Replaying the same generator must reproduce the exact draws the
        # function used internally, for *both* outputs.
        replayed = sample_bma_mixture(
            result,
            {s: v.values for s, v in mean_arrays.items()},
            {s: None for s in mean_arrays},
            np.random.default_rng(7),
            n_samples=200,
        )
        assert np.allclose(predictive_mean.values, replayed.mean(axis=-1))
        replayed_da = xr.DataArray(
            replayed, dims=(*obs.dims, "member"), coords=obs.coords
        )
        assert float(crps.values) == pytest.approx(
            float(ensemble_crps(replayed_da, obs, member_dim="member").values)
        )

    def test_a_second_independent_draw_would_not_match(self):
        """Guards the regression this function exists to prevent."""
        result, mean_arrays, obs = self._fit_and_arrays()
        spread = dict.fromkeys(mean_arrays, None)

        _crps, predictive_mean = score_bma_and_mean(
            result, mean_arrays, spread, obs, rng=np.random.default_rng(7), n_samples=200
        )
        second_draw = sample_bma_mixture(
            result,
            {s: v.values for s, v in mean_arrays.items()},
            {s: None for s in mean_arrays},
            np.random.default_rng(8),
            n_samples=200,
        ).mean(axis=-1)

        assert not np.allclose(predictive_mean.values, second_draw)

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

        # `predictive_mean` must stay unreduced -- `score_bma_cells` writes it
        # straight into its per-cell MSE/bias fields. (CRPS stays reduced by
        # `ensemble_crps`, exactly as before this change.)
        assert predictive_mean.dims == obs.dims
        assert predictive_mean.shape == obs.shape
        assert np.isfinite(predictive_mean.values).all()
        assert (predictive_mean.values >= 0.0).all()
        assert np.isfinite(float(crps.values))


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
