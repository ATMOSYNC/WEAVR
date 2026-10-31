#!/usr/bin/env python3
"""Run Tail Repair via Extreme-Value Tail (Step 12) -- LOYO evaluation of H7 (EVT part).

Replaces the 80-line command-line stub. That stub parsed arguments, printed
one line and returned 0: it fitted nothing and scored nothing, so
`results/tail_repair_evt.csv` never existed and H7 (EVT part) had no evidence
behind it.

The problem it addresses
------------------------
Tier 2's EMOS-CSG fits one regression per rain-intensity bin. Bins with too
few training observations -- notably `extremely_heavy` (>= 204.5 mm), where a
two-season evidence base still yields only a few dozen events -- cannot be
fitted at all, and those cells receive a point-mass fallback: a hard
probability of **zero**. The dashboard showed those cells as "not a real
probability", which is at least honest, but it means the model cannot ever
issue a rare-extreme warning.

This runner fits a pooled Generalized Pareto tail above
`u = 64.5 mm` (IMD's heavy-rainfall threshold) -- one shared shape
parameter, per-region scales -- and splices it onto the fitted CSGD
distribution above `u`, so the upper tail is described by a statistic that
does not need thousands of localised training events.

What it evaluates, per lead, under leave-one-year-out
----------------------------------------------------
Three arms, all scored on the same held-out days against the same
observations:

- `csgd_only`      -- today's behaviour: fitted bins use their CSGD
                      exceedance probability; unfittable bins get the
                      point-mass fallback of exactly 0.0.
- `csgd_gpd_tail`  -- unfittable bins are repaired by splicing the pooled GPD
                      onto the nearest fittable bin's distribution.
- `raw_ifs_ens_member_counting` -- the simple benchmark: the empirical
                      fraction of IFS-ENS members above the threshold. Only
                      computed when an IFS-ENS store is supplied; skipped and
                      recorded as unavailable otherwise, never silently.
- `qm_csgd_only`, `qm_csgd_gpd_tail` -- step 11's QM variant
                      (`--qm-variant`, default `same_period`): EMOS-CSG refitted
                      on quantile-mapped members, without and with the tail.
                      Step 12's prompt says both "build on step 11's best QM
                      variant if H7-QM passed" and "compare against the step-11
                      QM variant"; step 11 also found QM ahead of EMOS-CSG
                      raises CRPS, so both are reported rather than one chosen.

Scores per arm: Brier and BSS at 115.6/204.5 mm, SEDI from expected counts,
tw-CRPS at 64.5 and 115.6 mm (closed form: `weavr.emos.csgd_twcrps`,
`weavr.tail.spliced_twcrps`, `weavr.verify.twcrps_ensemble` for the members),
and Brier at 7.5 mm, the guard H7 pre-registers against a tail repair that
inflates everything.

The headline claim in docs/tail-repair-results.md is that splicing
"eliminates false zero-probability assignments across all five leads". So
that is measured directly, not assumed: `false_zero_cells` counts test cells
whose exceedance probability is exactly 0.0 at 115.6 and 204.5 mm.

Ensembles come from the **lagged GraphCast store**, which exists for both
seasons. That is deliberate: `weavr.emos.fit_emos_csg` fits any ensemble
source, so this step does not need the daily IFS-ENS store and is not blocked
on it. The IFS-ENS store only adds the third benchmark arm.

Outputs
-------
`results/tail_repair_evt.csv`
    Per `(lead, fold, arm, threshold)`: Brier, Brier skill score against the
    IMD JJAS climatology, SEDI, RMSE of the ensemble mean, plus the
    false-zero count and the per-method cell breakdown.
`results/tail_repair_evt_paired.csv`
    Per `(comparison, lead, threshold)`: A minus B with 95% CIs on Brier, SEDI,
    tw-CRPS@64.5/115.6 and Brier@7.5, and H7's per-lead outcome. Comparisons:
    tail vs `csgd_only` (H7's), vs the member-counting benchmark, vs step 11's
    QM variant, and QM+tail vs QM.
`results/tail_repair_evt_reliability.csv`
    Per `(lead, arm, threshold)`, pooled over folds: forecast count, mean
    forecast probability and observed frequency in each probability bin, with
    exact zeros in a bin of their own, and a 95% block-bootstrap CI on the
    observed frequency.
`results/per_day/evt_{arm}_t{t}__lead<L>.csv`
    Per-day rows in `weavr.score_io.per_day_scores` schema so
    `scripts/run_scorecard.py` can bootstrap them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_seeps_climatology import load_climatology  # noqa: E402
from run_scorecard import (  # noqa: E402
    MIN_LEADS_FOR_PASS,
    _bootstrap_categorical_difference,
    _categorical_from_counts,
    _pooled_sedi,
)
from run_tail_repair_qm_emos import quantile_mapped_members  # noqa: E402
from run_tier2_hierarchical_baseline import (  # noqa: E402
    load_graphcast_ensemble as _tier2_load_graphcast_ensemble,
)
from run_tier2_hierarchical_baseline import (
    load_ifs_ensemble as _tier2_load_ifs_ensemble,
)

from weavr.emos import (  # noqa: E402
    csgd_twcrps,
    exceedance_probability_csgd,
    fit_emos_csg,
    predict_csgd_params,
)
from weavr.quantile_mapping import climatology_excluding_years  # noqa: E402
from weavr.rain_bins import RAIN_BIN_LABELS, classify_rain_bin  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.score_io import (  # noqa: E402
    FORCE_HELP,
    exceedance_event,
    guard_result_overwrites,
    per_day_scores,
    resolve_result_paths,
    write_per_day_scores,
)
from weavr.significance import (  # noqa: E402
    block_bootstrap_indices,
    bootstrap_is_degenerate,
    paired_difference_ci,
)
from weavr.splits import iter_evaluation_folds  # noqa: E402
from weavr.stores import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    DEFAULT_IFS_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
    open_multi_season,
    resolve_store_paths,
)
from weavr.tail import (  # noqa: E402
    DEFAULT_TAIL_THRESHOLD_U,
    exceedance_probability_with_tail,
    fit_pooled_gpd,
    spliced_twcrps,
)
from weavr.verify import expected_contingency_counts, twcrps_ensemble  # noqa: E402

LEAD_HOURS = [24, 48, 72, 96, 120]
EXTREME_THRESHOLDS = [115.6, 204.5]
GRAPHCAST_ENSEMBLE_SOURCE = "graphcast"
TWCRPS_THRESHOLDS = (64.5, 115.6)
BRIER_GUARD_THRESHOLD = 7.5
H7_SEDI_THRESHOLD = 115.6
QM_VARIANTS = ("same_period", "climatology")
COUNT_KEYS = ("hits", "misses", "false_alarms", "correct_negatives")
#: (A, B) pairs scored as A minus B. The first is H7's.
COMPARISONS = [
    ("csgd_gpd_tail", "csgd_only"),
    ("csgd_gpd_tail", "raw_ifs_ens_member_counting"),
    ("csgd_gpd_tail", "qm_csgd_only"),
    ("qm_csgd_gpd_tail", "qm_csgd_only"),
]


def load_graphcast_ensemble(paths: list[str | Path], lead_hours: int) -> xr.DataArray:
    """GraphCast's lagged pseudo-ensemble at one lead, in mm, IMD-day-aligned.

    A thin alias over `run_tier2_hierarchical_baseline.load_graphcast_ensemble`
    rather than a second implementation. That loader is the canonical one: the
    lagged store keeps `(nominal_time, lead_hours, member_offset_hours, lat,
    lon)`, so the pseudo-ensemble has to be rebuilt through
    `run_phase2_ensemble_baseline.build_ensembles_for_lead` and re-densified
    from `lead_hours`/`nominal_time`, which are not `prediction_timedelta`/
    `time`. Re-deriving that here got it wrong twice.
    """
    return _tier2_load_graphcast_ensemble(paths, lead_hours)


def fit_tail_on_obs(
    obs: xr.DataArray, region_labels: xr.DataArray, threshold_u: float
):
    """Pooled GPD fitted on `obs`'s exceedances above u, pooled per region.

    `main` passes the IMD JJAS archive (2006-2020) with the fold's test year
    removed, as step 12's prompt specifies ("IMD archive exceedances above u,
    EXCLUDING the test year"). Until 31 Oct it passed the training season
    alone, ~122 days, on the grounds of staying inside the cross-validation
    boundary; dropping the test year from the archive does that too, with
    about 14 times the exceedances the pooled fit exists to use. `obs` is
    `(time, latitude, longitude)`.
    """
    values = obs.values
    exceedances: dict[str, np.ndarray] = {}
    for region in np.unique(region_labels.values):
        mask = region_labels.values == region
        pooled = values[..., mask].ravel()
        pooled = pooled[np.isfinite(pooled) & (pooled > threshold_u)]
        if pooled.size:
            exceedances[str(region)] = pooled
    if not exceedances:
        raise ValueError(
            f"No training exceedances above u={threshold_u} mm; cannot fit a tail. "
            "The evidence base has too few heavy-rainfall events to repair."
        )
    return fit_pooled_gpd(exceedances, threshold_u=threshold_u)


def csgd_only_probabilities(
    forecasts: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    emos_results_by_bin: dict,
    threshold: float,
) -> np.ndarray:
    """Today's behaviour: CSGD where fitted, hard 0.0 where not.

    Deliberately reproduces the point-mass fallback rather than improving it,
    because this is the arm the GPD tail has to beat.
    """
    ensemble_mean = forecasts.mean(dim="member", skipna=True)
    # ddof=1, as `fit_emos_csg` computes the spread it fits on. ddof=0 here fed
    # the regression a spread 6-9% narrower than it was trained on (6-9 lagged
    # members), skewing every predicted distribution.
    ensemble_spread = forecasts.std(dim="member", ddof=1, skipna=True)
    probabilities = np.full(ensemble_mean.shape, np.nan, dtype=float)

    labels = rain_bin_labels.values
    for bin_label in RAIN_BIN_LABELS:
        result = emos_results_by_bin.get(str(bin_label))
        if result is None:
            # Unfittable bin: the point-mass fallback that produces the
            # false zero-probability assignments this step exists to remove.
            probabilities[labels == bin_label] = 0.0
            continue
        mask = labels == bin_label
        if not mask.any():
            continue
        mean, std, shift = predict_csgd_params(
            result,
            ensemble_mean.values[mask],
            ensemble_spread.values[mask],
        )
        probabilities[mask] = exceedance_probability_csgd(mean, std, shift, threshold)
    return probabilities


def as_data_array(values: np.ndarray, template: xr.DataArray, name: str) -> xr.DataArray:
    return xr.DataArray(values, coords=template.coords, dims=template.dims, name=name)


def expected_counts(probability: np.ndarray, obs_binary: np.ndarray) -> dict[str, np.ndarray]:
    """Expected contingency counts implied by a probability forecast, per day.

    SEDI is defined on probabilities, not on a hard decision: expected counts
    let the existing `_pooled_sedi` helper work unchanged and avoid inventing an
    arbitrary decision threshold (0.5 would silently discard exactly the
    low-probability extreme cells this step is about).

    Delegates to `weavr.verify.expected_contingency_counts`. Until 31 Oct this
    function computed misses as `sum(p * (1 - o))` and false alarms as
    `sum((1 - p) * o)` -- the other way round -- so every SEDI it fed, and the
    hits/misses columns written into the per-day files, were wrong.
    """
    return expected_contingency_counts(probability, obs_binary, axis=(1, 2))


#: Reliability bin edges. Exact zero gets a bin of its own -- a hard 0.0 is the
#: failure this step exists to remove -- and the rest are log-spaced, because
#: extreme-rain probabilities live almost entirely below 0.05 and equal-width
#: bins would put nearly every forecast in the first one.
RELIABILITY_EDGES = np.array([0.0, 1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])


def reliability_counts(probability: np.ndarray, obs_binary: np.ndarray) -> pd.DataFrame:
    """Per-bin forecast count, summed probability and summed outcome.

    Sums rather than means, so folds can be pooled by adding rows; the mean
    forecast and observed frequency are taken once the sums are complete.
    Bin 0 holds exactly-zero probabilities; bin `k` holds `(edges[k-1], edges[k]]`.
    """
    finite = np.isfinite(probability) & np.isfinite(obs_binary)
    p = np.clip(probability[finite], 0.0, 1.0)
    o = obs_binary[finite]
    index = np.searchsorted(RELIABILITY_EDGES, p, side="left")
    n_bins = RELIABILITY_EDGES.size
    return pd.DataFrame(
        {
            "bin": np.arange(n_bins),
            "p_lo": np.concatenate([[0.0], RELIABILITY_EDGES[:-1]]),
            "p_hi": RELIABILITY_EDGES,
            "n": np.bincount(index, minlength=n_bins),
            "sum_forecast": np.bincount(index, weights=p, minlength=n_bins),
            "sum_observed": np.bincount(index, weights=o, minlength=n_bins),
        }
    )


def csgd_only_twcrps(
    ensemble_mean: np.ndarray,
    ensemble_spread: np.ndarray,
    labels: np.ndarray,
    emos_results_by_bin: dict,
    obs_values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Per-cell tw-CRPS of `csgd_only`: the fitted CSGD, or the point mass at 0.

    An unfittable bin's forecast is the point-mass fallback `csgd_only` scores
    as probability 0, so its tw-CRPS is `max(y, t) - t`, the same distribution
    `csgd_only_probabilities` uses.
    """
    scores = np.full(ensemble_mean.shape, np.nan, dtype=float)
    valid = np.isfinite(obs_values) & np.isfinite(ensemble_mean)
    for bin_label in RAIN_BIN_LABELS:
        mask = (labels == bin_label) & valid
        if not mask.any():
            continue
        result = emos_results_by_bin.get(str(bin_label))
        if result is None:
            scores[mask] = np.maximum(obs_values[mask], threshold) - threshold
            continue
        mean, std, shift = predict_csgd_params(
            result, ensemble_mean[mask], ensemble_spread[mask]
        )
        scores[mask] = csgd_twcrps(mean, std, shift, obs_values[mask], threshold)
    return scores


def tail_twcrps(
    p_exceed_u: np.ndarray,
    valid: np.ndarray,
    tail_fit,
    region_grid: np.ndarray,
    obs_values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Per-cell tw-CRPS of the spliced arm at `threshold >= u` (`spliced_twcrps`).

    A fitted shape xi >= 1 has no finite tail mean, so the spliced forecast's
    tw-CRPS is infinite: recorded as NaN and said so, rather than failing the
    run and losing every other arm. (Real fits have xi ~ 0.1; the synthetic
    replica can produce xi > 1.)
    """
    scores = np.full(p_exceed_u.shape, np.nan, dtype=float)
    if tail_fit.shape_xi >= 1.0:
        print(f"  tw-CRPS of the spliced tail undefined: xi = {tail_fit.shape_xi:.3f} >= 1")
        return scores
    regions = np.broadcast_to(region_grid, p_exceed_u.shape)
    scores[valid] = spliced_twcrps(
        p_exceed_u[valid], tail_fit, regions[valid], obs_values[valid], threshold
    )
    return scores


def ensemble_twcrps_per_day(
    ensemble: xr.DataArray, obs: xr.DataArray, threshold: float
) -> np.ndarray:
    """Per-day tw-CRPS of a raw ensemble (the member-counting benchmark's),
    reduced over space exactly as `per_day_scores` reduces an ensemble's."""
    spatial = [d for d in obs.dims if d != "sample"]
    return np.asarray(twcrps_ensemble(ensemble, obs, threshold, dim=spatial).values, dtype=float)


def reliability_day_sums(
    probability: np.ndarray, obs_binary: np.ndarray
) -> dict[str, np.ndarray]:
    """`reliability_counts` per day: `(n_days, n_bins)` arrays to bootstrap over days."""
    n_days = probability.shape[0]
    n_bins = RELIABILITY_EDGES.size
    out = {key: np.zeros((n_days, n_bins)) for key in ("n", "sum_forecast", "sum_observed")}
    for day in range(n_days):
        counts = reliability_counts(probability[day], obs_binary[day])
        for key in out:
            out[key][day] = counts[key].to_numpy(dtype=float)
    return out


def reliability_with_ci(
    sums: dict[str, np.ndarray], block_days: int, n_resamples: int, seed: int
) -> pd.DataFrame:
    """Reliability table with 95% moving-block bootstrap CIs on each bin's
    observed frequency (days resampled, as every other CI here).

    Step 12 item 3 asks for these, and expects them wide: above 115.6 mm a
    bin can hold a handful of events, which is the point of reporting them.
    """
    n, fc, ob = sums["n"], sums["sum_forecast"], sums["sum_observed"]
    total_n = n.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_forecast = np.where(total_n > 0, fc.sum(axis=0) / total_n, np.nan)
        observed = np.where(total_n > 0, ob.sum(axis=0) / total_n, np.nan)
        ci_lo = np.full(total_n.shape, np.nan)
        ci_hi = np.full(total_n.shape, np.nan)
        if not bootstrap_is_degenerate(n.shape[0], block_days):
            indices = block_bootstrap_indices(n.shape[0], block_days, n_resamples, seed)
            rep_n = n[indices].sum(axis=1)
            rep_freq = np.where(rep_n > 0, ob[indices].sum(axis=1) / rep_n, np.nan)
            for b in range(total_n.size):
                finite = rep_freq[:, b][np.isfinite(rep_freq[:, b])]
                if finite.size:
                    ci_lo[b], ci_hi[b] = np.quantile(finite, [0.025, 0.975])
    return pd.DataFrame(
        {
            "bin": np.arange(total_n.size),
            "p_lo": np.concatenate([[0.0], RELIABILITY_EDGES[:-1]]),
            "p_hi": RELIABILITY_EDGES,
            "n": total_n.astype(int),
            "mean_forecast": mean_forecast,
            "observed_frequency": observed,
            "observed_frequency_ci_lo": ci_lo,
            "observed_frequency_ci_hi": ci_hi,
            "sum_forecast": fc.sum(axis=0),
            "sum_observed": ob.sum(axis=0),
        }
    )


def paired_arm_comparison(
    a_days: pd.DataFrame,
    b_days: pd.DataFrame,
    threshold: float,
    block_days: int,
    n_resamples: int,
    seed: int,
) -> dict:
    """Arm A minus arm B on the same days: Brier and SEDI at `threshold`,
    tw-CRPS at 64.5 and 115.6, Brier at 7.5 -- each with a 95% paired CI.
    Negative is better for every difference (SEDI is negated, as in the
    scorecard)."""
    if not a_days["date"].reset_index(drop=True).equals(b_days["date"].reset_index(drop=True)):
        raise RuntimeError("paired arms are not on the same days in the same order")

    def mean_ci(column: str):
        return paired_difference_ci(
            a_days[column].to_numpy(float),
            b_days[column].to_numpy(float),
            block_days=block_days,
            n_resamples=n_resamples,
            seed=seed,
            aggregate="mean",
        )

    def below(ci) -> bool:
        return bool(np.isfinite(ci.ci_hi) and ci.ci_hi < 0.0)

    brier_ci = mean_ci(f"brier_{threshold}")
    sedi_ci = _bootstrap_categorical_difference(
        _categorical_from_counts(a_days, threshold, "sedi"),
        _categorical_from_counts(b_days, threshold, "sedi"),
        "sedi",
        block_days,
        n_resamples,
        seed,
    )
    row = {
        "n_test_days": int(len(a_days)),
        "brier_delta": brier_ci.estimate,
        "brier_ci_lo": brier_ci.ci_lo,
        "brier_ci_hi": brier_ci.ci_hi,
        "brier_improved": below(brier_ci),
        "sedi_delta": sedi_ci.estimate,
        "sedi_ci_lo": sedi_ci.ci_lo,
        "sedi_ci_hi": sedi_ci.ci_hi,
        "sedi_improved": below(sedi_ci),
        "bootstrap_degenerate": bool(brier_ci.degenerate),
    }
    for t in TWCRPS_THRESHOLDS:
        ci = mean_ci(f"twcrps_{t}_mm")
        row.update(
            {
                f"twcrps_{t}_delta": ci.estimate,
                f"twcrps_{t}_ci_lo": ci.ci_lo,
                f"twcrps_{t}_ci_hi": ci.ci_hi,
                f"twcrps_{t}_improved": below(ci),
            }
        )
    guard = mean_ci(f"brier_{BRIER_GUARD_THRESHOLD}")
    row.update(
        {
            f"brier_{BRIER_GUARD_THRESHOLD}_delta": guard.estimate,
            f"brier_{BRIER_GUARD_THRESHOLD}_ci_lo": guard.ci_lo,
            f"brier_{BRIER_GUARD_THRESHOLD}_ci_hi": guard.ci_hi,
            f"brier_{BRIER_GUARD_THRESHOLD}_degraded": bool(
                np.isfinite(guard.ci_lo) and guard.ci_lo > 0.0
            ),
        }
    )
    return row


def climatology_exceedance(
    archive_rain: xr.DataArray, thresholds
) -> dict[float, xr.DataArray]:
    """Per-cell climatological P(rain >= t), the Brier skill score's reference.

    `archive_rain` must already exclude the fold's test year
    (`climatology_excluding_years`), as H3's climatology and step 11's QM
    reference do: a reference that has seen the test season is in-sample, and
    until 31 Oct this one had.
    """
    return {t: (archive_rain >= t).mean(dim="time") for t in thresholds}


def member_counting_probabilities(
    ifs_ensemble: xr.DataArray, threshold: float, test_mask: np.ndarray
) -> np.ndarray:
    """`P(rain > threshold)` from the raw IFS-ENS members, nothing else.

    This is the benchmark step 12 is measured *against*: the empirical
    fraction of members exceeding the threshold, with no calibration applied.
    It is the comparison a reader should ask for before believing the spliced
    tail adds anything, and it could not be run until step 07 produced daily
    full-member IFS-ENS stores for both seasons (#74 recorded it as NOT RUN for
    exactly that reason).

    Returns a `(sample, latitude, longitude)` grid containing **only the test
    samples**, so it lines up with `test_obs` rather than with the full
    two-season series. Returning the full grid with NaN padding is the trap
    here: the shapes then differ by the fold size and the scoring step raises
    deep inside a boolean mask.
    """
    exceed = (ifs_ensemble > threshold).mean(dim="member", skipna=True)

    # `_align_to_imd_day` (inside `load_ifs_ensemble`) renames the time axis to
    # `sample` and has already selected one lead; a raw store would still carry
    # `time`/`prediction_timedelta`. Accept either, and drop any leftover
    # singleton axis rather than assuming a particular caller's shape.
    time_dim = "sample" if "sample" in exceed.dims else "time"
    for dim in (d for d in exceed.dims if d not in (time_dim, "latitude", "longitude")):
        if exceed.sizes[dim] == 1:
            exceed = exceed.squeeze(dim, drop=True)
        else:
            raise ValueError(
                f"member_counting_probabilities expects a single lead, but "
                f"{ifs_ensemble.name or 'the ensemble'} still has "
                f"{exceed.sizes[dim]} values along {dim!r}. Select one lead first."
            )

    grid = exceed.transpose(time_dim, "latitude", "longitude").values.astype(float)
    mask = np.asarray(test_mask, dtype=bool)
    if grid.shape[0] != mask.size:
        raise ValueError(
            f"IFS-ENS ensemble has {grid.shape[0]} samples but the fold mask has "
            f"{mask.size}; they must describe the same days."
        )
    return grid[mask]


def benchmark_row(
    probs: np.ndarray,
    obs_test: np.ndarray,
    climatology_grid: np.ndarray,
    threshold: float,
    lead_hours: int,
    split_label: str,
    split_kind: str,
    n_train: int,
    n_test: int,
) -> dict:
    """Score one (lead, fold, threshold) of the member-counting benchmark.

    Scored with the *same* helpers as the spliced-tail arms -- `expected_counts`
    then `compute_sedi` -- so the comparison differs only in the forecast being
    scored, never in how it is scored.
    """
    # `>=`, as the other two arms and the climatology reference define the
    # event; `>` scored a cell observing exactly the threshold differently.
    # NaN where nothing was observed: member counting is finite over the whole
    # box, and an unobserved cell is not an observed non-event (F8).
    obs_binary = exceedance_event(obs_test, threshold)
    finite = np.isfinite(probs) & np.isfinite(obs_binary)
    p = probs[finite]
    o = obs_binary[finite]
    # The climatological reference is a per-cell `(latitude, longitude)` grid,
    # not a scalar, so it has to be broadcast up to the forecast's shape and
    # then reduced with the same mask `finite` uses. Broadcasting (rather than
    # collapsing the sample axis first) is what keeps it aligned: the grid
    # repeats once per sample, exactly as the forecast does.
    climatology_broadcast = np.broadcast_to(
        np.asarray(climatology_grid, dtype=float), probs.shape
    )
    climatology_selected = climatology_broadcast[finite]
    brier = float(np.mean((p - o) ** 2))
    counts = expected_counts(probs, obs_binary)
    sedi = compute_sedi(
        float(np.sum(counts["hits"])),
        float(np.sum(counts["misses"])),
        float(np.sum(counts["false_alarms"])),
        float(np.sum(counts["correct_negatives"])),
    )
    climat_brier = float(np.mean((climatology_selected - o) ** 2))
    return {
        "lead_hours": lead_hours,
        "fold": split_label,
        "split": split_kind,
        "arm": "raw_ifs_ens_member_counting",
        "threshold": threshold,
        "n_train": n_train,
        "n_test": n_test,
        "brier": brier,
        "climatology_brier": climat_brier,
        "brier_skill_score": 1.0 - (brier / climat_brier) if climat_brier > 0 else float("nan"),
        "sedi": sedi,
        "rmse_of_mean_mm": float("nan"),
        "false_zero_cells": int(np.sum(np.isfinite(probs) & (probs == 0.0))),
        "threshold_u": float("nan"),
        "tail_shape_xi": float("nan"),
        "fit_bins": "",
        "n_bins_total": 0,
    }


def compute_sedi(
    hits: float, misses: float, false_alarms: float, correct_negatives: float
) -> float:
    """SEDI from pooled counts.

    Kept as the module's public entry point (the stub's signature, and what
    `tests/test_run_tail_repair_evt.py` exercises) but implemented by
    delegating to `run_scorecard._pooled_sedi`, which is the canonical version
    and mirrors `weavr.verify.sedi` including Ferro & Stephenson's degenerate
    cases. The stub carried its own eps-guarded copy, which returned 0.0 for a
    perfectly separated forecast instead of the +inf that formulation implies
    -- one implementation, not two that can drift.
    """
    return _pooled_sedi(hits, misses, false_alarms, correct_negatives)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Split out of `main` so the argument surface can be tested directly --
    `tests/test_run_tail_repair_evt.py` imports this by name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lagged-stores", nargs="+", default=None, help="Lagged ensemble stores")
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="Baseline stores (for imd_observed and climatology)",
    )
    parser.add_argument("--baseline-store", default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument(
        "--ifs-ensemble-stores",
        nargs="+",
        default=None,
        help="Optional. Enables the raw IFS-ENS member-counting benchmark arm.",
    )
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Default: <results-dir>/tail_repair_evt.csv",
    )
    parser.add_argument(
        "--paired-out-csv",
        type=Path,
        default=None,
        help="Default: <results-dir>/tail_repair_evt_paired.csv",
    )
    parser.add_argument(
        "--reliability-out-csv",
        type=Path,
        default=None,
        help="Default: <results-dir>/tail_repair_evt_reliability.csv",
    )
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--qm-variant",
        choices=QM_VARIANTS,
        default="same_period",
        help="Step 11 reference variant for the QM comparator arms (default: same_period)",
    )
    parser.add_argument(
        "--threshold-u",
        type=float,
        default=DEFAULT_TAIL_THRESHOLD_U,
        help=f"Tail splice threshold in mm (default: {DEFAULT_TAIL_THRESHOLD_U})",
    )
    return parser.parse_args(args)


def _arm_frame(
    mean: xr.DataArray,
    test_obs: xr.DataArray,
    split_label: str,
    threshold: float,
    probs: np.ndarray,
    probs_guard: np.ndarray,
    twcrps_by_threshold: dict[float, np.ndarray],
) -> pd.DataFrame:
    """One arm's per-day frame at one threshold: Brier at the threshold and at
    7.5 mm, tw-CRPS at 64.5 and 115.6 mm, and the threshold's contingency
    columns replaced by the probability forecast's expected counts."""
    frame = per_day_scores(
        mean,
        test_obs,
        fold=split_label,
        probabilities={
            BRIER_GUARD_THRESHOLD: as_data_array(probs_guard, mean, "p_guard"),
            threshold: as_data_array(probs, mean, f"p_exceed_{threshold}"),
        },
    )
    for t, values in twcrps_by_threshold.items():
        if values.ndim == 1:  # already per day (the ensemble benchmark)
            frame[f"twcrps_{t}_mm"] = values
        else:
            with np.errstate(invalid="ignore"):
                frame[f"twcrps_{t}_mm"] = np.nanmean(values, axis=(1, 2))
    counts = expected_counts(probs, exceedance_event(test_obs.values, threshold))
    for name, values in counts.items():
        frame[f"{name}_{threshold}"] = values
    return frame


def main() -> int:
    args = parse_args()

    _paths = resolve_result_paths(
        args.results_dir,
        {
            "out_csv": "tail_repair_evt.csv",
            "paired_out_csv": "tail_repair_evt_paired.csv",
            "reliability_out_csv": "tail_repair_evt_reliability.csv",
        },
        {
            "out_csv": args.out_csv,
            "paired_out_csv": args.paired_out_csv,
            "reliability_out_csv": args.reliability_out_csv,
        },
    )
    args.out_csv = _paths["out_csv"]
    args.paired_out_csv = _paths["paired_out_csv"]
    args.reliability_out_csv = _paths["reliability_out_csv"]
    guard_result_overwrites(_paths.values(), force=args.force)

    # `to_csv` needs its parent to exist and will not create it, and
    # `write_per_day_scores` only makes the `per_day/` subdirectory -- so a
    # fresh --results-dir failed at the very last step after all the compute.
    for _p in _paths.values():
        Path(_p).parent.mkdir(parents=True, exist_ok=True)

    legacy = args.baseline_store if args.baseline_store is not None else args.store
    baseline_paths = resolve_store_paths(
        args.baseline_stores, legacy, DEFAULT_BASELINE_DAILY_STORES, "data/baseline_2020_jjas.zarr"
    )
    # The lagged defaults, not the IFS ones: a bad merge had passed
    # DEFAULT_IFS_DAILY_STORES here, so a run without --lagged-stores loaded the
    # IFS-ENS stores as the GraphCast ensemble.
    lagged_paths = resolve_store_paths(args.lagged_stores, None, DEFAULT_LAGGED_DAILY_STORES, None)
    # The `--ifs-ensemble-stores` flag existed but was never resolved into
    # paths, so `ifs_available` was decided on a non-None argument that nothing
    # then read. That is why the runner announced the benchmark as "enabled"
    # and emitted no rows for it.
    ifs_paths = resolve_store_paths(
        args.ifs_ensemble_stores,
        None,
        DEFAULT_IFS_DAILY_STORES,
        None,
    )

    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    climatology = load_climatology(args.climatology).load()
    climatology_rain = climatology["rain"]
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    region_grid = region_labels.transpose("latitude", "longitude").values

    ifs_available = args.ifs_ensemble_stores is not None
    print(f"Tail repair via extreme-value tail (Step 12), u={args.threshold_u} mm")
    print(f"Lagged stores: {lagged_paths}")
    print(f"Step-11 QM comparator: EMOS-CSG on {args.qm_variant} quantile-mapped members")
    print(
        "IFS-ENS member-counting benchmark: "
        + (
            "enabled"
            if ifs_available
            else "SKIPPED (no --ifs-ensemble-stores; recorded unavailable)"
        )
    )

    metric_rows: list[dict] = []
    pooled_per_day: dict[tuple[str, float, int], list[pd.DataFrame]] = {}
    reliability_sums: dict[tuple[str, float, int], list[dict[str, np.ndarray]]] = {}

    for lead_hours in LEAD_HOURS:
        forecasts = load_graphcast_ensemble(lagged_paths, lead_hours)
        ifs_ensemble = (
            _tier2_load_ifs_ensemble(ifs_paths, lead_hours) if ifs_available else None
        )
        ensemble_mean_all = forecasts.mean(dim="member", skipna=True)
        sample_values = ensemble_mean_all["sample"].values
        obs_aligned = obs["rain"].reindex(time=sample_values).rename(time="sample")
        has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
        forecasts = forecasts.isel(sample=has_obs.values)
        obs_aligned = obs_aligned.isel(sample=has_obs.values)
        if ifs_ensemble is not None:
            # The IFS series comes from a different store and is not filtered by
            # `has_obs`, so at some leads it is longer than the sample set the
            # forecasts were trimmed to (242 vs 240 at lead 48). Reindex onto the
            # forecasts' own `sample` axis so the benchmark is scored on exactly
            # the days the other arms were scored on, rather than raising a
            # length mismatch several minutes into the run.
            ifs_ensemble = ifs_ensemble.reindex(sample=forecasts["sample"].values)
            if bool(np.isnan(ifs_ensemble).all()):
                raise ValueError(
                    "the IFS-ENS ensemble shares no days with the forecast "
                    "sample set; check that both stores cover the same season."
                )

        # Derived only after the IMD filter: two IMD days in this period fall
        # outside the grid, so every array built from these stays the same
        # length as `forecasts`/`obs_aligned` and the fold masks line up.
        ensemble_mean = forecasts.mean(dim="member", skipna=True)
        rain_bin_labels = classify_rain_bin(ensemble_mean)

        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))

        for train_mask, test_mask, split_label in folds:
            train_obs = obs_aligned.isel(sample=train_mask)
            test_obs = obs_aligned.isel(sample=test_mask)
            test_mean = ensemble_mean.isel(sample=test_mask)
            test_labels = rain_bin_labels.isel(sample=test_mask)
            obs_values = test_obs.values
            split_kind = (
                "leave_one_year_out"
                if split_label != "seasonal_block_split"
                else "seasonal_block_split"
            )

            emos_results = fit_emos_csg(
                forecasts,
                obs_aligned,
                rain_bin_labels,
                train_mask,
                source=GRAPHCAST_ENSEMBLE_SOURCE,
            )
            # The 15-year IMD archive minus this fold's test year (step 12 item 1).
            tail_archive = climatology_excluding_years(climatology_rain, sample_times[test_mask])
            tail_fit = fit_tail_on_obs(tail_archive, region_labels, args.threshold_u)
            # The Brier skill reference, from the same archive without the test year.
            climatology_probability = climatology_exceedance(tail_archive, EXTREME_THRESHOLDS)
            fitable = sorted(emos_results.keys())

            # Step 11's QM variant: members mapped on the training days only
            # (the climatology reference without the test year), EMOS-CSG
            # refitted on them; rain bins stay those of the raw mean, as in
            # run_tail_repair_qm_emos.py.
            reference = (
                train_obs
                if args.qm_variant == "same_period"
                else climatology_excluding_years(climatology_rain, sample_times[test_mask])
            )
            qm_forecasts = quantile_mapped_members(
                forecasts, train_mask, reference, region_labels, lead_hours,
                GRAPHCAST_ENSEMBLE_SOURCE,
            )
            qm_results = fit_emos_csg(
                qm_forecasts, obs_aligned, rain_bin_labels, train_mask,
                source=GRAPHCAST_ENSEMBLE_SOURCE,
            )

            family_inputs = {
                "": (emos_results, forecasts.isel(sample=test_mask)),
                "qm_": (qm_results, qm_forecasts.isel(sample=test_mask)),
            }
            for prefix, (results, members) in family_inputs.items():
                mean = members.mean(dim="member", skipna=True)
                spread = members.std(dim="member", ddof=1, skipna=True)
                valid = np.isfinite(obs_values) & np.isfinite(mean.values)

                def with_tail(threshold: float):
                    # Bins from the raw mean (what EMOS was fitted on), the
                    # CSGD's predictors from this family's members.
                    return exceedance_probability_with_tail(
                        test_mean.values, mean.values, spread.values, results,
                        tail_fit, region_labels.values, threshold,
                    )

                p_u, _ = with_tail(args.threshold_u)
                twcrps = {
                    f"{prefix}csgd_only": {
                        t: csgd_only_twcrps(
                            mean.values, spread.values, test_labels.values, results,
                            obs_values, t,
                        )
                        for t in TWCRPS_THRESHOLDS
                    },
                    f"{prefix}csgd_gpd_tail": {
                        t: tail_twcrps(p_u, valid, tail_fit, region_grid, obs_values, t)
                        for t in TWCRPS_THRESHOLDS
                    },
                }
                guard = {
                    f"{prefix}csgd_only": csgd_only_probabilities(
                        members, test_labels, results, BRIER_GUARD_THRESHOLD
                    ),
                    f"{prefix}csgd_gpd_tail": with_tail(BRIER_GUARD_THRESHOLD)[0],
                }

                for threshold in EXTREME_THRESHOLDS:
                    obs_binary = exceedance_event(obs_values, threshold)
                    baseline_probs = csgd_only_probabilities(
                        members, test_labels, results, threshold
                    )
                    tail_probs, methods = with_tail(threshold)

                    for arm, probs in (
                        (f"{prefix}csgd_only", baseline_probs),
                        (f"{prefix}csgd_gpd_tail", tail_probs),
                    ):
                        frame = _arm_frame(
                            mean, test_obs, split_label, threshold, probs, guard[arm],
                            twcrps[arm],
                        )
                        counts = expected_counts(probs, obs_binary)
                        climat_p_da = as_data_array(
                            np.broadcast_to(
                                climatology_probability[threshold].values, probs.shape
                            ).copy(),
                            mean,
                            "climat",
                        )
                        brier = float(frame[f"brier_{threshold}"].mean())
                        climat_brier = float(
                            per_day_scores(
                                mean,
                                test_obs,
                                fold=split_label,
                                probabilities={threshold: climat_p_da},
                            )[f"brier_{threshold}"].mean()
                        )
                        bss = 1.0 - (brier / climat_brier) if climat_brier > 0 else float("nan")

                        row = {
                            "lead_hours": lead_hours,
                            "fold": split_label,
                            "split": split_kind,
                            "arm": arm,
                            "threshold": threshold,
                            "n_train": int(train_mask.sum()),
                            "n_test": int(test_mask.sum()),
                            "brier": brier,
                            "climatology_brier": climat_brier,
                            "brier_skill_score": bss,
                            "sedi": _pooled_sedi(
                                *[float(np.sum(counts[k])) for k in COUNT_KEYS]
                            ),
                            "rmse_of_mean_mm": float(np.sqrt(frame["mse_mm2"].mean())),
                            "false_zero_cells": int(np.sum(probs == 0.0)),
                            "threshold_u": args.threshold_u,
                            "tail_shape_xi": float(tail_fit.shape_xi),
                            "tail_shape_xi_se": (
                                float(tail_fit.xi_se)
                                if tail_fit.xi_se is not None else float("nan")
                            ),
                            "tail_shape_xi_ci_lo": (
                                float(tail_fit.xi_ci[0])
                                if tail_fit.xi_ci is not None else float("nan")
                            ),
                            "tail_shape_xi_ci_hi": (
                                float(tail_fit.xi_ci[1])
                                if tail_fit.xi_ci is not None else float("nan")
                            ),
                            "tail_fit_days": int(tail_archive.sizes["time"]),
                            "fit_bins": ";".join(
                                sorted(results.keys()) if prefix else fitable
                            ),
                            "n_bins_total": len(RAIN_BIN_LABELS),
                        }
                        if arm.endswith("csgd_gpd_tail"):
                            for method in ("csgd", "csgd+gpd_tail", "fallback"):
                                row[f"cells_{method}"] = int(np.sum(methods == method))
                        row.update(
                            {
                                f"brier_{BRIER_GUARD_THRESHOLD}": float(
                                    frame[f"brier_{BRIER_GUARD_THRESHOLD}"].mean()
                                ),
                                **{
                                    f"twcrps_{t}_mm": float(np.nanmean(twcrps[arm][t]))
                                    for t in TWCRPS_THRESHOLDS
                                },
                                "qm_variant": args.qm_variant if prefix else "",
                            }
                        )
                        metric_rows.append(row)
                        reliability_sums.setdefault((arm, threshold, lead_hours), []).append(
                            reliability_day_sums(probs, obs_binary)
                        )
                        pooled_per_day.setdefault((arm, threshold, lead_hours), []).append(frame)

                    if not prefix:
                        print(
                            f"[lead {lead_hours:>3}h | fold {split_label} | t={threshold:>5}mm] "
                            f"fallback bins unfitted: "
                            f"{sorted(set(RAIN_BIN_LABELS) - set(fitable))} | "
                            f"zero-prob cells: csgd_only={int(np.sum(baseline_probs == 0.0))} "
                            f"tail={int(np.sum(tail_probs == 0.0))}"
                        )

            # Raw IFS-ENS member-counting benchmark: the arm step 12 is measured
            # against, scored with the same helpers as the arms above so only
            # the forecast differs, never the scoring. Once per fold and
            # threshold: it used to sit inside the per-arm, per-threshold loop
            # above, which wrote every benchmark row four times over.
            if ifs_available and ifs_ensemble is not None:
                ifs_test = ifs_ensemble.isel(sample=test_mask)
                ifs_mean = ifs_test.mean(dim="member", skipna=True)
                ifs_guard = member_counting_probabilities(
                    ifs_ensemble, BRIER_GUARD_THRESHOLD, test_mask
                )
                ifs_twcrps = {
                    t: ensemble_twcrps_per_day(ifs_test, test_obs, t) for t in TWCRPS_THRESHOLDS
                }
                for threshold in EXTREME_THRESHOLDS:
                    bench_probs = member_counting_probabilities(
                        ifs_ensemble, threshold, test_mask
                    )
                    row = benchmark_row(
                        bench_probs,
                        obs_values,
                        climatology_probability[threshold].values,
                        threshold,
                        lead_hours,
                        split_label,
                        split_kind,
                        int(train_mask.sum()),
                        int(test_mask.sum()),
                    )
                    frame = _arm_frame(
                        ifs_mean, test_obs, split_label, threshold, bench_probs, ifs_guard,
                        ifs_twcrps,
                    )
                    row.update(
                        {
                            f"brier_{BRIER_GUARD_THRESHOLD}": float(
                                frame[f"brier_{BRIER_GUARD_THRESHOLD}"].mean()
                            ),
                            **{
                                f"twcrps_{t}_mm": float(np.nanmean(ifs_twcrps[t]))
                                for t in TWCRPS_THRESHOLDS
                            },
                            "qm_variant": "",
                        }
                    )
                    metric_rows.append(row)
                    arm = "raw_ifs_ens_member_counting"
                    obs_binary = exceedance_event(obs_values, threshold)
                    reliability_sums.setdefault((arm, threshold, lead_hours), []).append(
                        reliability_day_sums(bench_probs, obs_binary)
                    )
                    pooled_per_day.setdefault((arm, threshold, lead_hours), []).append(frame)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    reliability_rows = []
    for (arm, threshold, lead_hours), parts in reliability_sums.items():
        sums = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        table = reliability_with_ci(sums, args.block_days, args.n_resamples, args.seed)
        reliability_rows.append(table.assign(arm=arm, threshold=threshold, lead_hours=lead_hours))
    reliability_columns = [
        "lead_hours", "arm", "threshold", "bin", "p_lo", "p_hi", "n",
        "mean_forecast", "observed_frequency", "sum_forecast", "sum_observed",
        "observed_frequency_ci_lo", "observed_frequency_ci_hi",
    ]
    pd.concat(reliability_rows, ignore_index=True)[reliability_columns].to_csv(
        args.reliability_out_csv, index=False
    )
    print(f"Wrote {args.reliability_out_csv}")

    pooled = {key: pd.concat(frames, ignore_index=True) for key, frames in pooled_per_day.items()}
    for (arm, threshold, lead_hours), frame in pooled.items():
        write_per_day_scores(
            f"evt_{arm}_t{threshold:g}",
            lead_hours,
            frame,
            out_dir=args.results_dir,
        )

    paired_rows = []
    for arm_a, arm_b in COMPARISONS:
        for lead_hours in LEAD_HOURS:
            lead_rows = []
            for threshold in EXTREME_THRESHOLDS:
                a_days = pooled.get((arm_a, threshold, lead_hours))
                b_days = pooled.get((arm_b, threshold, lead_hours))
                if a_days is None or b_days is None:
                    continue
                lead_rows.append(
                    {
                        "comparison": f"{arm_a}_vs_{arm_b}",
                        "lead_hours": lead_hours,
                        "threshold": threshold,
                        **paired_arm_comparison(
                            a_days, b_days, threshold,
                            args.block_days, args.n_resamples, args.seed,
                        ),
                    }
                )
            # H7's per-lead columns (the names tail_repair_qm_paired.csv uses):
            # tw-CRPS@64.5 and the Brier@7.5 guard are lead-level, SEDI comes
            # from the 115.6 mm row. Repeated on both threshold rows.
            sedi_row = next((r for r in lead_rows if r["threshold"] == H7_SEDI_THRESHOLD), None)
            for r in lead_rows:
                if sedi_row is None:
                    continue
                r.update(
                    {
                        f"sedi_{H7_SEDI_THRESHOLD}_delta": sedi_row["sedi_delta"],
                        f"sedi_{H7_SEDI_THRESHOLD}_ci_lo": sedi_row["sedi_ci_lo"],
                        f"sedi_{H7_SEDI_THRESHOLD}_ci_hi": sedi_row["sedi_ci_hi"],
                        f"sedi_{H7_SEDI_THRESHOLD}_improved": sedi_row["sedi_improved"],
                        "h7_lead_improved": bool(
                            r["twcrps_64.5_improved"] or sedi_row["sedi_improved"]
                        ),
                    }
                )
            paired_rows.extend(lead_rows)

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.paired_out_csv, index=False)
    print(f"Wrote {args.paired_out_csv}")

    print("\n" + "=" * 78)
    print("H7 (EVT part) -- tw-CRPS@64.5 or SEDI@115.6 improves at >= 3 of 5 leads")
    print("(CI excluding 0) and Brier@7.5 is not significantly worse at any lead.")
    print("=" * 78)
    for comparison, subset in paired[paired["threshold"] == H7_SEDI_THRESHOLD].groupby(
        "comparison", sort=False
    ):
        improved = int(subset["h7_lead_improved"].sum())
        degraded = int(subset[f"brier_{BRIER_GUARD_THRESHOLD}_degraded"].sum())
        passes = improved >= MIN_LEADS_FOR_PASS and degraded == 0
        print(
            f"  {comparison:>48}: {improved}/{len(subset)} leads improved, "
            f"{degraded} with Brier@7.5 worse -> {'PASS' if passes else 'FAIL'}"
        )
    for threshold in EXTREME_THRESHOLDS:
        zero_base = metrics[
            (metrics["threshold"] == threshold) & (metrics["arm"] == "csgd_only")
        ]["false_zero_cells"]
        zero_tail = metrics[
            (metrics["threshold"] == threshold) & (metrics["arm"] == "csgd_gpd_tail")
        ]["false_zero_cells"]
        print(
            f"  t={threshold:>5}mm: zero-probability cells {int(zero_base.sum())} -> "
            f"{int(zero_tail.sum())} "
            f"({'eliminated' if bool((zero_tail == 0).all()) else 'REMAIN'})"
        )
    benchmark_rows = [r for r in metric_rows if r["arm"] == "raw_ifs_ens_member_counting"]
    if ifs_available and not benchmark_rows:
        raise RuntimeError(
            "IFS-ENS stores were supplied but the member-counting benchmark "
            "produced no rows. The benchmark is either silently skipped or "
            "broken; refusing to report the run as complete."
        )
    if not ifs_available:
        print(
            "\nNote: the raw IFS-ENS member-counting benchmark is not included. Supply "
            "--ifs-ensemble-stores to add it; it is omitted rather than faked."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
