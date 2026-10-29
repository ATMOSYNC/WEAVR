"""Miniature but schema-faithful replicas of every store the runners read.

Why this exists
---------------
A two-season Tier 2 run was lost twice, for two unrelated reasons, and neither
was caught by any test:

1. `data/ifs_ens_2020_jjas_daily.zarr` was 3.4 MB of **zeros** with the correct
   shape, the correct dates and the correct dtype. Zeros are not NaN, so every
   structural check passed and 2h15m of BMA sampling was spent scoring against
   an empty field.
2. The by-bin CSV write raised `ValueError: dict contains fields not in
   fieldnames: 'rmse_mm'` *after* all the expensive numbers existed, because
   fieldnames came from `rows[0]` and the pooled rows carry a column the
   per-fold rows do not. The per-day scores were still buffered in memory and
   went with it.

Both are invisible to a unit test that calls one function, and far too
expensive to find by running the real 2.3 GB stores. So this module builds the
*whole* store set at ~1/1000 scale with the same groups, variable names, dims,
coords and dtypes as the real thing, which lets the real runners be executed
end to end in seconds -- including the fault variants, which is the part that
matters.

Schema fidelity is the whole point, so this mirrors what the real stores
actually contain (verified by reading them), including the asymmetries that
have bitten us: `pangu` has temperature but no precipitation, the lagged store
indexes on `nominal_time`/`lead_hours`/`member_offset_hours` while the baseline
indexes on `time`/`prediction_timedelta`, and the climatology is one group per
year rather than a single variable.

Faults are injectable via :func:`build_all` so each known failure can be
reproduced deliberately rather than waited for.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LEAD_HOURS = [24, 48, 72, 96, 120]
MEMBER_OFFSETS = [-48, -36, -24, -12, 0, 12, 24, 36, 48]
CLIMATOLOGY_YEARS = list(range(2006, 2021))

#: Seasons the synthetic set covers. Two, so `iter_evaluation_folds` produces
#: two genuine leave-one-year-out folds rather than degrading to a block split.
SEASONS = (2018, 2020)

PRECIP = "total_precipitation_24hr"

#: **Units contract.** WeatherBench 2 / ECMWF `total_precipitation_24hr` is in
#: METRES; IMD `rain` is in MILLIMETRES. Every runner multiplies the forecast
#: by 1000 before scoring (`PRECIP_M_TO_MM` in `run_tier0_baseline.py`), and
#: the reason is written into that script's docstring: get it wrong and you get
#: a bias of roughly `+obs_mean` -- a plausible-looking number, not a crash.
#:
#: The replica honours the contract, so every forecast group below is divided
#: by 1000 on the way in. Writing millimetres here is the single easiest way to
#: make this whole suite report nonsense, and it was exactly the mistake made
#: while building it.
_MM_PER_M = 1000.0
TEMP = "2m_temperature"
RAIN = "rain"


@dataclass
class SyntheticSpec:
    """Shape of the miniature store set."""

    n_times: int = 8
    n_members: int = 6
    seed: int = 0
    #: Stride over the real 0.25 deg India grid. 4 keeps the real coordinate
    #: values (so resolution checks behave) while cutting cells ~16x. The grid
    #: MUST span lat 6.5-38.5 / lon 66.5-100: tier 3 selects a monsoon core
    #: zone at lat (18, 28) x lon (65, 88) and raises `EmptyCoreZoneError` on a
    #: grid that does not reach it, which is a real constraint of the product
    #: rather than an artefact of the replica.
    grid_stride: int = 4

    # --- fault switches -----------------------------------------------------
    #: Fill an entire store with zeros: the 2020-IFS-ENS failure.
    zero_ifs_ens_years: tuple[int, ...] = ()
    #: Blank one variable to NaN for the given seasons: a season that scores nan.
    nan_observed_years: tuple[int, ...] = ()
    #: Write only the first season's stores: "one season present", which makes a
    #: two-season run silently degrade to a single block split.
    seasons_written: tuple[int, ...] = SEASONS
    #: Truncate the IFS-ENS time axis, so a store is structurally short.
    truncate_ifs_ens_times: dict[int, int] = field(default_factory=dict)
    #: Drop a whole store entirely.
    omit_stores: tuple[str, ...] = ()
    #: Drop a group that the runners expect.
    omit_groups: tuple[str, ...] = ()
    #: Remove pangu's precipitation... pangu has none by design, so instead:
    #: give pangu a precipitation variable it should not have.
    pangu_has_precip: bool = False

    # --- derived ------------------------------------------------------------
    #: The real store's exact extents, read from
    #: `data/baseline_2018_jjas_daily.zarr/imd_observed`.
    LAT_MIN, LAT_MAX = 6.5, 38.5
    LON_MIN, LON_MAX = 66.5, 100.0
    REAL_N_LAT, REAL_N_LON = 129, 135

    @property
    def n_lat(self) -> int:
        return len(range(0, self.REAL_N_LAT, self.grid_stride))

    @property
    def n_lon(self) -> int:
        return len(range(0, self.REAL_N_LON, self.grid_stride))

    @property
    def lat(self) -> np.ndarray:
        return self.LAT_MIN + 0.25 * np.arange(0, self.REAL_N_LAT, self.grid_stride)

    @property
    def lon(self) -> np.ndarray:
        return self.LON_MIN + 0.25 * np.arange(0, self.REAL_N_LON, self.grid_stride)

    def times(self, year: int) -> pd.DatetimeIndex:
        return pd.date_range(f"{year}-06-01", periods=self.n_times, freq="D")

    def members(self) -> np.ndarray:
        return np.arange(1, self.n_members + 1)


def _rng(spec: SyntheticSpec, salt: int) -> np.random.Generator:
    return np.random.default_rng(spec.seed + salt)


def _precip_field(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """A precipitation-like *truth* field: mostly dry, heavy in a minority.

    Real JJAS stores sit around 85% non-zero, and the rain-bin classifier needs
    a genuine spread across `light`/`heavy`/`very_heavy`/`extremely_heavy` for
    the per-bin and BMA paths to be exercised at all.
    """
    data = rng.gamma(shape=0.6, scale=6.0, size=shape)
    data[rng.random(shape) < 0.15] = 0.0
    flat = data.reshape(-1)
    idx = rng.choice(flat.size, size=max(1, flat.size // 200), replace=False)
    flat[idx] = rng.uniform(120.0, 260.0, size=idx.size)
    return data


def _to_metres(precip_mm: np.ndarray) -> np.ndarray:
    """Millimetres -> metres, to match the WeatherBench 2 convention."""
    return precip_mm / _MM_PER_M


def _perturb(
    rng: np.random.Generator, truth: np.ndarray, noise: float, dry_bias: float = 0.0
) -> np.ndarray:
    """A forecast of `truth`: correlated signal plus multiplicative noise.

    **The sources must be correlated with each other and with the truth.**
    Generating each source as an independent random field -- which is the
    obvious thing to do -- makes the EMOS regression ill-posed: with no
    relationship between forecast and observation, the fitted censored-gamma
    parameters run away, and the rehearsal reports `emos_ifs_ens_rmse_mm` of
    2e13 and a `tier0_bias_mm` of +3954. That is not a bug in the pipeline, it
    is a replica that does not resemble the thing being measured. Real
    forecasts of the same rain field agree to within tens of percent, which is
    precisely what makes EMOS well-conditioned and what the rain-bin
    classifier's spread feature assumes.
    """
    out = truth * (1.0 + rng.normal(0.0, noise, size=truth.shape))
    if dry_bias:
        # Independent chance of a false zero, so the dry fraction exceeds the
        # truth's -- false zeros are the failure mode the whole project cares
        # about, so the replica has to contain them.
        out = np.where(rng.random(truth.shape) < dry_bias, 0.0, out)
    return np.clip(out, 0.0, None)


def _temp_field(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """A temperature-like field in Kelvin, so > 0 everywhere and never zero."""
    return 288.0 + rng.normal(0.0, 4.0, size=shape)


def _baseline_coords(spec: SyntheticSpec, year: int, with_leads: bool) -> dict:
    coords = {
        "time": spec.times(year).values,
        "latitude": spec.lat,
        "longitude": spec.lon,
    }
    if with_leads:
        coords["prediction_timedelta"] = np.array(LEAD_HOURS)
    return coords


def _lagged_coords(spec: SyntheticSpec, year: int) -> dict:
    return {
        "nominal_time": spec.times(year).values,
        "lead_hours": np.array(LEAD_HOURS),
        "member_offset_hours": np.array(MEMBER_OFFSETS),
        "latitude": spec.lat,
        "longitude": spec.lon,
    }


def _write_group(path: Path, group: str, ds: xr.Dataset) -> None:
    """Write one group, with its coords carried in the Dataset itself.

    Coords are embedded rather than added afterwards with a raw zarr write: a
    bare `group["time"] = ...` creates an array with no `dimension_names`
    attribute, and xarray then refuses to open the store at all. Every real
    store here carries full coords per group, so the replica does too.
    """
    if group:
        ds.to_zarr(path, group=group, mode="a")
    else:
        ds.to_zarr(path, mode="w")


def build_baseline(root: Path, spec: SyntheticSpec, year: int) -> Path:
    path = root / f"baseline_{year}_jjas_daily.zarr"
    if "baseline" in spec.omit_stores:
        return path
    if path.exists():
        shutil.rmtree(path)
    shape = (spec.n_times, len(LEAD_HOURS), spec.n_lat, spec.n_lon)

    # One shared truth per (time, lead) field, then each source as a noisy
    # view of it. HRES is the most accurate single source, IFS-ENS mean sits
    # between GraphCast and HRES, which is roughly the real skill ordering and
    # gives the "best source" and independence logic something to rank.
    truth = _precip_field(_rng(spec, 1), shape)
    graphcast_precip = _to_metres(
        _perturb(_rng(spec, 2), truth, noise=0.35, dry_bias=0.05)
    )
    hres_precip = _to_metres(_perturb(_rng(spec, 3), truth, noise=0.25, dry_bias=0.03))
    ifs_mean = _to_metres(_perturb(_rng(spec, 4), truth, noise=0.30, dry_bias=0.04))

    for group, precip in (
        ("graphcast", graphcast_precip),
        ("hres", hres_precip),
        ("ifs_ens_mean", ifs_mean),
    ):
        _write_group(
            path,
            group,
            xr.Dataset(
                {
                    PRECIP: (
                        ("time", "prediction_timedelta", "latitude", "longitude"),
                        precip,
                    ),
                    TEMP: (
                        ("time", "prediction_timedelta", "latitude", "longitude"),
                        _temp_field(_rng(spec, 10 + len(group)), shape),
                    ),
                },
                coords=_baseline_coords(spec, year, with_leads=True),
            ),
        )

    pangu_vars = {
        TEMP: (
            ("time", "prediction_timedelta", "latitude", "longitude"),
            _temp_field(_rng(spec, 20), shape),
        )
    }
    if spec.pangu_has_precip:
        pangu_vars[PRECIP] = (
            ("time", "prediction_timedelta", "latitude", "longitude"),
            _precip_field(_rng(spec, 21), shape),
        )
    _write_group(
        path,
        "pangu",
        xr.Dataset(pangu_vars, coords=_baseline_coords(spec, year, with_leads=True)),
    )

    # Observations share the forecast truth: the 24h-accumulation truth at
    # lead index 0, with observation error on top.
    obs = _perturb(
        _rng(spec, 30), truth[:, 0, :, :], noise=0.30, dry_bias=0.0
    )
    if year in spec.nan_observed_years:
        obs = obs.copy()
        obs[:] = np.nan
    _write_group(
        path,
        "imd_observed",
        xr.Dataset(
            {RAIN: (("time", "latitude", "longitude"), obs)},
            coords=_baseline_coords(spec, year, with_leads=False),
        ),
    )
    return path


def build_lagged(root: Path, spec: SyntheticSpec, year: int) -> Path:
    """The lagged ensemble store: `nominal_time` / `lead_hours` / offsets.

    Its axis names differ from the baseline store's, which is exactly why it
    gets its own builder here -- a replica that used the baseline's names would
    pass tests and then fail against the real thing.
    """
    path = root / f"lagged_ensemble_inputs_{year}_jjas_daily.zarr"
    if "lagged" in spec.omit_stores:
        return path
    if path.exists():
        shutil.rmtree(path)
    dims = ("nominal_time", "lead_hours", "member_offset_hours", "latitude", "longitude")
    shape = (spec.n_times, len(LEAD_HOURS), len(MEMBER_OFFSETS), spec.n_lat, spec.n_lon)

    # Same correlated-truth construction as the baseline, and METRES like every
    # other forecast store. `load_graphcast_ensemble` multiplies by
    # `PRECIP_M_TO_MM`, so a lagged store left in millimetres is silently
    # multiplied by 1000 and Tier 2's `tier0` columns read ~4000 mm -- which is
    # precisely the bug this replica had while being written.
    lagged_truth = _precip_field(_rng(spec, 40), shape)
    graphcast_precip = _to_metres(
        _perturb(_rng(spec, 43), lagged_truth, noise=0.35, dry_bias=0.05)
    )

    _write_group(
        path,
        "graphcast",
        xr.Dataset(
            {
                PRECIP: (dims, graphcast_precip),
                TEMP: (dims, _temp_field(_rng(spec, 41), shape)),
            },
            coords=_lagged_coords(spec, year),
        ),
    )
    _write_group(
        path,
        "pangu",
        xr.Dataset(
            {TEMP: (dims, _temp_field(_rng(spec, 42), shape))},
            coords=_lagged_coords(spec, year),
        ),
    )
    return path




def build_ifs_ens(root: Path, spec: SyntheticSpec, year: int) -> Path:
    path = root / f"ifs_ens_{year}_jjas_daily.zarr"
    if "ifs_ens" in spec.omit_stores:
        return path
    if path.exists():
        shutil.rmtree(path)
    n_times = spec.truncate_ifs_ens_times.get(year, spec.n_times)
    shape = (n_times, spec.n_members, len(LEAD_HOURS), spec.n_lat, spec.n_lon)
    if year in spec.zero_ifs_ens_years:
        data = np.zeros(shape, dtype=np.float64)  # the real failure (metres)
    else:
        # Members must be tight around a shared truth, because that spread is
        # exactly the EMOS/CSGD conditioning variable. Independent members
        # would give a meaningless spread.
        member_truth = _precip_field(_rng(spec, 50 + year), shape)
        data = np.empty(shape, dtype=np.float64)
        for m in range(spec.n_members):
            data[:, m] = _to_metres(
                _perturb(_rng(spec, 50 + year * 10 + m), member_truth[:, m], noise=0.30)
            )

    ds = xr.Dataset(
        {
            PRECIP: (
                ("time", "member", "prediction_timedelta", "latitude", "longitude"),
                data,
            )
        },
        coords={
            "time": spec.times(year).values[:n_times],
            "member": spec.members(),
            "prediction_timedelta": np.array(LEAD_HOURS),
            "latitude": spec.lat,
            "longitude": spec.lon,
        },
    )
    ds.to_zarr(path, mode="w")
    return path


def build_climatology(root: Path, spec: SyntheticSpec) -> Path:
    path = root / "imd_seeps_climatology_jjas.zarr"
    if "climatology" in spec.omit_stores:
        return path
    if path.exists():
        shutil.rmtree(path)
    shape = (spec.n_times, spec.n_lat, spec.n_lon)
    for year in CLIMATOLOGY_YEARS:
        data = _precip_field(_rng(spec, 60 + year), shape)
        data = np.clip(data, 0.0, None)
        _write_group(
            path,
            f"y{year}",
            xr.Dataset(
                {RAIN: (("time", "latitude", "longitude"), data)},
                coords={
                    "time": pd.date_range(
                        f"{year}-06-01", periods=spec.n_times, freq="D"
                    ).values,
                    "latitude": spec.lat,
                    "longitude": spec.lon,
                },
            ),
        )
    return path


def build_all(root: Path, spec: SyntheticSpec | None = None) -> dict[str, Path]:
    """Build the full miniature store set and return the paths runners need."""
    spec = spec or SyntheticSpec()
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for year in spec.seasons_written:
        out[f"baseline_{year}"] = build_baseline(root, spec, year)
        out[f"lagged_{year}"] = build_lagged(root, spec, year)
        out[f"ifs_ens_{year}"] = build_ifs_ens(root, spec, year)
    out["climatology"] = build_climatology(root, spec)
    return out


#: Which store flag names which synthetic path key.
STORE_FLAGS: dict[str, str] = {
    "baseline": "--baseline-stores",
    "lagged": "--lagged-stores",
    "ifs_ens": "--ifs-ensemble-stores",
}


def store_args(
    paths: dict[str, Path],
    seasons: tuple[int, ...] = SEASONS,
    kinds: tuple[str, ...] = ("baseline", "lagged", "ifs_ens"),
) -> list[str]:
    """CLI fragments naming the stores a runner needs, in the runner's own flags.

    `kinds` exists because runners differ: Tier 0 reads only the baseline
    store, Phase 2 only the lagged one, and passing a flag a runner does not
    define is an immediate argparse exit 2.
    """
    args: list[str] = []
    for kind in kinds:
        flag = STORE_FLAGS.get(kind)
        if flag is None:
            continue
        present = [
            str(paths[f"{kind}_{y}"]) for y in seasons if f"{kind}_{y}" in paths
        ]
        if present:
            args += [flag, *present]
    return args
