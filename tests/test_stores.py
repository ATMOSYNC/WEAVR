import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.stores import open_multi_season


def _write_season(path, year, latitude=(6.5, 6.75), time_dim="time"):
    ds = xr.Dataset(
        {"rain": ((time_dim, "latitude", "longitude"), np.ones((2, 2, 2)))},
        coords={
            time_dim: pd.date_range(f"{year}-06-01", periods=2),
            "latitude": list(latitude),
            "longitude": [66.5, 66.75],
        },
    )
    ds.to_zarr(path, group="graphcast", mode="w")


@pytest.mark.parametrize("time_dim", ["time", "nominal_time"])
def test_open_multi_season_combines_matching_grids(tmp_path, time_dim):
    first, second = tmp_path / "2018.zarr", tmp_path / "2020.zarr"
    _write_season(first, 2018, time_dim=time_dim)
    _write_season(second, 2020, time_dim=time_dim)

    combined = open_multi_season([first, second], "graphcast")
    assert combined.sizes[time_dim] == 4
    assert pd.DatetimeIndex(combined[time_dim].values).year.tolist() == [2018, 2018, 2020, 2020]
    combined.close()


def test_open_multi_season_rejects_grid_mismatch(tmp_path):
    first, second = tmp_path / "2018.zarr", tmp_path / "2020.zarr"
    _write_season(first, 2018)
    _write_season(second, 2020, latitude=(6.5, 7.0))

    with pytest.raises(ValueError, match="latitude grid differs"):
        open_multi_season([first, second], "graphcast")


def test_open_multi_season_rejects_overlapping_seasons(tmp_path):
    first, second = tmp_path / "first.zarr", tmp_path / "second.zarr"
    _write_season(first, 2018)
    _write_season(second, 2018)

    with pytest.raises(ValueError, match="overlap"):
        open_multi_season([first, second], "graphcast")


def test_resolve_store_paths_precedence(tmp_path):
    from weavr.stores import resolve_store_paths

    # 1. Explicit legacy path
    assert resolve_store_paths(legacy_single_path="legacy.zarr") == ["legacy.zarr"]

    # 2. Explicit specified paths
    assert resolve_store_paths(specified_paths=["a.zarr", "b.zarr"]) == ["a.zarr", "b.zarr"]

    # 3. All defaults present -> all of them
    f1 = tmp_path / "f1.zarr"
    f2 = tmp_path / "f2.zarr"
    f1.mkdir()
    f2.mkdir()
    res = resolve_store_paths(default_multi_paths=[f1, f2])
    assert res == [f1, f2]

    # 3b. Only SOME defaults present -> refuses, instead of silently using the
    # subset. This is the case that scored a whole Tier 2 fold as NaN.
    import pytest

    with pytest.raises(FileNotFoundError, match="partial multi-season"):
        resolve_store_paths(default_multi_paths=[f1, tmp_path / "nonexistent.zarr"])

    # 4. No defaults, legacy store present -> refuses unless explicitly allowed
    leg = tmp_path / "leg.zarr"
    leg.mkdir()
    import pytest

    with pytest.raises(FileNotFoundError, match="legacy single-season store"):
        resolve_store_paths(
            default_multi_paths=[tmp_path / "missing1.zarr"],
            legacy_fallback_path=leg,
        )

    # ...and the opt-in still works, for a genuinely single-season run
    res2 = resolve_store_paths(
        default_multi_paths=[tmp_path / "missing1.zarr"],
        legacy_fallback_path=leg,
        allow_legacy_fallback=True,
    )
    assert res2 == [leg]

    # 5. Nothing on disk at all -> hand back the defaults so the open() fails
    res3 = resolve_store_paths(
        default_multi_paths=[tmp_path / "nope1.zarr", tmp_path / "nope2.zarr"],
        legacy_fallback_path=tmp_path / "noleg.zarr",
    )
    assert res3 == [tmp_path / "nope1.zarr", tmp_path / "nope2.zarr"]


def test_legacy_fallback_no_longer_masks_a_missing_daily_set(tmp_path):
    """The IFS-ENS case: two daily defaults absent, one weekly store present.

    Nothing is partially present -- both daily stores are missing -- so the
    partial-set guard cannot fire. Without the opt-in, falling through to the
    weekly store would hand the run 18 weekly samples in place of 122 daily
    ones for both seasons.
    """
    import pytest

    from weavr.stores import resolve_store_paths

    weekly = tmp_path / "ifs_ens_2020_jjas.zarr"
    weekly.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_store_paths(
            default_multi_paths=[
                tmp_path / "ifs_ens_2018_jjas_daily.zarr",
                tmp_path / "ifs_ens_2020_jjas_daily.zarr",
            ],
            legacy_fallback_path=weekly,
        )
    assert "allow_legacy_fallback=True" in str(excinfo.value)
    assert "ifs_ens_2020_jjas.zarr" in str(excinfo.value)


def test_partial_ifs_ensemble_stores_refuse_to_run(tmp_path, monkeypatch):
    """The exact 2018/2020 IFS-ENS case: only the 2020 store on disk.

    Silently binding IFS-ENS to one season left the fold testing on the other
    season scoring NaN for both EMOS-CSG variants and BMA.
    """
    import pytest

    from weavr.stores import resolve_store_paths

    ifs_2020 = tmp_path / "ifs_ens_2020_jjas_daily.zarr"
    ifs_2020.mkdir()
    defaults = [
        tmp_path / "ifs_ens_2018_jjas_daily.zarr",
        ifs_2020,
    ]

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_store_paths(default_multi_paths=defaults)
    message = str(excinfo.value)
    assert "ifs_ens_2018_jjas_daily.zarr" in message
    assert "ifs_ens_2020_jjas_daily.zarr" in message

    # Scoping the run deliberately to the season on hand still works.
    assert resolve_store_paths(specified_paths=[ifs_2020]) == [ifs_2020]


def test_missing_store_message_names_how_to_scope_the_run(tmp_path):
    """The error must be actionable, not just a refusal."""
    import pytest

    from weavr.stores import resolve_store_paths

    present = tmp_path / "present.zarr"
    present.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_store_paths(
            default_multi_paths=[present, tmp_path / "absent.zarr"],
        )
    message = str(excinfo.value)
    assert "--ifs-ensemble-stores" in message or "--baseline-stores" in message
    assert "NaN" in message

