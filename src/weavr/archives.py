"""WeatherBench 2 archive selection for the two evaluation seasons."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class Archive:
    path: str
    coordinate_rename: tuple[tuple[str, str], ...] = ()


_ROOT = "gs://weatherbench2/datasets"
ARCHIVES: dict[tuple[str, int], Archive] = {
    ("graphcast", 2018): Archive(
        f"{_ROOT}/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours_derived.zarr",
        (("lat", "latitude"), ("lon", "longitude")),
    ),
    ("graphcast", 2020): Archive(
        f"{_ROOT}/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr",
        (("lat", "latitude"), ("lon", "longitude")),
    ),
}
for _year in (2018, 2020):
    ARCHIVES[("pangu", _year)] = Archive(f"{_ROOT}/pangu/2018-2022_0012_0p25.zarr")
    ARCHIVES[("hres", _year)] = Archive(f"{_ROOT}/hres/2016-2022-0012-1440x721.zarr")
    ARCHIVES[("ifs_ens_mean", _year)] = Archive(
        f"{_ROOT}/ifs_ens/2018-2022-1440x721_mean.zarr"
    )
    ARCHIVES[("ifs_ens", _year)] = Archive(f"{_ROOT}/ifs_ens/2018-2022-1440x721.zarr")


def archive_for(source: str, year: int) -> Archive:
    """Return the exact archive and coordinate convention for a source/year."""
    try:
        return ARCHIVES[(source, year)]
    except KeyError as exc:
        raise ValueError(f"No WeatherBench 2 archive for {source!r} in {year}") from exc


def normalize_coordinates(ds: xr.Dataset, archive: Archive) -> xr.Dataset:
    """Rename native spatial coordinates before slicing and regridding."""
    rename = dict(archive.coordinate_rename)
    missing = [name for name in rename if name not in ds.coords]
    if missing:
        raise ValueError(f"Archive is missing expected coordinates: {missing}")
    normalized = ds.rename(rename) if rename else ds
    for name in ("latitude", "longitude"):
        if name not in normalized.coords:
            raise ValueError(f"Archive is missing expected coordinate {name!r}")
    return normalized
