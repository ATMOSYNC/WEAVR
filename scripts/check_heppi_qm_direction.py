#!/usr/bin/env python3
"""HEPPI-UQM directional check (step 11 item 3, #73).

Fit weavr's empirical per-region quantile map on NEPS-G raw 2018 against IMD
2018, apply it to NEPS-G raw 2019, and compare quantiles of the result with
HEPPI's own UQM output for 2019, region by region. Directional, not a
reproduction: the question is whether both corrections move each quantile the
same way. Results and their reading: docs/tail-repair-results.md section 4.3.

HEPPI is third-party data (https://edata.bham.ac.uk/698/) and is not in this
repository. Only aggregate statistics per region x quantile are written,
never a grid.

Usage: python scripts/check_heppi_qm_direction.py HEPPI_DIR [OUT_CSV]
       (default OUT_CSV: results/heppi_directional_check.csv)
"""

import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
if len(sys.argv) < 2:
    sys.exit(__doc__)
heppi_dir = Path(sys.argv[1])
out_csv = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "results" / "heppi_directional_check.csv"
)
sys.path.insert(0, str(repo / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from weavr.data.heppi_reference import load_imd_observed, load_ncmrwf_forecast  # noqa: E402
from weavr.quantile_mapping import apply_regional_quantile_maps, fit_regional_quantile_maps  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402

DATE_MAP = repo / "docs" / "heppi-date-map.csv"
QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
WET_MM = 0.1


def by_year(da, year):
    t = pd.DatetimeIndex(da["time"].values)
    return da.isel(sample=np.where(t.year == year)[0])


obs = load_imd_observed(heppi_dir / "IMD_observed.nc", DATE_MAP).load()
raw = load_ncmrwf_forecast(heppi_dir, "orig", DATE_MAP)
uqm = load_ncmrwf_forecast(heppi_dir, "uqm", DATE_MAP)
for name, da in (("obs", obs), ("raw", raw), ("uqm", uqm)):
    t = pd.DatetimeIndex(da["time"].values)
    print(
        name,
        dict(da.sizes),
        "years",
        sorted(set(t.year)),
        "per year",
        pd.Series(t.year).value_counts().to_dict(),
    )

regions = assign_regions(obs["lat"].values, obs["lon"].values)
regions = regions.rename({"latitude": "lat", "longitude": "lon"})

obs18, obs19 = by_year(obs, 2018), by_year(obs, 2019)
raw18 = by_year(raw, 2018).load().rename({"lat": "latitude", "lon": "longitude"})
obs18_ = obs18.rename({"lat": "latitude", "lon": "longitude"})
reg_ = regions.rename({"lat": "latitude", "lon": "longitude"})
land = np.isfinite(obs18.values).any(axis=0)
reg_ = reg_.where(land, "")
maps = fit_regional_quantile_maps(
    raw18, obs18_, reg_, member_dim="member", wet_threshold=WET_MM, source="neps_g_raw"
)
maps.pop("", None)
print("fitted regions:", sorted(maps))
del raw18

raw19 = by_year(raw, 2019).load()
ours19 = apply_regional_quantile_maps(
    maps, raw19.rename({"lat": "latitude", "lon": "longitude"}), reg_
)
uqm19 = by_year(uqm, 2019).load()
print(
    "2019 days: obs",
    obs19.sizes["sample"],
    "raw",
    raw19.sizes["sample"],
    "uqm",
    uqm19.sizes["sample"],
)
assert (obs19["time"].values == raw19["time"].values).all() and (
    raw19["time"].values == uqm19["time"].values
).all()

rows = []
reg_vals = reg_.values
obs_ok = np.isfinite(obs19.values)  # (sample, lat, lon)
for region in sorted(maps):
    cell = reg_vals == region
    m = obs_ok[:, cell]  # (sample, ncell): only cell-days with an observation
    o = obs19.values[:, cell][m]

    def pick(a):
        return np.moveaxis(a[:, :, cell], 1, -1)[m].ravel()  # members pooled

    series = {
        "obs": o,
        "raw": pick(raw19.values),
        "weavr_qm": pick(np.asarray(ours19)),
        "heppi_uqm": pick(uqm19.values),
    }
    for name, v in series.items():
        v = v[np.isfinite(v)]
        rows.append(
            {
                "region": region,
                "series": name,
                "stat": "wet_fraction",
                "value": float(np.mean(v >= WET_MM)),
                "n": v.size,
            }
        )
        for q in QUANTILES:
            rows.append(
                {
                    "region": region,
                    "series": name,
                    "stat": f"q{q:g}",
                    "value": float(np.quantile(v, q)),
                    "n": v.size,
                }
            )

df = pd.DataFrame(rows)
wide = df.pivot_table(index=["region", "stat"], columns="series", values="value").reset_index()
wide["weavr_shift"] = wide.weavr_qm - wide.raw
wide["heppi_shift"] = wide.heppi_uqm - wide.raw
tol = 0.01 * wide.raw.abs().clip(lower=0.1)  # a shift under 1% of the raw value counts as "no move"


def sgn(s):
    return np.where(s.abs() <= tol, 0, np.sign(s))


wide["same_direction"] = sgn(wide.weavr_shift) == sgn(wide.heppi_shift)
wide["weavr_closer_to_obs"] = (wide.weavr_qm - wide.obs).abs() < (wide.heppi_uqm - wide.obs).abs()
wide["raw_abs_err"] = (wide.raw - wide.obs).abs()
wide["weavr_abs_err"] = (wide.weavr_qm - wide.obs).abs()
wide["heppi_abs_err"] = (wide.heppi_uqm - wide.obs).abs()
out_csv.parent.mkdir(parents=True, exist_ok=True)
wide.to_csv(out_csv, index=False)
pd.set_option("display.width", 200)
print(wide.round(3).to_string(index=False))
print("\nsame direction:", int(wide.same_direction.sum()), "/", len(wide))
for stat, g in wide.groupby("stat"):
    print(
        f"{stat:13s} same dir {int(g.same_direction.sum())}/{len(g)}  "
        f"mean |err| raw {g.raw_abs_err.mean():.2f} weavr {g.weavr_abs_err.mean():.2f} "
        f"heppi {g.heppi_abs_err.mean():.2f}"
    )
