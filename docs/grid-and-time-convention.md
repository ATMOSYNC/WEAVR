# Grid and time convention

Locked decision, implemented in [`src/weavr/grid.py`](../src/weavr/grid.py).
Every ingestion script must call `regrid_to_common` / `resample_to_imd_day`
from that module rather than reimplementing regridding or resampling
inline — this is exactly the kind of thing that silently diverges between
scripts if left to each author's judgement.

## Common grid: 0.25° lat/lon, bounds 6.5–38.5N / 66.5–100.0E

This matches IMD gridded rainfall's own native resolution and the
resolution WeatherBench 2's regridded products are served at, so the two
primary data sources ([docs/data-sources.md](data-sources.md)) need no
further interpolation to compare against each other.

The bounds are **not** the rough 6–38N/68–98E estimate a first pass at this
decision might reach for — they're taken directly from a real IMD
gridded-rainfall pull (`imdlib`, `var_type="rain"`, year 2023), which
returned exactly `lat 6.5–38.5, lon 66.5–100.0` at 0.25° spacing. Using
IMD's actual grid rather than an approximation means regridding onto it
needs no further interpolation at the domain edges.

`regrid_to_common()` refuses (`SourceTooCoarseError`) to regrid a source
whose native resolution is coarser than 0.25° — upsampling a coarser source
would create the illusion of matching resolution while hiding real skill
differences behind interpolation artifacts. If a genuinely coarser source
needs to be included later, it should go through an explicit, documented
area-weighted remap (e.g. `xesmf`), not silently through this function.

## Daily accumulation window: IMD's 03 UTC → 03 UTC (next day)

This is IMD's own observation-day convention (03:00 UTC = 08:30 IST), not
UTC midnight-to-midnight. A day labelled `2024-06-15` covers
`2024-06-15T03:00 UTC` through `2024-06-16T03:00 UTC`. Every other source
(WeatherBench 2, ECMWF open data) must be resampled onto this window, not
the other way around — IMD is the ground truth we verify against
(§3.2/§3.6 of the research brief), so its convention is the one that must
not silently drift.

`resample_to_imd_day()` sums (not averages) over each 24h window, and
labels the resulting day by the date the window *starts* on — matching how
IMD itself labels a day by its 03 UTC start, not by midnight UTC.

**Implementation note**: `xarray.Dataset.resample(freq="1D", offset=...)`
does *not* actually shift calendar-day bin boundaries in the xarray/pandas
versions this was tested against — the `offset` kwarg is silently ignored,
which would produce midnight-UTC-aligned bins while claiming to be
03-UTC-aligned. This was caught by a unit test (`test_grid.py`) checking an
exact 03:00 UTC-aligned day's sum, which failed on the naive
`resample(..., offset=...)` implementation. The fix shifts the time
coordinate back by the offset before resampling to `"1D"`, then relabels —
verified by hand against three separate cases (full days, exact boundary,
partial edge windows) in the test suite.

## Why this matters

Grid/time misalignment is a common source of silent errors: two sources
that look aligned (both "daily", both "0.25°") can still be comparing
different physical windows or different physical points, producing skill
differences that reflect the alignment bug rather than real model
performance. Locking this in one place, with tests that fail loudly on a
misimplementation, is cheaper than re-discovering the bug downstream in a
skill-score comparison.
