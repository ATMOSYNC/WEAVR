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

## Daily accumulation window: IMD's 03 UTC → 03 UTC, labelled by its end date

This is IMD's own observation-day convention (03:00 UTC = 08:30 IST), not
UTC midnight-to-midnight. **IMD labels a day by the date its window ends
on.** IMD's rainfall information page tags "Rainfall recorded from 0830 hrs
of 21-10-2026 to 0830 hrs of 22-10-2026" as the value for **22-10-2026**
(<https://mausam.imd.gov.in/responsive/rainfallinformation_msd.php>). So a
day labelled `2024-06-15` covers `2024-06-14T03:00 UTC` through
`2024-06-15T03:00 UTC`. Every other source (WeatherBench 2, ECMWF open data)
must be resampled onto this window, not the other way around — IMD is the
ground truth we verify against (§3.2/§3.6 of the research brief), so its
convention is the one that must not silently drift.

> **Correction (issue #106).** This section used to say a day is labelled by
> the date its window *starts* on. That was wrong, and every forecast was
> paired with the IMD day *before* the one it covers: a 24 h forecast from
> 00 UTC on day `t` (window 00 UTC `t` → 00 UTC `t+1`) was scored against IMD
> day `t`, which covers 03 UTC `t−1` → 03 UTC `t` and shares only 3 of its 24
> hours. The matching day is `t+1` (21 h overlap). The evidence is in the
> issue: on the 2020 store the forecast–observation day-to-day correlation
> peaks one day later than the old pairing for GraphCast, HRES and the IFS
> mean at 24 h and 72 h (GraphCast 24 h: 0.40 → 0.51).

### Pairing a forecast with an IMD day

There is exactly one definition, `weavr.grid.imd_day_for(init_time,
lead_hours)`, and every runner calls it. `total_precipitation_24hr` valid at
`init + lead` is the accumulation over the 24 h ending at that instant; the
IMD day it validates on is the one whose window ends at or just after that
instant. So each extra lead day moves the label one day later, and a season's
last `lead/24` init days have no observation to score against (an init-day
count of 122 gives 121 rows at 24 h, 117 at 120 h). `imd_day_containing(t)`
gives the label of the window that contains a single instant (used to bucket
events such as monsoon-depression track fixes); windows are half-open,
`[03:00, next 03:00)`.

`tests/test_imd_day_pairing_real_store.py` asserts, on the real store, that
the forecast–observation correlation peaks at offset 0 under this pairing.
That is the check that catches a wrong reading of the label.

`resample_to_imd_day()` sums (not averages) over each 24h window, and
labels the resulting day by the date the window *ends* on, matching IMD.

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
