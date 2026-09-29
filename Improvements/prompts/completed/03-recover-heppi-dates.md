# Step 03: Recover HEPPI's missing calendar

**Type**: data-recovery prompt. Uses local data only; runs in minutes.
**Plan items**: A3 part 1 (plan §2 F4).
**Depends on**: nothing. `HEPPI/` and `data/imd_seeps_climatology_jjas.zarr`
already exist locally.

## Goal

HEPPI pairs **NCMRWF's NEPS-G 23-member ensemble** with IMD observations on
WEAVR's exact 129×135 grid. It was rejected as training data in
`docs/phase4-data-and-combiner-scope.md` only because its 334 samples have
no dates.

The plan's F4 found the dates are recoverable. Each HEPPI observation field
was matched against all 1,830 JJAS days in the local 2006–2020 IMD archive:

- HEPPI indices **0–119** → 2018-06-02 … 2018-09-30, skipping 2018-06-24
  (HEPPI's own `verif_hitmiss.m` mentions a missing 24 June).
- Indices **181–302** → 2019-06-01 … 2019-09-30.
- Under that mapping, the hypothesised date is the single best match for
  **241 of 242** days. Shifting by ±1 day drops that to 1 of 241 and 0 of
  240.

Make this a committed, tested, reproducible date map, and attach real dates
in WEAVR's HEPPI loader.

## Why this matters

- It is the only route to India's own operational ensemble in this project.
  It makes slides 1–2's NEPS-G claim *true* (after step 08) instead of
  deleted.
- It enables a head-to-head against a **published** benchmark: HEPPI's
  `NCMRWF_EMOS_forecast.nc` is Angus et al. 2024's post-processing (step 08).
- Indices 120–180 and 303–333 (92 samples) are probably October–November
  2018 and October 2019, since 181 + 153 = 334 fits exactly. That can be
  confirmed cheaply with IMD's full-year archive instead of left as a guess.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/ (HEPPI is at
../HEPPI/):

1. Add scripts/recover_heppi_dates.py.
   - Load HEPPI/IMD_observed.nc (variable IMD_rainfall_observed, dims
     forecast x lon x lat; transpose to forecast x lat x lon) and every
     y<year> group of data/imd_seeps_climatology_jjas.zarr.
   - For each HEPPI index, compute the mean absolute difference against
     every candidate IMD day, over cells valid in both (NaN = -1 masks are
     fine; the plan's appendix A shows the method).
   - Record: best_match_date, best_mae, second_best_mae.
   - Test the hypothesised mapping (plan section 2 F4) at offsets -1, 0 and
     +1 day, and report how often the hypothesised date is the unique best
     match. Expected about 241/242 at offset 0 (verify).
   - For indices 120-180 and 303-333: fetch IMD full-year 2018 and 2019
     with weavr.data.imd_gridded.fetch_year (cached, about 51 MB/year) and
     test the "Oct-Nov 2018 / Oct 2019" hypothesis the same way. Report
     what it shows; if it fails, leave those indices undated.
   - Write docs/heppi-date-map.csv with columns: heppi_index, date (empty
     if unconfirmed), best_match_date, best_mae, second_best_mae,
     rank_of_mapped_date, confirmed (bool).
   - Note in the doc that values are close but not bit-identical, most
     likely because IMD revises its gridded product. So IMD obs used for
     scoring must still come from imdlib; HEPPI supplies only the NEPS-G
     FORECASTS.

2. Establish forecast validity. The date map proves which day each
   OBSERVATION index is. Check that forecast index i verifies against obs
   index i, not i+1 or i-1:
   - Compute the correlation of the NCMRWF_orig_forecast ensemble mean
     (load lazily; the files are about 1 GB each) with IMD obs at index
     offsets -2..+2.
   - Report the result.
   The data cannot reveal the forecast LEAD (day-1 vs day-2). State the
   lead as an assumption, citing Angus et al. 2024 (day-1 NEPS-G) and
   HEPPI_readme.txt, and flag it as such everywhere it is used.

3. Extend src/weavr/data/heppi_reference.py.
   - Add load_date_map(csv_path) and an optional date_map argument to
     load_imd_observed / load_ncmrwf_forecast. It attaches a real `time`
     coordinate, and by default drops samples with no confirmed date.
   - Keep the existing HeppiGridMismatchError guard.
   - Update the `note` attribute text so it no longer says the dates are
     unknown when a map is applied.
   - Extend tests/data/test_heppi_reference.py with synthetic fixtures:
     dates attach correctly, unconfirmed samples are dropped, and a date
     map of the wrong length raises.

4. Docs:
   - Update docs/heppi-reference-dataset.md's "Known gap: no calendar
     dates" to "resolved", with the method and confidence numbers.
   - In docs/phase4-data-and-combiner-scope.md, append a dated note under
     the HEPPI decision saying the premise no longer holds, and link the
     new section. Don't rewrite the original decision text; it was correct
     at the time.
   - README paragraph.

5. Licence:
   - Commit ONLY the script and the derived date map. Never commit HEPPI
     .nc data or HEPPI-derived grids (convention 7).
   - In the PR description, include a short DRAFT email the user may send
     to the HEPPI authors (Angus et al., University of Birmingham / WCSSP
     India) asking them to confirm the recovered dates and the dataset's
     redistribution terms. Do NOT send it.
```

## Done when

- `docs/heppi-date-map.csv` exists, produced by a committed script, with
  confidence columns.
- The JJAS 2018 and 2019 mapping is confirmed (about 241/242, or the real
  number).
- The non-JJAS indices are confirmed or honestly left undated.
- Forecast/obs index pairing is checked.
- `weavr.data.heppi_reference` attaches real dates, with tests.
- The docs no longer describe HEPPI as undatable.
