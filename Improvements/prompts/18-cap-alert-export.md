# Step 18: CAP 1.2 alert export

**Type**: implementation prompt (export only; nothing is ever sent).
**Plan items**: C3.
**Depends on**: step 14 (district warnings) and step 17 (the district card
the download button lives on).

## Goal

Export every WEAVR district warning (Yellow/Orange/Red) as a
**Common Alerting Protocol v1.2** XML `<alert>`, validated against the
official OASIS CAP 1.2 schema. Download it from the district card.

The pitch point: WEAVR's output is in the standard format alerting systems
ingest. Plan §4 C3 notes that India's national Integrated Alert System
(NDMA's SACHET) is CAP-based. **Verify that from NDMA's public
documentation before saying it on a slide.** Only claim what you confirm:
"CAP 1.2-conformant, schema-validated" is always safe; "SACHET-compatible"
needs NDMA's India CAP profile to be checked.

## Safety rules

- **`<status>` is always `Exercise`** (a CAP-defined value), with a `<note>`
  saying "Demonstration output from WEAVR; not an official warning". An
  exported file must never be mistakable for a real alert.
- The `<sender>` is a clearly non-operational identifier (e.g. an
  `example.invalid` address), never a real agency's.
- Export and download only. Never POST, email or publish alerts anywhere.

## CAP field mapping (verify against the CAP 1.2 spec text)

- `<msgType>` Alert, `<scope>` Public
- `<info>`: `<category>` Met, `<event>` "Heavy Rainfall", `<language>`
  en-IN (add a hi-IN `<info>` block only if a correct translation of the
  fixed strings is available; don't machine-translate warning text
  silently)
- `<urgency>` from the lead: Expected (≤ 24 h) / Future (> 24 h)
- `<severity>` from colour: Red → Extreme, Orange → Severe,
  Yellow → Moderate. This is WEAVR's own mapping; document it.
- `<certainty>` from probability, per the CAP 1.2 definitions:
  Likely (p > ~50%), Possible (p ≤ ~50% but not negligible)
- `<onset>` / `<expires>` from the IMD day (03–03 UTC) the lead validates
  on
- `<parameter>` entries with the three probabilities
- `<area>`: `<areaDesc>` "District, State"; `<polygon>` from the simplified
  boundary ("lat,lon" pairs, first = last, the WGS 84 order CAP requires).
  Handle MultiPolygons with multiple `<polygon>` elements.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Read the CAP 1.2 specification (OASIS) for every field used, and look
   for NDMA's public CAP / SACHET documentation for an India profile.
   Summarise what you found, with links, in docs/cap-export.md. If no
   India profile is publicly available, say so. Do not claim SACHET
   compatibility.

2. Add src/weavr/cap.py (stdlib xml.etree only):
   - build_cap_alert(district_warning, lead, issue_time, polygons) ->
     XML string, with the mapping above.
   - build_cap_bundle(warnings, ...) -> one alert per warned district
     (Green is skipped).
   Tests: required elements are present; status is always Exercise;
   polygons are closed and use lat,lon order; certainty and severity
   mapping edge cases (p = 0.5 exactly, each colour).

3. Schema validation: the OASIS CAP 1.2 XSD is needed in tests.
   - Check its licence/terms before vendoring it into tests/data/.
   - Choose a validator: e.g. the pure-Python `xmlschema` package added to
     the `dev` group, or lxml. Pick one and justify it.
   - Every alert produced from the committed example district warnings
     must validate.

4. API + UI:
   - GET /api/cap?lead=&district_id= returns application/xml (422 on bad
     params; 404 for a Green district, with a clear message).
   - GET /api/cap-bundle?lead= returns all of that lead's alerts.
   - Add tests.
   - Add a "Download CAP (exercise)" button to the district card
     (districtWarnings.js), shown only for warned districts.
   - Browser-verify: the download produces a file, and its contents
     validate (re-run the validator on the downloaded file).

5. Docs: docs/cap-export.md (mapping table, safety rules, validation
   method, what is and isn't verified about NDMA compatibility). README
   paragraph. One PR.
```

## Done when

- Schema-validated CAP 1.2 exercise alerts are produced for every warned
  district.
- They are downloadable from the district card.
- The NDMA / SACHET compatibility statement says only what was verified.
