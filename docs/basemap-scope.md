# OpenStreetMap basemap — scope and decisions

Step 01 of the basemap task. This is the scoping record; later steps append
their own sections. No dashboard code changes in this step.

## What exists today

| Item | Fact |
|---|---|
| Frontend | Vanilla HTML/CSS/JS in `dashboard-web/`, plain `<script>` tags, no build step, no third-party JS |
| Server | FastAPI (`dashboard/api.py`), API under `/api/…`, static frontend mounted last at `/` |
| Grid | 0.25° regular lat/lon, lat 6.5–38.5 and lon 66.5–100.0 (129 × 135 cells); values sit at cell centres so a cell spans ±0.125° around its coordinate ([grid convention](grid-and-time-convention.md)) |
| Blended map | `<canvas>` heat-map, one `fillRect` per cell, discrete rain-bin colours from `/api/colors`, rows flipped (latitude ascending, canvas y = 0 at top), **no geography** |
| Extreme-probability map | `<canvas>` two-pass heat-map: a continuous 5-stop probability scale plus a grey overlay for cells where EMOS-CSG had no fittable bin |

## Decisions (made with the user)

| Question | Decision |
|---|---|
| India boundary | The user will supply an official-depiction GeoJSON. Until it exists, the map draws **no outline** and shows a visible "no official boundary configured" note. The basemap never draws boundaries itself. |
| Where | Both map views (blended, extreme-probability), national and zoomed, **off by default** behind a "Show geography" toggle. The existing canvas stays the default and the fallback. |
| Offline | Fully offline. Tiles come from a local file; no runtime request to any remote host, and never to the public OSM tile servers. |
| Dependency | One vendored library pair, MapLibre GL JS plus the PMTiles plug-in (about 1 MB, committed with licences). No CDN, no npm, no build step. |

### Rejected options

| Option | Why rejected |
|---|---|
| Leaflet + `tile.openstreetmap.org` | Bakes OSM's own boundaries into the picture, needs the network, and breaches the public tile servers' usage policy for a demo. |
| Third-party hosted tiles | Adds an account/key and an uptime risk, and still draws boundaries unless a style can switch them off. |
| Stretching the existing canvas over the map | Web Mercator is not linear in latitude, so cells would be misplaced by kilometres, more so toward the north. Cells are drawn as polygons instead. |

## Tile source and measured size

- **Source:** the public Protomaps daily build, `https://build.protomaps.com/YYYYMMDD.pmtiles`
  (checked: the 2026-09-28 build exists, answers range requests, and is
  138.4 GB for the planet, which we do not download).
- **Method:** `pmtiles extract … --bbox=66.5,6.5,100.0,38.5 --maxzoom=Z --dry-run`
  (pmtiles CLI installed with Homebrew). A dry run reads directory headers and
  writes no file.
- **Measured archive size for the India bounding box:**

| Max zoom | Tiles kept | Archive size | Data transferred | Requests |
|---|---|---|---|---|
| 8 | 798 | **22 MB** | 23 MB | 24 |
| 10 | 10,830 | **148 MB** | 155 MB | 51 |
| 11 | 40,404 | **305 MB** | 321 MB | 61 |

Recommendation for step 02: **max zoom 10** (148 MB). It is enough to
identify a district's roads and towns at 0.25° cell scale; zoom 11 doubles the
size for detail that a 25 km grid cannot use.

- **Layers in the build** (from its metadata): `boundaries`, `buildings`,
  `earth`, `landcover`, `landuse`, `places`, `pois`, `roads`, `water`.
  **`boundaries` must be left out of the map style.** Buildings and POIs are
  also omitted, since they add nothing at this scale. Text labels stay off in
  the first version (the `places` layer names disputed areas, and the style
  would need local glyphs anyway).
- **Licence and attribution:** the build's own attribution is "© OpenStreetMap";
  its description says it is derived from OpenStreetMap and Natural Earth.
  OpenStreetMap data is ODbL, which allows building and hosting our own tiles
  with attribution. The map must show **© OpenStreetMap contributors**.
  Re-check the build's licence page when step 02 records the manifest.

## The one non-negotiable rule

The basemap draws **no national or disputed boundaries**. OSM depicts
boundaries as mapped by contributors, which differs from the Survey of India's
depiction in Kashmir and Arunachal Pradesh, and showing a non-official map
can be a legal problem in India. WEAVR overlays the user-supplied official
boundary on top. This is tested in steps 03, 06 and 07, not just intended.

## Files each later step will touch

| Step | Files |
|---|---|
| 02 | `scripts/build_basemap.py`, `tests/test_build_basemap.py`, `data/basemap/` (gitignored), this doc |
| 03 | `dashboard-web/vendor/`, `dashboard-web/basemap/style.json`, `dashboard/api.py`, `tests/test_dashboard_basemap.py` |
| 04 | `dashboard-web/js/charts/basemapMap.js`, `blendedMap.js`, `index.html` |
| 05 | `basemapMap.js`, `extremeProbability.js` |
| 06 | `dashboard/api.py`, `basemapMap.js`, a border-check script/test |
| 07 | docs and verification only |
