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

## Building the basemap (step 02)

```bash
python scripts/build_basemap.py --dry-run      # estimate only, downloads nothing
python scripts/build_basemap.py --source-url https://build.protomaps.com/20260928.pmtiles
```

Requires the `pmtiles` CLI (`brew install pmtiles`). The script extracts only
the India bounding box (66.5, 6.5, 100.0, 38.5, the common grid's extent) up
to max zoom 10, writes to `india.pmtiles.partial` first and renames on
success (so an interrupted run never leaves a truncated file that looks
valid), then writes `data/basemap/manifest.json`. A second run is a no-op
while the file still matches the manifest; `--force` rebuilds. With no
`--source-url` it uses the newest daily build in the last ten days.
Pass an explicit URL when you want a reproducible build.

**Measured (2026-09-29):** 148.0 MB archive (147,968,833 bytes), 155 MB
transferred, 51 requests, about 41 s wall time. Header check: bounds
66.5–100.0 E, 6.5–38.5 N, zoom 0–10, 13,091 addressed tiles, 10,830 tile
entries. `data/basemap/` is gitignored (`git check-ignore` confirms it).

The manifest records source URL, bounding box, max zoom, tool version, size,
SHA-256, build time and the attribution string.

### Layers in the file, and which carry boundaries

| Layer | Use in the map style |
|---|---|
| `boundaries` | **Carries national, disputed and administrative boundaries. Never included.** |
| `earth`, `water`, `landcover`, `landuse` | Polygons: land, water bodies, cover. Included. |
| `roads` | Lines: roads, rail, ferries. Included, muted. |
| `places` | Point labels, including names of disputed areas. Excluded until local fonts exist and the labels are reviewed. |
| `buildings`, `pois` | Excluded (nothing useful at 0.25° scale). |

Only `boundaries` is documented as holding boundary lines. That comes from the
build's layer list, not from decoding the tiles, so step 06 still has to
verify that no other layer carries a boundary feature.

## Serving the basemap (step 03)

**Vendored libraries** (`dashboard-web/vendor/`, versions and licences in
`VERSIONS.md`): MapLibre GL JS 5.24.0 (1.06 MB) and PMTiles 4.5.0 (20 KB),
both BSD-3-Clause. Version 5 was chosen over the newest 6.x because 6.x is
ES-module only, which the no-build-step frontend cannot load with plain
script tags.

**Style** (`dashboard-web/basemap/style.json`): background, `earth`,
`landcover`, `landuse`, `water` (polygons and lines) and two `roads` layers
(major from zoom 5, minor from zoom 8). No `boundaries`, `places`,
`buildings` or `pois`, no symbol layers, no glyphs or sprite, no remote URLs.
The vector source URL is a placeholder (`pmtiles://__BASEMAP_TILES_URL__`);
`basemapMap.js` (step 04) replaces it with an absolute
`pmtiles://<origin>/basemap/india.pmtiles` at runtime.

**Routes** (`dashboard/api.py`):

| Route | Behaviour |
|---|---|
| `GET /basemap/india.pmtiles` | Streams `data/basemap/india.pmtiles`; answers `Range` requests with `206 Partial Content`. Missing file: `404` plain text telling you to run `scripts/build_basemap.py`. |
| `GET /api/basemap/status` | `{"available": bool, "bytes": int \| null}` so the frontend can fall back before creating a map. |

**Tests** (`tests/test_dashboard_basemap.py`, 12): status with and without the
file; full, range (`bytes 10-19/1024`) and missing-file responses; the real
file's first bytes read as `PMTiles`; the style contains no forbidden source
layers, no symbol layers, no boundary-like names, no remote URLs, uses only
layers that exist in the tile file, and names OpenStreetMap in its
attribution.

**Browser check (2026-09-29):** on a fresh port, the vendored libraries and
the real tile file rendered the Kerala coast at zoom 7 (land, sea, roads,
"© OpenStreetMap contributors" attribution) with no map errors, and every
network request went to `127.0.0.1` only. `curl -r 0-99` on the tile route
returns `206` with `Content-Range: bytes 0-99/147968833`.
