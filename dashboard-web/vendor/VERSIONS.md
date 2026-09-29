# Vendored libraries

Committed so the dashboard works offline; nothing loads from a CDN at runtime.
No build step: both are single plain-script files exposing a global.

| Library | Version | Files | Licence | Source |
|---|---|---|---|---|
| MapLibre GL JS | 5.24.0 | `maplibre-gl/maplibre-gl.js` (global `maplibregl`), `maplibre-gl.css`, `LICENSE.txt` | BSD-3-Clause | npm `maplibre-gl@5.24.0`, `dist/` |
| PMTiles | 4.5.0 | `pmtiles/pmtiles.js` (global `pmtiles`), `LICENSE` | BSD-3-Clause | npm `pmtiles@4.5.0`, `dist/pmtiles.js`; licence text from the protomaps/PMTiles repository |

**Why MapLibre 5.24.0, not the newest 6.x:** version 6 ships only as ES
modules split across several files, which needs `type="module"` loading and
breaks the repo's "plain script tags, no build step" rule. The last 5.x
release is a single UMD file.

The trailing `sourceMappingURL` comment was removed from both scripts so the
browser does not request `.map` files that are not vendored.

To upgrade, download the new files, update this table, and re-run the
browser check in `docs/basemap-scope.md`.
