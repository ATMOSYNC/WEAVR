"""Check that the basemap draws no boundary lines of its own.

The basemap must not draw national or disputed boundaries: WEAVR overlays the
official one on top (docs/basemap-scope.md). The tile file does contain a
`boundaries` layer (the Protomaps build ships one), so this checks what the
map can actually render:

1. The style references no boundary/label layers (also tested in
   tests/test_dashboard_basemap.py).
2. In real tiles sampled over Kashmir and Arunachal Pradesh at zooms 3-10, every
   style-referenced layer holds only what it should: `earth`, `landcover` and
   `landuse` hold no line features at all (an outline there would be a border),
   `roads` holds only lines of known road kinds, `water` holds polygons and
   lines. Any feature whose `kind` looks like a boundary fails the check.

No new dependency: it reads tiles with the `pmtiles` CLI and decodes the
Mapbox Vector Tile protobuf with a small wire-format reader below.

Usage:
    python scripts/check_basemap_boundaries.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TILES = ROOT / "data" / "basemap" / "india.pmtiles"
DEFAULT_STYLE = ROOT / "dashboard-web" / "basemap" / "style.json"

# Regions where the depiction of the boundary is sensitive.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Kashmir": (73.0, 32.5, 80.5, 37.1),
    "Arunachal Pradesh": (91.5, 26.5, 97.5, 29.8),
}
ZOOMS = range(3, 11)
MAX_TILES_PER_ZOOM_PER_REGION = 6

BORDER_WORDS = ("boundar", "border", "admin", "disputed", "country", "state", "line_of")
POINT, LINE, POLYGON = 1, 2, 3
GEOMETRY_NAMES = {POINT: "point", LINE: "line", POLYGON: "polygon"}
NO_LINES_LAYERS = {"earth", "landcover", "landuse"}
# Every kind the `roads` layer carries (the style draws only four of them).
ROAD_KINDS = {
    "highway", "major_road", "medium_road", "minor_road",
    "other", "path", "rail", "ferry", "aeroway",
}


# --- Minimal Mapbox Vector Tile reader --------------------------------------


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(data: bytes) -> list[tuple[int, int, Any]]:
    """Protobuf wire format -> [(field number, wire type, value)]."""
    out: list[tuple[int, int, Any]] = []
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        field, wire = key >> 3, key & 7
        value: Any
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 2:
            length, pos = _varint(data, pos)
            value = data[pos : pos + length]
            pos += length
        elif wire == 1:
            value, pos = data[pos : pos + 8], pos + 8
        elif wire == 5:
            value, pos = data[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.append((field, wire, value))
    return out


def _packed_varints(data: bytes) -> list[int]:
    values, pos = [], 0
    while pos < len(data):
        value, pos = _varint(data, pos)
        values.append(value)
    return values


def decode_tile(data: bytes) -> dict[str, list[dict[str, Any]]]:
    """Decode a (decompressed) MVT into {layer name: [{"type", "kind"}]}.

    `kind` is the feature's `kind` property when it has one, else None.
    """
    layers: dict[str, list[dict[str, Any]]] = {}
    for field, _, raw in _fields(data):
        if field != 3:  # Tile.layers
            continue
        name = ""
        keys: list[str] = []
        values: list[Any] = []
        raw_features: list[bytes] = []
        for lfield, _, lvalue in _fields(raw):
            if lfield == 1:
                name = lvalue.decode()
            elif lfield == 2:
                raw_features.append(lvalue)
            elif lfield == 3:
                keys.append(lvalue.decode())
            elif lfield == 4:
                text = next((v for f, w, v in _fields(lvalue) if f == 1), None)
                values.append(text.decode() if isinstance(text, bytes) else None)
        features = []
        for raw_feature in raw_features:
            geometry_type = 0
            tags: list[int] = []
            for ffield, _, fvalue in _fields(raw_feature):
                if ffield == 3:
                    geometry_type = fvalue
                elif ffield == 2:
                    tags = _packed_varints(fvalue)
            kind = None
            for k, v in zip(tags[0::2], tags[1::2], strict=False):
                if k < len(keys) and keys[k] == "kind" and v < len(values):
                    kind = values[v]
            features.append({"type": geometry_type, "kind": kind})
        layers.setdefault(name, []).extend(features)
    return layers


# --- Sampling and rules ------------------------------------------------------


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def sample_tiles(
    bbox: tuple[float, float, float, float], zoom: int, limit: int
) -> list[tuple[int, int]]:
    """Distinct tiles over the bbox: corners, edge midpoints and centre first."""
    west, south, east, north = bbox
    mid_lon, mid_lat = (west + east) / 2, (south + north) / 2
    points = [
        (mid_lon, mid_lat), (west, north), (east, north), (west, south), (east, south),
        (mid_lon, north), (mid_lon, south), (west, mid_lat), (east, mid_lat),
    ]
    tiles: list[tuple[int, int]] = []
    for lon, lat in points:
        tile = lonlat_to_tile(lon, lat, zoom)
        if tile not in tiles:
            tiles.append(tile)
    return tiles[:limit]


def looks_like_border(text: str | None) -> bool:
    return text is not None and any(word in text.lower() for word in BORDER_WORDS)


def check_layers(layers: dict[str, list[dict[str, Any]]], style_layers: set[str]) -> list[str]:
    """Rule violations for one decoded tile (empty list = fine)."""
    problems = []
    for name, features in layers.items():
        if name not in style_layers:
            continue
        for feature in features:
            kind, geometry = feature["kind"], feature["type"]
            if looks_like_border(kind):
                problems.append(f"{name}: feature kind {kind!r} looks like a boundary")
            if name in NO_LINES_LAYERS and geometry == LINE:
                problems.append(f"{name}: line feature (an outline would be a border)")
            if name == "roads":
                if geometry != LINE:
                    shape = GEOMETRY_NAMES.get(geometry, geometry)
                    problems.append(f"roads: non-line feature ({shape})")
                if kind not in ROAD_KINDS:
                    problems.append(f"roads: unexpected kind {kind!r}")
    return sorted(set(problems))


def style_source_layers(style_path: Path) -> set[str]:
    style = json.loads(style_path.read_text())
    return {layer["source-layer"] for layer in style["layers"] if "source-layer" in layer}


def read_tile(tiles: Path, zoom: int, x: int, y: int) -> dict[str, list[dict[str, Any]]]:
    result = subprocess.run(
        ["pmtiles", "tile", str(tiles), str(zoom), str(x), str(y)],
        capture_output=True,
        check=False,
    )
    data = result.stdout
    if not data:
        return {}
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return decode_tile(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args(argv)

    if not args.tiles.is_file():
        print(f"{args.tiles} not found; run scripts/build_basemap.py first.", file=sys.stderr)
        return 2
    used = style_source_layers(args.style)
    failures: list[str] = []
    present: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    unused_counts: dict[str, int] = {}
    tiles_read = 0

    for region, bbox in REGIONS.items():
        for zoom in ZOOMS:
            for x, y in sample_tiles(bbox, zoom, MAX_TILES_PER_ZOOM_PER_REGION):
                layers = read_tile(args.tiles, zoom, x, y)
                if not layers:
                    continue
                tiles_read += 1
                present.update(layers)
                for name, features in layers.items():
                    if name not in used:
                        unused_counts[name] = unused_counts.get(name, 0) + len(features)
                    if name in used:
                        for feature in features:
                            key = GEOMETRY_NAMES.get(feature["type"], "?")
                            counts.setdefault(name, {}).setdefault(key, 0)
                            counts[name][key] += 1
                for problem in check_layers(layers, used):
                    failures.append(f"{region} z{zoom} tile {x}/{y}: {problem}")

    print(f"tiles read: {tiles_read}; layers seen: {sorted(present)}")
    print(f"layers the style uses: {sorted(used)}")
    unused = sorted(present - used)
    print(f"present in the file but NOT referenced by the style (never drawn): {unused}")
    print(f"  features in those layers across the samples: {unused_counts}")
    for name in sorted(counts):
        print(f"  {name}: {counts[name]}")
    if tiles_read == 0:
        print("FAILED: no tiles could be read", file=sys.stderr)
        return 1
    missing = used - present
    if missing:
        failures.append(f"style uses layers never seen in the samples: {sorted(missing)}")
    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures[:40]:
            print("  " + failure, file=sys.stderr)
        return 1
    print("PASSED: the style draws no boundary lines from these tiles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
