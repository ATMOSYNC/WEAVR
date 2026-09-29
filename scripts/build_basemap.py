"""Build the local India basemap file the dashboard's geography toggle reads.

Extracts an India-only, OpenStreetMap-derived vector-tile archive
(PMTiles) from the public Protomaps daily build, using the `pmtiles` CLI
(`brew install pmtiles`). Reads only the tiles inside the bounding box, so
the 138 GB planet file is never downloaded. Measured sizes and the reasoning
are in docs/basemap-scope.md.

The output (data/basemap/india.pmtiles) is gitignored. Alongside it a
manifest.json records where it came from, so the file can be reproduced and
correctly attributed.

Usage:
    python scripts/build_basemap.py --dry-run
    python scripts/build_basemap.py                 # asks nothing; run --dry-run first
    python scripts/build_basemap.py --source-url https://build.protomaps.com/20260928.pmtiles
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# The common grid's extent (src/weavr/grid.py): lon 66.5-100.0, lat 6.5-38.5.
DEFAULT_BBOX = (66.5, 6.5, 100.0, 38.5)  # min_lon, min_lat, max_lon, max_lat
DEFAULT_MAX_ZOOM = 10
DEFAULT_OUT = Path("data/basemap/india.pmtiles")
BUILD_URL = "https://build.protomaps.com/{date}.pmtiles"
LOOKBACK_DAYS = 10
ATTRIBUTION = "© OpenStreetMap contributors"
MANIFEST_VERSION = 1


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """Parse 'min_lon,min_lat,max_lon,max_lat' and validate it."""
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"bbox needs 4 comma-separated numbers, got {text!r}")
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"bbox values must be numbers: {text!r}") from exc
    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError(f"bbox longitudes must satisfy -180 <= min < max <= 180: {text!r}")
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError(f"bbox latitudes must satisfy -90 <= min < max <= 90: {text!r}")
    return min_lon, min_lat, max_lon, max_lat


def candidate_build_urls(today: dt.date, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    """Newest-first daily build URLs to try when no --source-url is given."""
    return [
        BUILD_URL.format(date=(today - dt.timedelta(days=i)).strftime("%Y%m%d"))
        for i in range(1, lookback_days + 1)
    ]


def build_manifest(
    *,
    source_url: str,
    bbox: tuple[float, float, float, float],
    max_zoom: int,
    tool_version: str,
    size_bytes: int,
    sha256: str,
    built_at: dt.datetime,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "source_url": source_url,
        "bbox": list(bbox),
        "max_zoom": max_zoom,
        "tool": "pmtiles extract",
        "tool_version": tool_version,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "built_at": built_at.isoformat(),
        "attribution": ATTRIBUTION,
        "licence_note": (
            "OpenStreetMap data, ODbL. Attribution required. Build derived "
            "from OpenStreetMap and Natural Earth (per the build's metadata)."
        ),
    }


def is_current(
    manifest: dict[str, Any] | None,
    *,
    source_url: str | None,
    bbox: tuple[float, float, float, float],
    max_zoom: int,
    file_size: int | None,
) -> bool:
    """True if an existing file already matches the requested build.

    `source_url=None` means "any source is fine" (the user asked for the
    default, so an existing build is not replaced merely because a newer day
    exists).
    """
    if manifest is None or file_size is None:
        return False
    if manifest.get("size_bytes") != file_size:
        return False
    if tuple(manifest.get("bbox", ())) != tuple(bbox):
        return False
    if manifest.get("max_zoom") != max_zoom:
        return False
    return source_url is None or manifest.get("source_url") == source_url


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def url_exists(url: str) -> bool:
    # The host rejects urllib's default User-Agent with a 403.
    headers = {"User-Agent": "weavr-build-basemap"}
    request = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, TimeoutError):
        return False


def resolve_source_url(source_url: str | None) -> str:
    if source_url:
        return source_url
    for url in candidate_build_urls(dt.date.today()):
        if url_exists(url):
            return url
    raise SystemExit(
        f"No Protomaps daily build found in the last {LOOKBACK_DAYS} days; "
        "pass --source-url explicitly."
    )


def pmtiles_binary() -> str:
    path = shutil.which("pmtiles")
    if path is None:
        raise SystemExit(
            "The `pmtiles` CLI is not installed. Install it with "
            "`brew install pmtiles` (or see https://github.com/protomaps/go-pmtiles) "
            "and re-run."
        )
    return path


def pmtiles_version(binary: str) -> str:
    result = subprocess.run([binary, "version"], capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-url", help="Protomaps build URL (default: newest daily build)")
    parser.add_argument("--bbox", default=",".join(str(v) for v in DEFAULT_BBOX))
    parser.add_argument("--max-zoom", type=int, default=DEFAULT_MAX_ZOOM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="Estimate only; download nothing")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the file is current")
    args = parser.parse_args(argv)

    bbox = parse_bbox(args.bbox)
    if not 0 <= args.max_zoom <= 15:
        parser.error("--max-zoom must be between 0 and 15")

    out: Path = args.out
    manifest_path = out.with_name("manifest.json")
    file_size = out.stat().st_size if out.exists() else None
    if not args.force and is_current(
        read_manifest(manifest_path),
        source_url=args.source_url,
        bbox=bbox,
        max_zoom=args.max_zoom,
        file_size=file_size,
    ):
        print(f"{out} already matches the requested build; use --force to rebuild.")
        return 0

    binary = pmtiles_binary()
    source_url = resolve_source_url(args.source_url)
    command = [
        binary, "extract", source_url, str(out),
        f"--bbox={args.bbox}", f"--maxzoom={args.max_zoom}",
    ]
    if args.dry_run:
        # A dry run reads directory headers and writes no file.
        print(f"source: {source_url}\nbbox: {args.bbox}  max zoom: {args.max_zoom}")
        return subprocess.run([*command, "--dry-run"], check=False).returncode

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)
    command[3] = str(partial)
    print("Running:", " ".join(command))
    if subprocess.run(command, check=False).returncode != 0:
        partial.unlink(missing_ok=True)
        print("pmtiles extract failed; nothing was written.", file=sys.stderr)
        return 1
    partial.replace(out)

    manifest = build_manifest(
        source_url=source_url,
        bbox=bbox,
        max_zoom=args.max_zoom,
        tool_version=pmtiles_version(binary),
        size_bytes=out.stat().st_size,
        sha256=sha256_of(out),
        built_at=dt.datetime.now(dt.UTC),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out} ({manifest['size_bytes'] / 1e6:.1f} MB) and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
