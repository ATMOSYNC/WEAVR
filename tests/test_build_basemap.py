import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_basemap as bb  # noqa: E402

BBOX = (66.5, 6.5, 100.0, 38.5)


class TestParseBbox:
    def test_default_round_trips(self):
        assert bb.parse_bbox("66.5,6.5,100.0,38.5") == BBOX

    @pytest.mark.parametrize(
        "text",
        ["1,2,3", "a,b,c,d", "10,0,5,10", "0,10,5,5", "-200,0,10,10", "0,-95,10,10"],
    )
    def test_invalid_bbox_is_rejected(self, text):
        with pytest.raises(ValueError):
            bb.parse_bbox(text)


class TestCandidateUrls:
    def test_newest_first_and_starts_yesterday(self):
        urls = bb.candidate_build_urls(dt.date(2026, 9, 29), lookback_days=3)
        assert urls == [
            "https://build.protomaps.com/20260928.pmtiles",
            "https://build.protomaps.com/20260927.pmtiles",
            "https://build.protomaps.com/20260926.pmtiles",
        ]


def _manifest(**over):
    base = bb.build_manifest(
        source_url="https://build.protomaps.com/20260928.pmtiles",
        bbox=BBOX,
        max_zoom=10,
        tool_version="v1",
        size_bytes=1000,
        sha256="abc",
        built_at=dt.datetime(2026, 9, 29, tzinfo=dt.UTC),
    )
    base.update(over)
    return base


class TestManifest:
    def test_records_provenance_fields(self):
        m = _manifest()
        for key in ("source_url", "bbox", "max_zoom", "tool_version",
                    "size_bytes", "sha256", "built_at", "attribution"):
            assert key in m
        assert "OpenStreetMap" in m["attribution"]


class TestIsCurrent:
    def _call(self, manifest, **over):
        args = dict(source_url=None, bbox=BBOX, max_zoom=10, file_size=1000)
        args.update(over)
        return bb.is_current(manifest, **args)

    def test_matching_build_is_current(self):
        assert self._call(_manifest())

    def test_missing_manifest_or_file_is_not_current(self):
        assert not self._call(None)
        assert not self._call(_manifest(), file_size=None)

    def test_size_mismatch_means_the_file_changed(self):
        assert not self._call(_manifest(), file_size=999)

    def test_different_zoom_or_bbox_is_not_current(self):
        assert not self._call(_manifest(), max_zoom=11)
        assert not self._call(_manifest(), bbox=(0.0, 0.0, 1.0, 1.0))

    def test_explicit_different_source_is_not_current(self):
        assert not self._call(_manifest(), source_url="https://example.com/x.pmtiles")

    def test_default_source_keeps_an_existing_older_build(self):
        assert self._call(_manifest(), source_url=None)


class TestDryRun:
    def test_dry_run_downloads_nothing_and_writes_nothing(self, tmp_path, monkeypatch):
        calls = []

        class Done:
            returncode = 0

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Done()

        monkeypatch.setattr(bb.subprocess, "run", fake_run)
        monkeypatch.setattr(bb, "pmtiles_binary", lambda: "pmtiles")
        out = tmp_path / "india.pmtiles"

        rc = bb.main([
            "--dry-run", "--out", str(out),
            "--source-url", "https://build.protomaps.com/20260928.pmtiles",
        ])

        assert rc == 0
        assert len(calls) == 1 and "--dry-run" in calls[0]
        assert not out.exists()
        assert not out.with_name("manifest.json").exists()
