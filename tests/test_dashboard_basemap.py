import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Same optional-group skip as tests/test_dashboard_api.py.
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import dashboard.api as api  # noqa: E402

client = TestClient(api.app)

STYLE_PATH = Path(__file__).resolve().parents[1] / "dashboard-web" / "basemap" / "style.json"
REAL_TILES = Path(__file__).resolve().parents[1] / "data" / "basemap" / "india.pmtiles"

# Anything that could draw a boundary, name a place, or add clutter.
FORBIDDEN_SOURCE_LAYERS = {"boundaries", "places", "buildings", "pois"}
FORBIDDEN_WORDS = ("boundar", "border", "admin", "disputed", "country", "state")


@pytest.fixture
def tiny_tiles(tmp_path, monkeypatch):
    path = tmp_path / "india.pmtiles"
    path.write_bytes(bytes(range(256)) * 4)  # 1024 bytes
    monkeypatch.setattr(api, "BASEMAP_TILES_PATH", path)
    return path


class TestStatus:
    def test_unavailable_when_the_file_is_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "BASEMAP_TILES_PATH", tmp_path / "missing.pmtiles")

        assert client.get("/api/basemap/status").json() == {"available": False, "bytes": None}

    def test_available_reports_the_size(self, tiny_tiles):
        assert client.get("/api/basemap/status").json() == {"available": True, "bytes": 1024}


class TestTileRoute:
    def test_missing_file_is_a_plain_404_with_instructions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "BASEMAP_TILES_PATH", tmp_path / "missing.pmtiles")

        response = client.get("/basemap/india.pmtiles")

        assert response.status_code == 404
        assert "build_basemap.py" in response.text

    def test_full_request_returns_the_file(self, tiny_tiles):
        response = client.get("/basemap/india.pmtiles")

        assert response.status_code == 200
        assert response.content == tiny_tiles.read_bytes()

    def test_range_request_returns_206_with_the_right_bytes(self, tiny_tiles):
        response = client.get("/basemap/india.pmtiles", headers={"Range": "bytes=10-19"})

        assert response.status_code == 206
        assert response.headers["content-range"] == "bytes 10-19/1024"
        assert response.content == tiny_tiles.read_bytes()[10:20]

    def test_the_response_advertises_range_support(self, tiny_tiles):
        response = client.get("/basemap/india.pmtiles")

        assert response.headers.get("accept-ranges") == "bytes"

    @pytest.mark.skipif(not REAL_TILES.exists(), reason="data/basemap/india.pmtiles not built")
    def test_real_file_header_is_readable_by_range(self, monkeypatch):
        monkeypatch.setattr(api, "BASEMAP_TILES_PATH", REAL_TILES)

        response = client.get("/basemap/india.pmtiles", headers={"Range": "bytes=0-6"})

        assert response.status_code == 206
        assert response.content == b"PMTiles"


@pytest.fixture(scope="module")
def style():
    return json.loads(STYLE_PATH.read_text())


class TestStyle:
    def test_is_a_version_8_style_with_one_vector_source(self, style):
        assert style["version"] == 8
        assert [s["type"] for s in style["sources"].values()] == ["vector"]

    def test_no_layer_can_draw_a_boundary_or_labels(self, style):
        for layer in style["layers"]:
            source_layer = layer.get("source-layer", "")
            assert source_layer not in FORBIDDEN_SOURCE_LAYERS, layer["id"]
            assert layer["type"] != "symbol", f"{layer['id']} is a text/icon layer"
            for word in FORBIDDEN_WORDS:
                assert word not in layer["id"].lower(), layer["id"]
                assert word not in source_layer.lower(), layer["id"]

    def test_only_uses_source_layers_present_in_the_tile_file(self, style):
        present = {"earth", "landcover", "landuse", "water", "roads"}
        used = {layer["source-layer"] for layer in style["layers"] if "source-layer" in layer}
        assert used <= present

    def test_has_no_remote_resources(self, style):
        text = json.dumps(style)
        assert "http://" not in text and "https://" not in text
        assert "glyphs" not in style and "sprite" not in style

    def test_attribution_names_openstreetmap(self, style):
        attributions = [s.get("attribution", "") for s in style["sources"].values()]
        assert any("OpenStreetMap" in a for a in attributions)
