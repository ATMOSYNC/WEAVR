import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_basemap_boundaries as cb  # noqa: E402

# --- a tiny MVT encoder, just enough to build fixtures -----------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, wire: int, payload: bytes | int) -> bytes:
    key = _varint((number << 3) | wire)
    if wire == 0:
        assert isinstance(payload, int)
        return key + _varint(payload)
    assert isinstance(payload, bytes)
    return key + _varint(len(payload)) + payload


def _layer(name: str, features: list[tuple[int, str | None]]) -> bytes:
    kinds = sorted({k for _, k in features if k is not None})
    body = _field(1, 2, name.encode())
    for geometry_type, kind in features:
        feature = _field(3, 0, geometry_type)
        if kind is not None:
            tags = _varint(0) + _varint(kinds.index(kind))
            feature += _field(2, 2, tags)
        body += _field(2, 2, feature)
    body += _field(3, 2, b"kind")
    for kind in kinds:
        body += _field(4, 2, _field(1, 2, kind.encode()))
    return _field(3, 2, body)


def _tile(layers: dict[str, list[tuple[int, str | None]]]) -> bytes:
    return b"".join(_layer(name, feats) for name, feats in layers.items())


class TestDecode:
    def test_round_trips_layers_geometry_types_and_kinds(self):
        data = _tile(
            {
                "roads": [(cb.LINE, "highway"), (cb.LINE, "minor_road")],
                "earth": [(cb.POLYGON, None)],
            }
        )

        decoded = cb.decode_tile(data)

        assert decoded["roads"] == [
            {"type": cb.LINE, "kind": "highway"},
            {"type": cb.LINE, "kind": "minor_road"},
        ]
        assert decoded["earth"] == [{"type": cb.POLYGON, "kind": None}]

    def test_empty_tile_decodes_to_nothing(self):
        assert cb.decode_tile(b"") == {}


class TestRules:
    STYLE = {"earth", "landcover", "landuse", "water", "roads"}

    def test_clean_tile_passes(self):
        layers = cb.decode_tile(
            _tile(
                {
                    "earth": [(cb.POLYGON, None)],
                    "water": [(cb.POLYGON, "ocean"), (cb.LINE, "river")],
                    "roads": [(cb.LINE, "highway")],
                }
            )
        )

        assert cb.check_layers(layers, self.STYLE) == []

    def test_a_line_in_earth_is_flagged_as_a_possible_border(self):
        layers = cb.decode_tile(_tile({"earth": [(cb.LINE, None)]}))

        problems = cb.check_layers(layers, self.STYLE)

        assert any("earth" in p and "line" in p for p in problems)

    def test_border_like_kinds_are_flagged_in_any_used_layer(self):
        for kind in ("admin_boundary", "disputed", "country_border"):
            layers = cb.decode_tile(_tile({"landuse": [(cb.POLYGON, kind)]}))

            assert cb.check_layers(layers, self.STYLE), kind

    def test_unexpected_road_kind_and_non_line_roads_are_flagged(self):
        layers = cb.decode_tile(
            _tile({"roads": [(cb.LINE, "mystery"), (cb.POLYGON, "highway")]})
        )

        problems = cb.check_layers(layers, self.STYLE)

        assert any("unexpected kind" in p for p in problems)
        assert any("non-line" in p for p in problems)

    def test_layers_the_style_does_not_use_are_ignored(self):
        # The file ships a `boundaries` layer; it is only a problem if drawn.
        layers = cb.decode_tile(_tile({"boundaries": [(cb.LINE, "country")]}))

        assert cb.check_layers(layers, self.STYLE) == []


class TestTiles:
    def test_lonlat_to_tile_known_points(self):
        assert cb.lonlat_to_tile(0.0, 0.0, 1) == (1, 1)
        assert cb.lonlat_to_tile(-179.9, 84.0, 3) == (0, 0)

    def test_sample_tiles_are_distinct_and_limited(self):
        tiles = cb.sample_tiles(cb.REGIONS["Kashmir"], 8, 6)

        assert 1 <= len(tiles) <= 6
        assert len(set(tiles)) == len(tiles)

    def test_low_zoom_collapses_to_few_tiles(self):
        assert len(cb.sample_tiles(cb.REGIONS["Arunachal Pradesh"], 3, 6)) <= 2


class TestStyleAgreement:
    def test_the_real_style_only_draws_known_road_kinds(self):
        import json

        style = json.loads(cb.DEFAULT_STYLE.read_text())
        for layer in style["layers"]:
            if layer.get("source-layer") == "roads":
                kinds = set(layer["filter"][2][1])  # ["in", get, ["literal", [...]]]
                assert kinds <= cb.ROAD_KINDS
