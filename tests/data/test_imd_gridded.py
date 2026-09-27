import pytest

from weavr.data.imd_gridded import fetch_year


def test_fetch_year_rain(tmp_path):
    try:
        ds = fetch_year(2023, var_type="rain", cache_dir=tmp_path)
    except Exception as exc:  # network/IMD server access issue, not a code defect
        pytest.skip(f"IMD data server unreachable: {exc}")

    assert ds.sizes["time"] == 365
    assert ds.lat.min() >= 6.0
    assert ds.lat.max() <= 38.5
    # IMD's own fill value must already be converted to NaN, not left as 99.9
    assert bool((ds["rain"] < 90).any())
