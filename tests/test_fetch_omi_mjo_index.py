import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_omi_mjo_index import (  # noqa: E402
    OMI_COLUMNS,
    build_omi_jjas_dataset,
    compute_mjo_phase,
    fetch_omi_raw,
)


class TestComputeMjoPhase:
    @pytest.mark.parametrize(
        "pc1,pc2,expected_phase",
        [
            (1.0, 0.0, 1),  # angle 0 deg -> sector [0, 45) -> phase 1
            (0.0, 1.0, 3),  # angle 90 deg -> sector [90, 135) -> phase 3
            (-1.0, 0.0, 5),  # angle 180 deg -> sector [180, 225) -> phase 5
            (0.0, -1.0, 7),  # angle 270 deg -> sector [270, 315) -> phase 7
            (1.0, -0.01, 8),  # angle just below 360 -> sector [315, 360) -> phase 8
        ],
    )
    def test_known_angles_map_to_expected_phase(self, pc1, pc2, expected_phase):
        result = compute_mjo_phase(np.array([pc1]), np.array([pc2]))
        assert result[0] == expected_phase

    def test_phase_is_always_in_1_to_8(self):
        rng = np.random.default_rng(0)
        pc1 = rng.normal(size=1000)
        pc2 = rng.normal(size=1000)

        result = compute_mjo_phase(pc1, pc2)

        assert result.min() >= 1
        assert result.max() <= 8


class TestBuildOmiJjasDataset:
    def _synthetic_frame(self) -> pd.DataFrame:
        dates = pd.date_range("2019-01-01", "2020-12-31", freq="D")
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "year": dates.year,
                "month": dates.month,
                "day": dates.day,
                "pc1": rng.normal(size=len(dates)),
                "pc2": rng.normal(size=len(dates)),
                "amplitude": rng.uniform(0.5, 2.0, size=len(dates)),
            }
        )

    def test_restricts_to_the_requested_jjas_season_only(self):
        frame = self._synthetic_frame()

        result = build_omi_jjas_dataset(frame, year=2020)

        times = pd.DatetimeIndex(result["time"].values)
        assert times.min() == pd.Timestamp("2020-06-01")
        assert times.max() == pd.Timestamp("2020-09-30")
        assert result.sizes["time"] == 122

    def test_raises_for_a_year_with_no_rows(self):
        frame = self._synthetic_frame()

        with pytest.raises(ValueError, match="No OMI rows"):
            build_omi_jjas_dataset(frame, year=1999)

    def test_output_is_sorted_by_time(self):
        frame = self._synthetic_frame().sample(frac=1.0, random_state=0)  # shuffle

        result = build_omi_jjas_dataset(frame, year=2020)

        times = pd.DatetimeIndex(result["time"].values)
        assert list(times) == sorted(times)


class TestFetchOmiRawColumnValidation:
    def test_wrong_column_count_raises(self, monkeypatch):
        class _FakeResponse:
            text = "2020 6 1 0.1 0.2\n2020 6 2 0.3 0.4\n"  # only 5 columns

            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            "fetch_omi_mjo_index.requests.get", lambda *a, **k: _FakeResponse()
        )

        with pytest.raises(ValueError, match="Expected 6 columns"):
            fetch_omi_raw()

    def test_expected_column_count_matches_omi_columns_constant(self):
        assert len(OMI_COLUMNS) == 6
