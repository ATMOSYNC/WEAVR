import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_evt_threshold_choice import plot_threshold_choice  # noqa: E402


def test_writes_a_png_from_the_diagnostics_schema(tmp_path):
    rows = []
    for fold in ("2018", "2020"):
        for u in (35.5, 64.5, 115.6):
            rows.append({
                "fold": fold, "threshold_u": u, "mean_residual_life_mm": 20 + u / 5,
                "xi": 0.1, "xi_ci_lo_block_bootstrap": 0.0, "xi_ci_hi_block_bootstrap": 0.2,
                "is_runner_threshold": u == 64.5,
            })
    out = plot_threshold_choice(pd.DataFrame(rows), tmp_path / "fig" / "t.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
