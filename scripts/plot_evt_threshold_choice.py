#!/usr/bin/env python3
"""Threshold-choice figure for step 12's GPD tail (prompt item 1).

Reads `results/tail_evt_diagnostics.csv` (written by
`scripts/diagnose_evt_tail.py`) and draws the two standard checks side by
side, one line per LOYO fold (labelled by its held-out year; the tail is
fitted on the other season):

- **Mean residual life**: the mean excess over each candidate threshold u.
  Above a threshold where a GPD fits, it is linear in u.
- **Parameter stability**: the fitted shape xi with its block-bootstrap 95%
  CI. Above a suitable u, xi stays constant within its CI.

The runner's threshold (u = 64.5 mm, IMD's "heavy" boundary, fixed before
fitting as `docs/preregistration.md` requires) is marked. The figure checks
that rule; it does not choose u.

Usage: python scripts/plot_evt_threshold_choice.py [--diagnostics CSV] [--out PNG]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill), light mode; validated for two series.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {"2018": "#2a78d6", "2020": "#eb6834"}


def plot_threshold_choice(diagnostics: pd.DataFrame, out: Path) -> Path:
    """Write the two-panel figure to `out` and return the path."""
    frame = diagnostics.sort_values(["fold", "threshold_u"])
    runner_u = float(frame.loc[frame["is_runner_threshold"].astype(bool), "threshold_u"].iloc[0])

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(runner_u, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("candidate threshold u (mm/day)")

    for fold, group in frame.groupby(frame["fold"].astype(str)):
        color = SERIES.get(fold, TEXT_SECONDARY)
        u = group["threshold_u"]
        left.plot(u, group["mean_residual_life_mm"], color=color, linewidth=2,
                  marker="o", markersize=5, label=f"{fold} held out")
        right.fill_between(u, group["xi_ci_lo_block_bootstrap"], group["xi_ci_hi_block_bootstrap"],
                           color=color, alpha=0.15, linewidth=0)
        right.plot(u, group["xi"], color=color, linewidth=2, marker="o", markersize=5,
                   label=f"{fold} held out")

    left.set_title("Mean residual life", color=TEXT_PRIMARY, loc="left")
    left.set_ylabel("mean excess over u (mm)")
    right.set_title("Shape stability (xi, 95% block-bootstrap CI)", color=TEXT_PRIMARY, loc="left")
    right.set_ylabel("GPD shape xi")
    right.axhline(0.0, color=GRID, linewidth=1)
    for ax in (left, right):
        ax.annotate(f"u = {runner_u:g} mm (pre-registered)", (runner_u, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", color=TEXT_SECONDARY, fontsize=8)
    left.legend(frameon=False, loc="lower right", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics", type=Path, default=Path("results/tail_evt_diagnostics.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/figures/evt_threshold_choice.png"))
    args = parser.parse_args()
    print(f"Wrote {plot_threshold_choice(pd.read_csv(args.diagnostics), args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
