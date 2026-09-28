"""WEAVR dashboard (Phase 7, issue #9): the real, assembled, runnable app.

Wires steps 2-4's 4 real views (`dashboard/views/`) into one navigable
Streamlit app. Every piece already existed by the time this step started
(`docs/phase7-dashboard-scope.md` scoped the data sources and stack; steps
2-4 built each view against real data) -- this file only adds navigation
and layout, it does not build a 5th view or change any of the 4 existing
ones.

Run with:
    pip install -e ".[dashboard]"
    streamlit run dashboard/app.py

`streamlit run` inserts this script's own directory (`dashboard/`) into
`sys.path`, not its parent -- a real bug caught while building step 3
(`from dashboard...` imports otherwise fail with `ModuleNotFoundError`).
The repo root is added to `sys.path` below, before any `dashboard.*`
import, to fix that for every view this app wires in.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.views import blended_map, extreme_probability, skill_trends, weight_map

st.set_page_config(page_title="WEAVR dashboard", layout="wide")

st.title("WEAVR dashboard")
st.caption(
    "Issue #9's 4 real views, built against this project's real, committed "
    "data (not mockups or placeholder numbers). Each view states its own "
    "real caveats below -- summarized here, not repeated in full:"
)
st.markdown(
    "- **Weight map** and **skill trends** read already-committed CSVs "
    "live (`results/tier1_regional_weights.csv`, the 4 tiers' own results "
    "CSVs) -- these two update automatically if those files change.\n"
    "- **Blended map** and **extreme-probability map** read small, "
    "committed *example* grids (`dashboard/data/*.npz`), not a live "
    "re-blend of the current local data store -- "
    "`docs/phase7-dashboard-scope.md` explains why (the real spatial data "
    "these two views need only exists in a local, gitignored Zarr store, "
    "not guaranteed to exist on whoever opens this dashboard). Re-run "
    "`scripts/export_dashboard_example_grids.py` to refresh them from a "
    "real local store.\n"
    "- The **blended map** uses Tier 1's real regional blend; the "
    "**extreme-probability map** uses EMOS-CSG's real fitted `ifs_ens` "
    "combiner -- two different combiners, for a real, stated reason (see "
    "each view's own caption), not an inconsistency."
)

PAGES = {
    "Blended map": blended_map.render,
    "Weight map": weight_map.render,
    "Skill trends": skill_trends.render,
    "Extreme-probability map": extreme_probability.render,
}

page = st.sidebar.radio("View", list(PAGES.keys()))
st.sidebar.caption(
    "See docs/phase7-dashboard-scope.md for the full data-source and "
    "colour-scheme decisions behind every view."
)

st.divider()
PAGES[page]()
