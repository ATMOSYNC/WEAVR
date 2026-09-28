"""Blended-map view: Tier 1's real regional blend over India, with IMD's
real rain-intensity colour breakpoints.

"Blended" means this project's actual, real blend -- Tier 1's regional
weighting (`weavr.weighting.fit_region_weights`), the same combiner
`docs/phase6-operational-scope.md` decided the daily pipeline itself
should use, because Tier 2/3's fitted EMOS-CSG/BMA parameters don't
transfer to a live source substitution. This is not "the best" combiner by
some unstated metric -- Tier 2 (EMOS-CSG/BMA) and Tier 3
(regime-conditioned) both have their own real, different outcomes (see
`docs/tier2-hierarchical-baseline-results.md`,
`docs/phase5-regime-conditioned-results.md`); Tier 1 is shown here because
it is what this project's own operational decision already settled on.

The grid itself is a small, committed example export
(`dashboard/data/example_blend_grid.npz`, built by
`scripts/export_dashboard_example_grids.py`), not a live re-blend --
`docs/phase7-dashboard-scope.md`'s own decision, since the real local Zarr
store this would otherwise read from is gitignored and not guaranteed to
exist on whoever opens this dashboard.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.colors import RAIN_BIN_COLORS, build_discrete_colorscale, rain_bin_index
from dashboard.data_loading import available_blend_grid_leads, load_blend_grid
from weavr.rain_bins import RAIN_BIN_LABELS


def render() -> None:
    st.subheader("Blended map: Tier 1's real regional blend over India")
    st.caption(
        "This map shows Tier 1's real fitted regional blend "
        "(weavr.weighting.fit_region_weights) -- the combiner "
        "docs/phase6-operational-scope.md decided the daily pipeline itself "
        "uses, not necessarily this project's single 'best' combiner by any "
        "one metric (Tier 2/3 have their own different, real outcomes). "
        "The grid is a small, committed example snapshot, not a live "
        "re-blend of the current local data store -- see "
        "docs/phase7-dashboard-scope.md."
    )

    leads = available_blend_grid_leads()
    lead = st.selectbox("Lead time (hours)", leads, index=0, key="blended_map_lead")
    grid = load_blend_grid(lead)

    bin_index = rain_bin_index(grid["values"])
    colors = [RAIN_BIN_COLORS[label] for label in RAIN_BIN_LABELS]

    fig = px.imshow(
        bin_index,
        x=grid["longitude"],
        y=grid["latitude"],
        origin="lower",
        color_continuous_scale=build_discrete_colorscale(colors),
        zmin=0,
        zmax=len(RAIN_BIN_LABELS),
        labels={"x": "Longitude", "y": "Latitude", "color": "IMD category"},
        title=f"Blended precipitation at lead {lead}h (sample: {grid['sample_time']})",
        aspect="auto",
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickvals=[i + 0.5 for i in range(len(RAIN_BIN_LABELS))],
            ticktext=list(RAIN_BIN_LABELS),
        )
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Colour breakpoints are IMD's real published 24-hour "
        "rainfall-warning thresholds (7.5 / 64.5 / 115.6 / 204.5mm) -- "
        "see docs/phase7-dashboard-scope.md section 3."
    )
