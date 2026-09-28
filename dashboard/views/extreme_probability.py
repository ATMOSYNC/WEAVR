"""Extreme-probability-map view: P(rain > 204.5mm) per gridpoint, from a
real fitted EMOS-CSG distribution, in IMD's real colour identities.

This view diverges from the blended-map's Tier 1 combiner (step 3):
Tier 1's deterministic regional blend has no predictive distribution to
compute an exceedance probability from at all. Instead this view uses
EMOS-CSG's `ifs_ens` combiner -- fit on the real 50-member IFS ensemble
(`weavr.emos.fit_emos_csg`), not BMA -- because the censored-shifted-gamma
has a real closed-form survival function
(`weavr.emos.exceedance_probability_csgd`), unlike BMA's mixture, which has
no closed form (`weavr.bma`'s own docstring) and would need a Monte Carlo
estimate per gridpoint. This choice was routed to the user via
`AskUserQuestion` rather than picked silently.

204.5mm is IMD's own real "extremely heavy rain" boundary
(`weavr.verify.IMD_RAIN_THRESHOLDS_MM[-1]`), the most literal reading of
issue #9's "extreme."

`docs/phase4-data-and-combiner-scope.md` already found the extremely_heavy
bin is never fittable at any lead in this project's real 2020 JJAS data (0
train days at every lead, checked directly again while building this
view's export). Cells whose own forecast fell in a bin EMOS-CSG could not
fit for real are rendered with a visible grey overlay, not a confident-
looking probability number.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.colors import PROBABILITY_COLORSCALE, fallback_overlay
from dashboard.data_loading import available_probability_grid_leads, load_probability_grid


def render() -> None:
    st.subheader("Extreme-probability map: P(rain > 204.5mm)")
    st.caption(
        "This map shows EMOS-CSG's real fitted `ifs_ens` combiner (the "
        "real 50-member IFS ensemble) -- chosen over BMA because its "
        "censored-shifted-gamma has a real closed-form exceedance "
        "probability, unlike BMA's mixture. This diverges from the "
        "blended map's Tier 1 combiner, which has no predictive "
        "distribution to compute a probability from. 204.5mm is IMD's own "
        "real 'extremely heavy rain' boundary. Grey-hatched cells have no "
        "real fitted probability -- their own forecast fell in a "
        "rain-intensity bin EMOS-CSG could not fit for real at this lead "
        "(docs/phase4-data-and-combiner-scope.md found the extremely_heavy "
        "bin is never fittable at any lead in this project's real data)."
    )

    leads = available_probability_grid_leads()
    lead = st.selectbox("Lead time (hours)", leads, index=0, key="extreme_probability_lead")
    grid = load_probability_grid(lead)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=grid["probability"],
            x=grid["longitude"],
            y=grid["latitude"],
            colorscale=PROBABILITY_COLORSCALE,
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "P(rain > 204.5mm)"},
            hovertemplate="lat=%{y}<br>lon=%{x}<br>P=%{z:.3f}<extra></extra>",
        )
    )
    overlay = fallback_overlay(grid["is_fallback"])
    if not np.all(np.isnan(overlay)):
        fig.add_trace(
            go.Heatmap(
                z=overlay,
                x=grid["longitude"],
                y=grid["latitude"],
                colorscale=[[0.0, "rgba(120,120,120,0.75)"], [1.0, "rgba(120,120,120,0.75)"]],
                showscale=False,
                hovertemplate="fallback cell (not fittable at this lead)<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"P(rain > 204.5mm) at lead {lead}h (sample: {grid['sample_time']})",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )
    st.plotly_chart(fig, width="stretch")

    n_fallback = int(np.sum(grid["is_fallback"]))
    if n_fallback:
        st.warning(
            f"{n_fallback} gridpoint(s) at this lead are shown grey: their "
            "forecast fell in a rain-intensity bin with too few real "
            "training days to fit EMOS-CSG (see caption above)."
        )
