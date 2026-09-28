"""Skill-trend view: real per-lead RMSE/CRPS across all 4 tiers' methods.

`dashboard.data.load_skill_trends_data` does not double-plot tier3's
duplicated emos_graphcast/emos_ifs_ens/bma reference numbers (checked:
bit-for-bit identical to tier2's own values) -- tier3 contributes only its
own new `regime_conditioned` method here.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.data_loading import load_skill_trends_data


def render() -> None:
    st.subheader("Skill trend charts: RMSE / CRPS across tiers 0-3")
    st.caption(
        "The real, checked outcome across tiers is mixed, not monotonically "
        "improving: tier1 beats tier0 domain-wide at every lead but not in "
        "every region x lead cell; tier2's EMOS-CSG and BMA trade off which "
        "wins; tier3's regime-conditioned reweighting does not beat tier2 at "
        "any lead (0 of 5). This chart plots every real method's own "
        "numbers -- it is not smoothed into a single improving trend line."
    )

    data = load_skill_trends_data()
    metric = st.radio("Metric", ["rmse_mm", "crps_mm"], horizontal=True)
    subset = data[data["metric"] == metric]
    if subset.empty:
        st.info(
            "No tier reports this metric here "
            "(tier0/tier1 are deterministic blends with no CRPS)."
        )
        return

    fig = px.line(
        subset.sort_values("lead_hours"),
        x="lead_hours",
        y="value",
        color="method",
        markers=True,
        title=f"{metric} by lead time, every real method across tiers 0-3",
    )
    st.plotly_chart(fig, use_container_width=True)
