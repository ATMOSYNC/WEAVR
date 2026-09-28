"""Weight-map view: real per-region, per-lead source weights (issue #9's
"core interpretability deliverable").

This project's real weighting scheme, per `docs/phase6-operational-scope.md`'s
decision for the daily pipeline, is Phase 3's `fit_region_weights` -- a
per-region ordinary-least-squares fit -- not the softmax/GBM gate issue #9's
own phrasing happens to reference. This view exists to explain *that* real,
already-fitted scheme, not a hypothetical one.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.data_loading import load_weight_map_data


def render() -> None:
    st.subheader("Weight map: per-region, per-lead source weights")
    st.caption(
        "This project's real weighting scheme is Phase 3's `fit_region_weights` "
        "-- a per-region ordinary-least-squares fit over graphcast / hres / "
        "ifs_ens_mean's own real historical errors, not a softmax or GBM gate. "
        "Regions shown hatched fell back to an equal split (not enough real "
        "training points to fit a weight) -- hover for the real reason."
    )

    data = load_weight_map_data()
    leads = sorted(data["lead_hours"].unique())
    lead = st.selectbox("Lead time (hours)", leads, index=0)
    subset = data[data["lead_hours"] == lead].copy()
    subset["fit_status"] = subset["is_fallback"].map(
        {True: "fallback (equal split)", False: "fitted (OLS)"}
    )

    fig = px.bar(
        subset,
        x="region",
        y="weight",
        color="source",
        pattern_shape="fit_status",
        hover_data=["reason", "n_train_points"],
        title=f"Regional source weights at lead {lead}h",
    )
    st.plotly_chart(fig, width="stretch")

    fallback_regions = sorted(subset.loc[subset["is_fallback"], "region"].unique())
    if fallback_regions:
        st.warning(
            "Fallback regions at this lead (equal split, not a real fit): "
            + ", ".join(fallback_regions)
        )
