import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_generator import generate_protocol_positions
from src.reverse_stress import (
    build_reverse_stress_frontier,
    find_critical_decline,
)


st.set_page_config(
    page_title="RiskForge · Reverse Stress Test",
    page_icon="🎯",
    layout="wide",
)

st.title("Reverse Stress Test")
st.write(
    "Solve for the smallest modeled market decline that breaches a chosen "
    "protocol risk threshold."
)
st.caption(
    "Reverse-stress thresholds are model-based breakpoints under the selected "
    "liquidity and liquidation assumptions. They are not market forecasts."
)

positions = generate_protocol_positions(n_positions=250, seed=42)

metric_labels = {
    "Cascade-aware debt exposure": "cascade_debt_share",
    "First-order debt exposure": "first_order_debt_share",
    "Cascade amplification": "amplification_debt_share",
    "Bad-debt share": "bad_debt_share",
}

with st.sidebar:
    st.header("Risk target")
    metric_label = st.selectbox(
        "Metric",
        options=list(metric_labels),
        index=0,
    )
    metric = metric_labels[metric_label]

    target_pct = st.slider(
        "Target threshold (%)",
        min_value=1,
        max_value=90,
        value=50 if metric != "bad_debt_share" else 10,
        step=1,
    )
    max_decline_pct = st.slider(
        "Maximum decline searched (%)",
        min_value=10,
        max_value=95,
        value=90,
        step=5,
    )
    tolerance_pct = st.select_slider(
        "Search tolerance (%)",
        options=[0.05, 0.10, 0.25, 0.50, 1.00],
        value=0.10,
    )

    st.header("Asset stress weights")
    st.caption(
        "A weight of 1.0 applies the base decline. Values above 1.0 make an "
        "asset fall more than the base stress."
    )
    eth_weight = st.slider("ETH weight", 0.0, 2.0, 1.0, 0.1)
    btc_weight = st.slider("BTC weight", 0.0, 2.0, 1.0, 0.1)
    sol_weight = st.slider("SOL weight", 0.0, 2.0, 1.0, 0.1)

    st.header("Market depth")
    eth_depth_m = st.number_input(
        "ETH depth ($M)", min_value=0.1, value=10.0, step=0.5
    )
    btc_depth_m = st.number_input(
        "BTC depth ($M)", min_value=0.1, value=15.0, step=0.5
    )
    sol_depth_m = st.number_input(
        "SOL depth ($M)", min_value=0.1, value=5.0, step=0.5
    )

    st.header("Liquidation mechanics")
    close_factor = st.slider("Close factor", 0.10, 1.00, 0.50, 0.05)
    liquidation_bonus = st.slider(
        "Liquidation bonus", 0.00, 0.20, 0.05, 0.01
    )
    price_impact_factor = st.slider(
        "Price-impact strength", 0.0, 3.0, 1.0, 0.1
    )
    max_rounds = st.slider(
        "Maximum cascade rounds", 2, 20, 10, 1
    )

market_depth = {
    "ETH": eth_depth_m * 1_000_000,
    "BTC": btc_depth_m * 1_000_000,
    "SOL": sol_depth_m * 1_000_000,
}
asset_weights = {
    "ETH": eth_weight,
    "BTC": btc_weight,
    "SOL": sol_weight,
}
solver_kwargs = {
    "positions": positions,
    "market_depth_usd": market_depth,
    "asset_stress_weights": asset_weights,
    "max_decline": max_decline_pct / 100,
    "tolerance": tolerance_pct / 100,
    "close_factor": close_factor,
    "liquidation_bonus": liquidation_bonus,
    "price_impact_factor": price_impact_factor,
    "max_rounds": max_rounds,
}

with st.spinner("Solving reverse-stress threshold..."):
    result = find_critical_decline(
        metric=metric,
        target=target_pct / 100,
        **solver_kwargs,
    )

if result.found and result.point is not None:
    point = result.point
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Critical base decline",
        f"{result.critical_decline:.2%}",
    )
    col2.metric(
        metric_label,
        f"{result.metric_value:.1%}",
        delta=f"target {target_pct}%",
    )
    col3.metric(
        "Cascade-created positions",
        f"{point.cascade_created_positions:,}",
    )
    col4.metric(
        "Bad-debt share at breach",
        f"{point.bad_debt_share:.1%}",
    )

    st.subheader("Critical Shock Vector")
    shock_table = pd.DataFrame(
        [
            {
                "Asset": asset,
                "Stress weight": asset_weights[asset],
                "Critical price shock (%)": shock * 100,
            }
            for asset, shock in point.shocks.items()
        ]
    )
    st.dataframe(shock_table, width="stretch", hide_index=True)

    st.subheader("Safe-to-Breach Bracket")
    bracket = pd.DataFrame(
        [
            {
                "State": "Largest tested safe shock",
                "Base decline (%)": (
                    result.lower_safe_decline * 100
                    if result.lower_safe_decline is not None
                    else None
                ),
                "Metric value (%)": (
                    result.lower_safe_value * 100
                    if result.lower_safe_value is not None
                    else None
                ),
            },
            {
                "State": "Smallest tested breach",
                "Base decline (%)": result.critical_decline * 100,
                "Metric value (%)": result.metric_value * 100,
            },
        ]
    )
    st.dataframe(bracket, width="stretch", hide_index=True)
else:
    st.warning(
        "The selected risk target was not breached within the configured "
        f"{max_decline_pct}% maximum decline."
    )

st.subheader("Reverse-Stress Frontier")
st.write(
    "Solve several thresholds for the same risk metric to see how the critical "
    "market decline changes as the target becomes more severe."
)

def _frontier_targets(selected_metric: str) -> list[float]:
    if selected_metric == "bad_debt_share":
        return [0.01, 0.05, 0.10, 0.20, 0.30]
    if selected_metric == "amplification_debt_share":
        return [0.01, 0.05, 0.10, 0.20, 0.30]
    return [0.10, 0.25, 0.50, 0.75]


with st.spinner("Building reverse-stress frontier..."):
    frontier = build_reverse_stress_frontier(
        metric=metric,
        targets=_frontier_targets(metric),
        **solver_kwargs,
    )

frontier_display = frontier.copy()
frontier_display["target_pct"] = frontier_display["target"] * 100
frontier_display["critical_decline_pct"] = (
    frontier_display["critical_decline"] * 100
)

found_frontier = frontier_display[frontier_display["found"]].copy()
if not found_frontier.empty:
    fig = px.line(
        found_frontier,
        x="target_pct",
        y="critical_decline_pct",
        markers=True,
        labels={
            "target_pct": "Risk threshold (%)",
            "critical_decline_pct": "Critical base decline (%)",
        },
        title="Risk Threshold vs Critical Market Decline",
    )
    st.plotly_chart(fig, width="stretch")

st.dataframe(
    frontier_display[
        [
            "target_pct",
            "found",
            "critical_decline_pct",
            "metric_value",
            "cascade_created_positions",
            "bad_debt_share_at_breach",
        ]
    ],
    width="stretch",
    hide_index=True,
)
