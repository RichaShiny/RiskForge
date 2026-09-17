import pandas as pd
import plotly.express as px
import streamlit as st

from src.cascade_simulation import (
    run_cascade_aware_monte_carlo,
    run_market_depth_sensitivity,
    summarize_cascade_monte_carlo,
)
from src.data_generator import generate_protocol_positions


st.set_page_config(
    page_title="RiskForge · Cascade Tail Risk",
    page_icon="📉",
    layout="wide",
)

st.title("Cascade Tail Risk")
st.write(
    "Compare first-order liquidation exposure with endogenous liquidation "
    "feedback on the exact same simulated ETH, BTC, and SOL market draws."
)
st.caption(
    "The cascade layer is a transparent stress model. Market-depth and "
    "price-impact assumptions are scenario inputs, not forecasts of execution."
)

positions = generate_protocol_positions(
    n_positions=250,
    seed=42,
)

with st.sidebar:
    st.header("Simulation")

    n_simulations = st.select_slider(
        "Simulations",
        options=[50, 100, 250],
        value=100,
    )
    horizon_days = st.select_slider(
        "Horizon (days)",
        options=[7, 14, 30, 60, 90],
        value=30,
    )
    return_model = st.radio(
        "Return model",
        options=["Student-t", "Normal"],
    )
    distribution = (
        "student_t"
        if return_model == "Student-t"
        else "normal"
    )

    st.header("Liquidation mechanics")
    close_factor = st.slider(
        "Close factor",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
    )
    liquidation_bonus = st.slider(
        "Liquidation bonus",
        min_value=0.00,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )
    price_impact_factor = st.slider(
        "Price-impact strength",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    max_rounds = st.slider(
        "Maximum cascade rounds",
        min_value=2,
        max_value=20,
        value=10,
        step=1,
    )

    st.header("Market depth")
    eth_depth_m = st.number_input(
        "ETH depth ($M)",
        min_value=0.1,
        value=10.0,
        step=0.5,
    )
    btc_depth_m = st.number_input(
        "BTC depth ($M)",
        min_value=0.1,
        value=15.0,
        step=0.5,
    )
    sol_depth_m = st.number_input(
        "SOL depth ($M)",
        min_value=0.1,
        value=5.0,
        step=0.5,
    )

market_depth = {
    "ETH": eth_depth_m * 1_000_000,
    "BTC": btc_depth_m * 1_000_000,
    "SOL": sol_depth_m * 1_000_000,
}

with st.spinner("Running paired cascade Monte Carlo stress test..."):
    results = run_cascade_aware_monte_carlo(
        positions=positions,
        market_depth_usd=market_depth,
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        seed=42,
        distribution=distribution,
        close_factor=close_factor,
        liquidation_bonus=liquidation_bonus,
        price_impact_factor=price_impact_factor,
        max_rounds=max_rounds,
    )
    summary = summarize_cascade_monte_carlo(results)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Mean first-order exposure",
    f"{summary['mean_first_order_debt_share']:.1%}",
)
col2.metric(
    "Mean cascade exposure",
    f"{summary['mean_cascade_debt_share']:.1%}",
    delta=f"+{summary['mean_amplification_debt_share']:.1%} of protocol debt",
)
col3.metric(
    "P(Cascade amplification)",
    f"{summary['probability_of_cascade_amplification']:.1%}",
)
col4.metric(
    "P95 bad-debt share",
    f"{summary['p95_bad_debt_share']:.1%}",
)

st.subheader("First-order vs Cascade Exposure Distribution")
comparison = pd.concat(
    [
        results[["simulation_id", "first_order_debt_share"]]
        .rename(columns={"first_order_debt_share": "debt_share"})
        .assign(model="First-order"),
        results[["simulation_id", "cascade_debt_share"]]
        .rename(columns={"cascade_debt_share": "debt_share"})
        .assign(model="Cascade-aware"),
    ],
    ignore_index=True,
)
comparison["debt_share_pct"] = comparison["debt_share"] * 100

fig = px.histogram(
    comparison,
    x="debt_share_pct",
    color="model",
    barmode="overlay",
    opacity=0.60,
    nbins=35,
    labels={
        "debt_share_pct": "Protocol debt exposed (%)",
        "model": "Stress model",
    },
    title="Paired Liquidation Exposure Distribution",
)
st.plotly_chart(fig, width="stretch")

st.subheader("Amplification by Simulated Market Draw")
amplification_display = results.copy()
amplification_display["first_order_pct"] = (
    amplification_display["first_order_debt_share"] * 100
)
amplification_display["cascade_pct"] = (
    amplification_display["cascade_debt_share"] * 100
)
amplification_display["amplification_pct"] = (
    amplification_display["amplification_debt_share"] * 100
)

scatter = px.scatter(
    amplification_display,
    x="first_order_pct",
    y="cascade_pct",
    size="cascade_created_positions",
    color="amplification_pct",
    hover_data=[
        "eth_return",
        "btc_return",
        "sol_return",
        "bad_debt_share",
        "rounds_executed",
    ],
    labels={
        "first_order_pct": "First-order exposure (%)",
        "cascade_pct": "Cascade-aware exposure (%)",
        "amplification_pct": "Amplification (pts)",
        "cascade_created_positions": "Cascade-created positions",
    },
    title="How Endogenous Feedback Changes the Same Market Shock",
)
st.plotly_chart(scatter, width="stretch")

st.subheader("Tail Risk")
tail_table = pd.DataFrame(
    [
        {
            "Metric": "P95 exposure",
            "First-order": summary["p95_first_order_debt_share"],
            "Cascade-aware": summary["p95_cascade_debt_share"],
        },
        {
            "Metric": "P99 exposure",
            "First-order": summary["p99_first_order_debt_share"],
            "Cascade-aware": summary["p99_cascade_debt_share"],
        },
        {
            "Metric": "P(Exposure > 25%)",
            "First-order": summary["probability_first_order_exposure_gt_25"],
            "Cascade-aware": summary["probability_cascade_exposure_gt_25"],
        },
        {
            "Metric": "P(Exposure > 50%)",
            "First-order": summary["probability_first_order_exposure_gt_50"],
            "Cascade-aware": summary["probability_cascade_exposure_gt_50"],
        },
        {
            "Metric": "P(Exposure > 75%)",
            "First-order": summary["probability_first_order_exposure_gt_75"],
            "Cascade-aware": summary["probability_cascade_exposure_gt_75"],
        },
    ]
)
for column in ["First-order", "Cascade-aware"]:
    tail_table[column] = tail_table[column].map(lambda value: f"{value:.1%}")
st.dataframe(tail_table, width="stretch", hide_index=True)

st.subheader("Cascade Diagnostics")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "P95 tail amplification",
    f"{summary['p95_tail_amplification']:.1%}",
)
col2.metric(
    "P99 tail amplification",
    f"{summary['p99_tail_amplification']:.1%}",
)
col3.metric(
    "Mean cascade-created positions",
    f"{summary['mean_cascade_created_positions']:.1f}",
)
col4.metric(
    "P95 endogenous price decline",
    f"{summary['p95_endogenous_price_decline']:.1%}",
)

with st.expander("Market-depth sensitivity"):
    st.write(
        "Re-run the same simulated market draws under shallower and deeper "
        "liquidity assumptions. Lower multipliers represent less market depth."
    )

    if st.button("Run depth sensitivity"):
        with st.spinner("Sweeping market-depth assumptions..."):
            sensitivity = run_market_depth_sensitivity(
                positions=positions,
                base_market_depth_usd=market_depth,
                depth_multipliers=[0.5, 1.0, 2.0, 4.0],
                n_simulations=min(n_simulations, 100),
                horizon_days=horizon_days,
                seed=42,
                distribution=distribution,
                close_factor=close_factor,
                liquidation_bonus=liquidation_bonus,
                price_impact_factor=price_impact_factor,
                max_rounds=max_rounds,
            )

        depth_chart = px.line(
            sensitivity,
            x="depth_multiplier",
            y=[
                "p95_cascade_debt_share",
                "mean_cascade_debt_share",
            ],
            markers=True,
            labels={
                "depth_multiplier": "Market-depth multiplier",
                "value": "Protocol debt exposure share",
                "variable": "Risk metric",
            },
            title="Liquidity Depth vs Cascade Exposure",
        )
        st.plotly_chart(depth_chart, width="stretch")
        st.dataframe(sensitivity, width="stretch", hide_index=True)
