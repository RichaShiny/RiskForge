import plotly.express as px
import streamlit as st

from src.data_generator import generate_protocol_positions
from src.tail_risk_attribution import run_tail_risk_attribution


st.set_page_config(
    page_title="RiskForge · Tail Risk Attribution",
    page_icon="🧭",
    layout="wide",
)

st.title("Tail Risk Attribution")

st.write(
    "Identify which collateral assets contribute most to modeled cascade tail risk. "
    "Each simulated market draw is evaluated once in full and then re-run with one "
    "asset's exogenous shock neutralized while every other assumption stays fixed."
)

st.info(
    "Leave-one-asset-out contributions are marginal counterfactual effects, not an "
    "additive decomposition. Cascade interactions can overlap, so ETH + BTC + SOL "
    "contributions do not have to equal total protocol risk."
)

positions = generate_protocol_positions()

with st.sidebar:
    st.header("Simulation")

    n_simulations = st.select_slider(
        "Market draws",
        options=[10, 25, 50],
        value=25,
    )

    horizon_days = st.select_slider(
        "Horizon (days)",
        options=[7, 14, 30, 60, 90],
        value=30,
    )

    distribution_label = st.radio(
        "Return distribution",
        options=["Student-t", "Normal"],
    )
    distribution = (
        "student_t" if distribution_label == "Student-t" else "normal"
    )

    tail_percentile = st.select_slider(
        "Tail cohort",
        options=[75, 90, 95],
        value=95,
        format_func=lambda value: f"Top {100 - value}% risk tail (P{value}+)",
    )

    st.header("Market depth")

    eth_depth_m = st.slider(
        "ETH depth ($M)",
        min_value=5,
        max_value=200,
        value=50,
        step=5,
    )
    btc_depth_m = st.slider(
        "BTC depth ($M)",
        min_value=5,
        max_value=250,
        value=75,
        step=5,
    )
    sol_depth_m = st.slider(
        "SOL depth ($M)",
        min_value=2,
        max_value=100,
        value=20,
        step=2,
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
        format="%.2f",
    )
    price_impact_factor = st.slider(
        "Price-impact strength",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

market_depth = {
    "ETH": eth_depth_m * 1_000_000.0,
    "BTC": btc_depth_m * 1_000_000.0,
    "SOL": sol_depth_m * 1_000_000.0,
}

with st.spinner("Running paired asset attribution..."):
    result = run_tail_risk_attribution(
        positions=positions,
        market_depth_usd=market_depth,
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        seed=42,
        distribution=distribution,
        tail_quantile=tail_percentile / 100,
        close_factor=close_factor,
        liquidation_bonus=liquidation_bonus,
        price_impact_factor=price_impact_factor,
    )

summary = result.asset_summary.copy()
summary["mean_contribution_pct"] = (
    summary["mean_cascade_exposure_contribution"] * 100
)
summary["tail_contribution_pct"] = (
    summary["tail_mean_cascade_exposure_contribution"] * 100
)
summary["tail_bad_debt_contribution_pct"] = (
    summary["tail_mean_bad_debt_contribution"] * 100
)
summary["positive_contribution_probability_pct"] = (
    summary["probability_positive_exposure_contribution"] * 100
)

scenario_results = result.scenario_results
full_tail_cutoff = scenario_results["cascade_debt_share"].quantile(
    tail_percentile / 100
)

leader = summary.iloc[0]
mean_exposure = scenario_results["cascade_debt_share"].mean()
p95_exposure = scenario_results["cascade_debt_share"].quantile(0.95)
mean_bad_debt = scenario_results["bad_debt_share"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Largest tail contributor", leader["asset"])
col2.metric("Mean cascade exposure", f"{mean_exposure:.1%}")
col3.metric("P95 cascade exposure", f"{p95_exposure:.1%}")
col4.metric("Mean bad debt", f"{mean_bad_debt:.1%}")

st.caption(
    f"{n_simulations} paired draws · {horizon_days}-day horizon · "
    f"{distribution_label} returns · P{tail_percentile}+ tail cohort begins at "
    f"{full_tail_cutoff:.1%} cascade exposure"
)

st.subheader("Asset Contribution to Cascade Exposure")

contribution_chart = px.bar(
    summary,
    x="asset",
    y=["mean_contribution_pct", "tail_contribution_pct"],
    barmode="group",
    labels={
        "asset": "Collateral Asset",
        "value": "Marginal Cascade Exposure Contribution (percentage points)",
        "variable": "Contribution Window",
    },
    title="Average vs Tail-Conditioned Marginal Contribution",
)

st.plotly_chart(contribution_chart, width="stretch")

st.subheader("Tail Bad-Debt Contribution")

bad_debt_chart = px.bar(
    summary,
    x="asset",
    y="tail_bad_debt_contribution_pct",
    labels={
        "asset": "Collateral Asset",
        "tail_bad_debt_contribution_pct": (
            "Tail Mean Bad-Debt Contribution (percentage points)"
        ),
    },
    title="Leave-One-Asset-Out Contribution in the Bad-Debt Tail",
)

st.plotly_chart(bad_debt_chart, width="stretch")

st.subheader("Attribution Summary")

display = summary[
    [
        "asset",
        "mean_contribution_pct",
        "tail_contribution_pct",
        "p95_cascade_exposure_contribution",
        "positive_contribution_probability_pct",
        "tail_bad_debt_contribution_pct",
        "mean_secondary_liquidation_contribution",
        "mean_endogenous_price_decline_contribution",
    ]
].copy()

display["p95_cascade_exposure_contribution"] *= 100
display["mean_endogenous_price_decline_contribution"] *= 100

display = display.rename(
    columns={
        "asset": "Asset",
        "mean_contribution_pct": "Mean exposure contribution (pts)",
        "tail_contribution_pct": "Tail mean exposure contribution (pts)",
        "p95_cascade_exposure_contribution": "P95 exposure contribution (pts)",
        "positive_contribution_probability_pct": "P(positive contribution) (%)",
        "tail_bad_debt_contribution_pct": "Tail mean bad-debt contribution (pts)",
        "mean_secondary_liquidation_contribution": "Mean secondary liquidation contribution",
        "mean_endogenous_price_decline_contribution": "Mean price-decline contribution (pts)",
    }
)

st.dataframe(display, width="stretch", hide_index=True)

st.subheader("Scenario-Level Attribution")

asset_choice = st.selectbox(
    "Inspect one collateral asset",
    options=summary["asset"].tolist(),
)

asset_rows = result.attribution_results.loc[
    result.attribution_results["asset"] == asset_choice
].copy()
asset_rows["cascade_exposure_contribution_pct"] = (
    asset_rows["contribution_cascade_debt_share"] * 100
)
asset_rows["bad_debt_contribution_pct"] = (
    asset_rows["contribution_bad_debt_share"] * 100
)

scenario_chart = px.scatter(
    asset_rows,
    x="asset_shock",
    y="cascade_exposure_contribution_pct",
    size=asset_rows["contribution_cascade_created_positions"].abs() + 1,
    labels={
        "asset_shock": f"{asset_choice} Simulated Return",
        "cascade_exposure_contribution_pct": (
            "Marginal Cascade Exposure Contribution (pts)"
        ),
    },
    title=f"{asset_choice} Shock vs Marginal Cascade Contribution",
)

st.plotly_chart(scenario_chart, width="stretch")

st.write(
    "A negative contribution is possible. It means neutralizing that asset's return "
    "made the modeled protocol outcome worse in that particular paired scenario, "
    "usually because the original asset return was positive or because nonlinear "
    "cascade interactions changed the liquidation path."
)
