import streamlit as st
import plotly.express as px

from src.calibration import (
    calibrate_market_parameters,
    download_historical_prices,
)
from src.data_generator import generate_protocol_positions
from src.simulation import (
    run_monte_carlo_simulation,
    run_threshold_sensitivity,
)
from src.stress_engine import (
    run_asset_specific_stress_test,
    run_protocol_shock_ladder,
)


st.set_page_config(
    page_title="RiskForge",
    page_icon="📊",
    layout="wide",
)

st.title("RiskForge")

st.write(
    "A quantitative protocol risk and stress-testing engine for "
    "simulating market shocks, estimating liquidation exposure, "
    "and analyzing systemic risk under uncertainty."
)

positions = generate_protocol_positions()

shocks = [
    0,
    -0.05,
    -0.10,
    -0.20,
    -0.30,
    -0.40,
    -0.50,
]

ladder = run_protocol_shock_ladder(
    positions,
    shocks,
)

total_positions = len(positions)
total_debt = positions["debt_usd"].sum()
total_collateral = positions["collateral_value"].sum()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Positions",
    f"{total_positions:,}",
)

col2.metric(
    "Total debt",
    f"${total_debt:,.0f}",
)

col3.metric(
    "Total collateral",
    f"${total_collateral:,.0f}",
)

st.subheader("Protocol Stress Test")

display_ladder = ladder.copy()

display_ladder["market_decline_pct"] = (
    -display_ladder["shock_pct"] * 100
)

display_ladder["liquidation_rate_pct"] = (
    display_ladder["liquidation_rate"] * 100
)

display_ladder["liquidatable_debt_share_pct"] = (
    display_ladder["liquidatable_debt_share"] * 100
)

display_ladder = display_ladder.sort_values(
    "market_decline_pct"
)

fig = px.line(
    display_ladder,
    x="market_decline_pct",
    y="liquidatable_debt_share_pct",
    markers=True,
    labels={
        "market_decline_pct": "Market Decline (%)",
        "liquidatable_debt_share_pct": "Liquidatable Debt Share (%)",
    },
    title="Market Decline vs Liquidation Exposure",
)

st.plotly_chart(
    fig,
    width="stretch",
)

st.subheader("Stress Scenario Summary")

stress_table = display_ladder[
    [
        "market_decline_pct",
        "liquidatable_positions",
        "liquidation_rate_pct",
        "liquidatable_debt",
        "liquidatable_debt_share_pct",
    ]
].copy()

stress_table = stress_table.rename(
    columns={
        "market_decline_pct": "Market decline (%)",
        "liquidatable_positions": "Liquidatable positions",
        "liquidation_rate_pct": "Liquidation rate (%)",
        "liquidatable_debt": "Liquidatable debt ($)",
        "liquidatable_debt_share_pct": "Debt exposure (%)",
    }
)

st.dataframe(
    stress_table,
    width="stretch",
)

st.subheader("Interactive Market Scenario")

st.write(
    "Adjust collateral prices independently to explore "
    "how different market conditions affect liquidation risk."
)

eth_shock = st.slider(
    "ETH price shock (%)",
    min_value=-80,
    max_value=20,
    value=-25,
    step=5,
)

btc_shock = st.slider(
    "BTC price shock (%)",
    min_value=-80,
    max_value=20,
    value=-15,
    step=5,
)

sol_shock = st.slider(
    "SOL price shock (%)",
    min_value=-80,
    max_value=20,
    value=-40,
    step=5,
)

asset_shocks = {
    "ETH": eth_shock / 100,
    "BTC": btc_shock / 100,
    "SOL": sol_shock / 100,
}

scenario = run_asset_specific_stress_test(
    positions,
    asset_shocks,
)

liquidatable = scenario[
    scenario["liquidatable"]
]

scenario_liquidatable_positions = len(
    liquidatable
)

scenario_liquidatable_debt = (
    liquidatable["debt_usd"].sum()
)

scenario_liquidation_rate = (
    scenario["liquidatable"].mean()
)

scenario_debt_share = (
    scenario_liquidatable_debt
    / scenario["debt_usd"].sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Liquidatable Positions",
    f"{scenario_liquidatable_positions:,}",
)

col2.metric(
    "Position Liquidation Rate",
    f"{scenario_liquidation_rate:.1%}",
)

col3.metric(
    "Liquidatable Debt",
    f"${scenario_liquidatable_debt:,.0f}",
)

col4.metric(
    "Debt Exposure",
    f"{scenario_debt_share:.1%}",
)

asset_risk = (
    scenario.groupby("asset")
    .agg(
        total_positions=(
            "position_id",
            "count",
        ),
        liquidatable_positions=(
            "liquidatable",
            "sum",
        ),
        total_debt=(
            "debt_usd",
            "sum",
        ),
    )
    .reset_index()
)

liquidatable_debt_by_asset = (
    liquidatable.groupby("asset")[
        "debt_usd"
    ].sum()
)

asset_risk["liquidatable_debt"] = (
    asset_risk["asset"]
    .map(liquidatable_debt_by_asset)
    .fillna(0)
)

asset_risk["liquidation_rate_pct"] = (
    asset_risk["liquidatable_positions"]
    / asset_risk["total_positions"]
    * 100
)

st.subheader("Liquidation Risk by Asset")

asset_table = asset_risk.copy()

asset_table = asset_table.rename(
    columns={
        "asset": "Asset",
        "total_positions": "Positions",
        "liquidatable_positions": "Liquidatable positions",
        "total_debt": "Total debt ($)",
        "liquidatable_debt": "Liquidatable debt ($)",
        "liquidation_rate_pct": "Liquidation rate (%)",
    }
)

st.dataframe(
    asset_table,
    width="stretch",
)

asset_chart = px.bar(
    asset_risk,
    x="asset",
    y="liquidatable_debt",
    labels={
        "asset": "Collateral Asset",
        "liquidatable_debt": "Liquidatable Debt ($)",
    },
    title="Liquidatable Debt by Asset",
)

st.plotly_chart(
    asset_chart,
    width="stretch",
)

st.subheader("Monte Carlo Risk Simulation")

st.write(
    "Simulate correlated ETH, BTC, and SOL market outcomes "
    "to estimate the distribution of liquidation exposure."
)

n_simulations = st.select_slider(
    "Number of simulations",
    options=[
        250,
        500,
        1000,
        2000,
    ],
    value=1000,
)

horizon_days = st.select_slider(
    "Simulation horizon",
    options=[
        7,
        14,
        30,
        60,
        90,
    ],
    value=30,
)

return_model = st.radio(
    "Return distribution",
    options=[
        "Student-t",
        "Normal",
    ],
    horizontal=True,
)

distribution = (
    "student_t"
    if return_model == "Student-t"
    else "normal"
)

calibration_source = st.radio(
    "Calibration source",
    options=[
        "Historical",
        "Assumed",
    ],
    horizontal=True,
)

annual_volatility = None
correlation_matrix = None
historical_volatility = None
historical_correlation = None

if calibration_source == "Historical":
    with st.spinner(
        "Loading historical crypto market data..."
    ):
        prices = download_historical_prices(
            start="2022-01-01",
        )

        (
            historical_volatility,
            historical_correlation,
        ) = calibrate_market_parameters(
            prices
        )

        annual_volatility = (
            historical_volatility.to_dict()
        )

        correlation_matrix = (
            historical_correlation.to_numpy()
        )

mc = run_monte_carlo_simulation(
    positions,
    n_simulations=n_simulations,
    horizon_days=horizon_days,
    distribution=distribution,
    annual_volatility=annual_volatility,
    correlation_matrix=correlation_matrix,
)

if calibration_source == "Historical":
    with st.expander(
        "Historical Calibration Details"
    ):
        st.write(
            "Calibration window: 2022-01-01 "
            "to latest available date."
        )

        st.write(
            "Annualized volatility"
        )

        volatility_table = (
            historical_volatility
            .rename(
                "annualized_volatility"
            )
            .to_frame()
        )

        volatility_table[
            "annualized_volatility_pct"
        ] = (
            volatility_table[
                "annualized_volatility"
            ]
            * 100
        )

        st.dataframe(
            volatility_table[
                [
                    "annualized_volatility_pct",
                ]
            ].rename(
                columns={
                    "annualized_volatility_pct":
                        "Annualized volatility (%)"
                }
            ),
            width="stretch",
        )

        st.write(
            "Return correlation matrix"
        )

        st.dataframe(
            historical_correlation,
            width="stretch",
        )

mean_exposure = (
    mc["liquidatable_debt_share"].mean()
)

median_exposure = (
    mc["liquidatable_debt_share"].median()
)

severe_probability = (
    mc["liquidatable_debt_share"] > 0.25
).mean()

extreme_probability = (
    mc["liquidatable_debt_share"] > 0.50
).mean()

catastrophic_probability = (
    mc["liquidatable_debt_share"] > 0.75
).mean()

max_exposure = (
    mc["liquidatable_debt_share"].max()
)

col1, col2, col3, col4, col5 = (
    st.columns(5)
)

col1.metric(
    "Mean Exposure",
    f"{mean_exposure:.1%}",
)

col2.metric(
    "P(Exposure > 25%)",
    f"{severe_probability:.1%}",
)

col3.metric(
    "P(Exposure > 50%)",
    f"{extreme_probability:.1%}",
)

col4.metric(
    "P(Exposure > 75%)",
    f"{catastrophic_probability:.1%}",
)

col5.metric(
    "Max Exposure",
    f"{max_exposure:.1%}",
)

st.caption(
    f"{return_model} returns · "
    f"{calibration_source.lower()} calibration · "
    f"{n_simulations:,} simulations · "
    f"{horizon_days}-day horizon"
)

mc_display = mc.copy()

mc_display[
    "liquidatable_debt_share_pct"
] = (
    mc_display[
        "liquidatable_debt_share"
    ]
    * 100
)

mc_hist = px.histogram(
    mc_display,
    x="liquidatable_debt_share_pct",
    nbins=40,
    labels={
        "liquidatable_debt_share_pct":
            "Liquidatable Debt Share (%)",
    },
    title=(
        "Distribution of Simulated "
        "Liquidation Exposure"
    ),
)

st.plotly_chart(
    mc_hist,
    width="stretch",
)

percentiles = (
    mc["liquidatable_debt_share"]
    .quantile(
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .reset_index()
)

percentiles.columns = [
    "percentile",
    "liquidatable_debt_share",
]

percentiles["percentile"] = (
    percentiles["percentile"] * 100
)

percentiles[
    "liquidatable_debt_share_pct"
] = (
    percentiles[
        "liquidatable_debt_share"
    ]
    * 100
)

st.subheader("Tail Risk")

tail_risk_table = percentiles[
    [
        "percentile",
        "liquidatable_debt_share_pct",
    ]
].rename(
    columns={
        "percentile": "Percentile",
        "liquidatable_debt_share_pct":
            "Liquidatable debt share (%)",
    }
)

st.dataframe(
    tail_risk_table,
    width="stretch",
)

st.subheader("Risk Parameter Sensitivity")

st.write(
    "Explore how counterfactual changes to liquidation thresholds "
    "alter liquidation exposure under a fixed market shock."
)

sensitivity_shock = st.slider(
    "Market shock for sensitivity test (%)",
    min_value=-50,
    max_value=-5,
    value=-20,
    step=5,
)

threshold_adjustments = [
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
]

sensitivity = run_threshold_sensitivity(
    positions,
    threshold_adjustments,
    market_shock=sensitivity_shock / 100,
)

sensitivity[
    "threshold_change_pp"
] = (
    sensitivity[
        "threshold_adjustment"
    ]
    * 100
)

sensitivity[
    "debt_exposure_pct"
] = (
    sensitivity[
        "liquidatable_debt_share"
    ]
    * 100
)

sensitivity_chart = px.line(
    sensitivity,
    x="threshold_change_pp",
    y="debt_exposure_pct",
    markers=True,
    labels={
        "threshold_change_pp":
            "Liquidation Threshold Change (pp)",
        "debt_exposure_pct":
            "Liquidatable Debt Share (%)",
    },
    title="Liquidation Threshold Sensitivity",
)

st.plotly_chart(
    sensitivity_chart,
    width="stretch",
)

st.caption(
    "Counterfactual analysis: positions were generated under baseline "
    "risk parameters and are re-evaluated under alternative liquidation "
    "thresholds. Results should not be interpreted as an optimal "
    "parameter recommendation."
)