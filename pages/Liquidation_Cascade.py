import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_generator import generate_protocol_positions
from src.liquidation_cascade import run_liquidation_cascade


st.set_page_config(
    page_title="RiskForge · Liquidation Cascade Lab",
    page_icon="🌊",
    layout="wide",
)

st.title("Liquidation Cascade Lab")

st.write(
    "Stress the protocol beyond first-order liquidation exposure. "
    "Liquidated collateral creates sell pressure, market depth converts "
    "that pressure into additional price impact, and the engine iterates "
    "until the cascade stabilizes or reaches the configured round limit."
)

st.caption(
    "This is a deterministic stress-testing model. It is not a forecast of "
    "realized execution or a recommendation for protocol parameters."
)

positions = generate_protocol_positions(
    n_positions=500,
    seed=42,
)

st.subheader("Initial market shock")

shock_col1, shock_col2, shock_col3 = st.columns(3)

eth_shock = shock_col1.slider(
    "ETH shock (%)",
    min_value=-70,
    max_value=10,
    value=-20,
    step=5,
)

btc_shock = shock_col2.slider(
    "BTC shock (%)",
    min_value=-70,
    max_value=10,
    value=-15,
    step=5,
)

sol_shock = shock_col3.slider(
    "SOL shock (%)",
    min_value=-80,
    max_value=10,
    value=-30,
    step=5,
)

st.subheader("Liquidity and liquidation assumptions")

left, middle, right = st.columns(3)

eth_depth_m = left.slider(
    "ETH market depth ($M)",
    min_value=1,
    max_value=200,
    value=40,
    step=1,
)

btc_depth_m = middle.slider(
    "BTC market depth ($M)",
    min_value=1,
    max_value=300,
    value=80,
    step=1,
)

sol_depth_m = right.slider(
    "SOL market depth ($M)",
    min_value=1,
    max_value=100,
    value=15,
    step=1,
)

control_col1, control_col2, control_col3, control_col4 = st.columns(4)

close_factor = control_col1.slider(
    "Close factor",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05,
)

liquidation_bonus_pct = control_col2.slider(
    "Liquidation bonus (%)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5,
)

price_impact_factor = control_col3.slider(
    "Price-impact strength",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
)

max_rounds = control_col4.slider(
    "Maximum rounds",
    min_value=1,
    max_value=30,
    value=12,
    step=1,
)

initial_shocks = {
    "ETH": eth_shock / 100,
    "BTC": btc_shock / 100,
    "SOL": sol_shock / 100,
}

market_depth_usd = {
    "ETH": eth_depth_m * 1_000_000,
    "BTC": btc_depth_m * 1_000_000,
    "SOL": sol_depth_m * 1_000_000,
}

result = run_liquidation_cascade(
    positions=positions,
    initial_shocks=initial_shocks,
    market_depth_usd=market_depth_usd,
    close_factor=close_factor,
    liquidation_bonus=liquidation_bonus_pct / 100,
    price_impact_factor=price_impact_factor,
    max_rounds=max_rounds,
)

summary = result.summary

st.subheader("Cascade outcome")

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

metric1.metric(
    "Initial liquidations",
    f"{summary['initial_liquidatable_positions']:,}",
)

metric2.metric(
    "Cascade-created liquidations",
    f"{summary['cascade_liquidated_positions']:,}",
)

metric3.metric(
    "Debt repaid",
    f"${summary['debt_repaid_usd']:,.0f}",
)

metric4.metric(
    "Bad debt",
    f"${summary['bad_debt_usd']:,.0f}",
)

metric5.metric(
    "Rounds executed",
    f"{summary['rounds_executed']:,}",
)

if summary["max_rounds_reached"]:
    st.warning(
        "The cascade still had liquidatable positions when the round limit "
        "was reached. Increase the maximum rounds to inspect convergence."
    )

asset_summary = result.asset_summary.copy()
asset_summary["endogenous_price_impact_pct"] *= 100

asset_display = asset_summary.rename(
    columns={
        "asset": "Asset",
        "base_price": "Base price",
        "shocked_price": "Price after initial shock",
        "final_price": "Final price",
        "endogenous_price_impact_pct": "Additional cascade impact (%)",
        "market_depth_usd": "Market depth ($)",
        "initial_debt_usd": "Initial debt ($)",
        "debt_repaid_usd": "Debt repaid ($)",
        "final_debt_usd": "Remaining debt ($)",
        "collateral_sold_usd": "Collateral sold ($)",
        "initial_liquidatable_positions": "Initial liquidations",
        "cascade_liquidated_positions": "Cascade-created liquidations",
        "remaining_liquidatable_positions": "Remaining liquidations",
        "bad_debt_usd": "Bad debt ($)",
    }
)

st.subheader("Asset-level cascade attribution")

st.dataframe(
    asset_display,
    width="stretch",
    hide_index=True,
)

history = result.round_history.copy()

if not history.empty:
    st.subheader("Endogenous price path")

    price_path = history[
        [
            "round",
            "asset",
            "ending_price",
        ]
    ].copy()

    round_zero = asset_summary[
        [
            "asset",
            "shocked_price",
        ]
    ].rename(
        columns={
            "shocked_price": "ending_price",
        }
    )
    round_zero["round"] = 0

    price_path = pd.concat(
        [
            round_zero[
                [
                    "round",
                    "asset",
                    "ending_price",
                ]
            ],
            price_path,
        ],
        ignore_index=True,
    )

    price_chart = px.line(
        price_path,
        x="round",
        y="ending_price",
        color="asset",
        markers=True,
        labels={
            "round": "Liquidation round",
            "ending_price": "Asset price",
            "asset": "Collateral asset",
        },
        title="Price impact generated by liquidation sell pressure",
    )

    st.plotly_chart(
        price_chart,
        width="stretch",
    )

    st.subheader("Liquidation flow by round")

    sales_chart = px.bar(
        history,
        x="round",
        y="collateral_sold_usd",
        color="asset",
        barmode="group",
        labels={
            "round": "Liquidation round",
            "collateral_sold_usd": "Collateral sold ($)",
            "asset": "Collateral asset",
        },
        title="Collateral liquidated into the market each round",
    )

    st.plotly_chart(
        sales_chart,
        width="stretch",
    )

    cascade_chart = px.bar(
        history,
        x="round",
        y="newly_liquidatable_positions",
        color="asset",
        barmode="stack",
        labels={
            "round": "Liquidation round",
            "newly_liquidatable_positions": "Newly liquidatable positions",
            "asset": "Collateral asset",
        },
        title="Positions pushed into liquidation by endogenous price impact",
    )

    st.plotly_chart(
        cascade_chart,
        width="stretch",
    )

st.subheader("Most vulnerable ending positions")

final_positions = result.final_positions.copy()
final_positions = final_positions.sort_values(
    [
        "health_factor",
        "bad_debt_usd",
    ],
    ascending=[True, False],
)

position_display = final_positions[
    [
        "position_id",
        "asset",
        "current_price",
        "debt_usd",
        "collateral_value",
        "health_factor",
        "liquidatable",
        "bad_debt_usd",
        "first_liquidation_round",
        "cascade_liquidated",
    ]
].head(25)

position_display = position_display.rename(
    columns={
        "position_id": "Position",
        "asset": "Asset",
        "current_price": "Final price",
        "debt_usd": "Remaining debt ($)",
        "collateral_value": "Remaining collateral ($)",
        "health_factor": "Health factor",
        "liquidatable": "Liquidatable",
        "bad_debt_usd": "Bad debt ($)",
        "first_liquidation_round": "First liquidation round",
        "cascade_liquidated": "Cascade-created",
    }
)

st.dataframe(
    position_display,
    width="stretch",
    hide_index=True,
)
