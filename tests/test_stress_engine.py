import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_protocol_positions
from src.risk_engine import LendingPosition
from src.stress_engine import (
    apply_price_shock,
    run_asset_specific_stress_test,
    run_protocol_shock_ladder,
    run_protocol_stress_test,
)


def test_price_shock_reduces_collateral_price():
    position = LendingPosition(
        collateral_asset="ETH",
        collateral_amount=10,
        collateral_price=4000,
        debt_usd=25000,
        liquidation_threshold=0.80,
    )

    stressed = apply_price_shock(position, -0.20)

    assert stressed.collateral_price == pytest.approx(3200)
    assert stressed.health_factor == pytest.approx(1.024)


def test_zero_shock_preserves_protocol():
    positions = generate_protocol_positions(
        n_positions=100,
        seed=42,
    )

    stressed = run_protocol_stress_test(
        positions,
        shock_pct=0,
    )

    assert np.allclose(
    stressed["stressed_collateral_value"],
    positions["collateral_value"],
)

    assert stressed["liquidatable"].sum() == 0


def test_larger_crash_increases_liquidations():
    positions = generate_protocol_positions(
        n_positions=500,
        seed=42,
    )

    mild = run_protocol_stress_test(
        positions,
        shock_pct=-0.10,
    )

    severe = run_protocol_stress_test(
        positions,
        shock_pct=-0.40,
    )

    assert (
        severe["liquidatable"].sum()
        >= mild["liquidatable"].sum()
    )


def test_asset_specific_shock_only_changes_target_asset():
    positions = pd.DataFrame(
        {
            "position_id": [1, 2],
            "asset": ["ETH", "BTC"],
            "collateral_amount": [10.0, 1.0],
            "collateral_price": [4000.0, 65000.0],
            "collateral_value": [40000.0, 65000.0],
            "debt_usd": [20000.0, 30000.0],
            "liquidation_threshold": [0.80, 0.75],
            "ltv": [0.50, 30000 / 65000],
            "health_factor": [1.60, 1.625],
        }
    )

    stressed = run_asset_specific_stress_test(
        positions,
        {
            "ETH": -0.50,
            "BTC": 0.00,
        },
    )

    eth = stressed.loc[
        stressed["asset"] == "ETH"
    ].iloc[0]

    btc = stressed.loc[
        stressed["asset"] == "BTC"
    ].iloc[0]

    assert eth["stressed_price"] == pytest.approx(2000)
    assert btc["stressed_price"] == pytest.approx(65000)


def test_shock_ladder_exposure_is_monotonic():
    positions = generate_protocol_positions(
        n_positions=500,
        seed=42,
    )

    ladder = run_protocol_shock_ladder(
        positions,
        [0, -0.10, -0.20, -0.30, -0.40],
    )

    exposure = ladder[
        "liquidatable_debt_share"
    ].tolist()

    assert exposure == sorted(exposure)