import pytest

from src.risk_engine import LendingPosition


def test_lending_position_metrics():
    position = LendingPosition(
        collateral_asset="ETH",
        collateral_amount=10,
        collateral_price=4000,
        debt_usd=25000,
        liquidation_threshold=0.80,
    )

    assert position.collateral_value == 40000
    assert position.ltv == pytest.approx(0.625)
    assert position.health_factor == pytest.approx(1.28)
    assert position.liquidation_price == pytest.approx(3125)
    assert position.distance_to_liquidation == pytest.approx(0.21875)
    assert position.risk_status == "MODERATE RISK"