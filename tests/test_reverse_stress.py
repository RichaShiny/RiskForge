import pandas as pd
import pytest

from src.data_generator import generate_protocol_positions
from src.reverse_stress import (
    build_reverse_stress_frontier,
    evaluate_reverse_stress_point,
    find_critical_decline,
)


def _positions() -> pd.DataFrame:
    return generate_protocol_positions(
        n_positions=120,
        seed=42,
    )


def _market_depth(multiplier: float = 1.0) -> dict[str, float]:
    return {
        "ETH": 8_000_000.0 * multiplier,
        "BTC": 12_000_000.0 * multiplier,
        "SOL": 4_000_000.0 * multiplier,
    }


def test_zero_decline_preserves_healthy_protocol():
    point = evaluate_reverse_stress_point(
        positions=_positions(),
        decline=0.0,
        market_depth_usd=_market_depth(),
    )

    assert point.first_order_debt_share == pytest.approx(0.0)
    assert point.cascade_debt_share == pytest.approx(0.0)
    assert point.amplification_debt_share == pytest.approx(0.0)
    assert point.bad_debt_share == pytest.approx(0.0)


def test_solver_returns_breach_with_safe_lower_bracket():
    result = find_critical_decline(
        positions=_positions(),
        metric="cascade_debt_share",
        target=0.25,
        market_depth_usd=_market_depth(),
        tolerance=0.002,
        max_rounds=10,
    )

    assert result.found
    assert result.critical_decline is not None
    assert result.metric_value is not None
    assert result.metric_value >= 0.25
    assert result.lower_safe_decline is not None
    assert result.lower_safe_value is not None
    assert result.lower_safe_value < 0.25
    assert (
        result.critical_decline - result.lower_safe_decline
        <= result.tolerance + 1e-12
    )


def test_higher_exposure_target_requires_at_least_as_large_a_decline():
    low = find_critical_decline(
        positions=_positions(),
        metric="cascade_debt_share",
        target=0.25,
        market_depth_usd=_market_depth(),
        tolerance=0.002,
        max_rounds=10,
    )
    high = find_critical_decline(
        positions=_positions(),
        metric="cascade_debt_share",
        target=0.50,
        market_depth_usd=_market_depth(),
        tolerance=0.002,
        max_rounds=10,
    )

    assert low.found and high.found
    assert high.critical_decline is not None
    assert low.critical_decline is not None
    assert high.critical_decline + 1e-12 >= low.critical_decline


def test_zero_price_impact_aligns_cascade_and_first_order_thresholds():
    kwargs = {
        "positions": _positions(),
        "target": 0.25,
        "market_depth_usd": _market_depth(),
        "tolerance": 0.001,
        "price_impact_factor": 0.0,
        "max_rounds": 10,
    }

    first_order = find_critical_decline(
        metric="first_order_debt_share",
        **kwargs,
    )
    cascade = find_critical_decline(
        metric="cascade_debt_share",
        **kwargs,
    )

    assert first_order.found and cascade.found
    assert first_order.critical_decline is not None
    assert cascade.critical_decline is not None
    assert cascade.critical_decline == pytest.approx(
        first_order.critical_decline,
        abs=0.001,
    )


def test_shallower_market_depth_breaches_cascade_target_no_later():
    shallow = find_critical_decline(
        positions=_positions(),
        metric="cascade_debt_share",
        target=0.25,
        market_depth_usd=_market_depth(0.25),
        tolerance=0.002,
        max_rounds=10,
    )
    deep = find_critical_decline(
        positions=_positions(),
        metric="cascade_debt_share",
        target=0.25,
        market_depth_usd=_market_depth(4.0),
        tolerance=0.002,
        max_rounds=10,
    )

    assert shallow.found and deep.found
    assert shallow.critical_decline is not None
    assert deep.critical_decline is not None
    assert shallow.critical_decline <= deep.critical_decline + 1e-12


def test_frontier_is_sorted_and_thresholds_are_monotonic():
    frontier = build_reverse_stress_frontier(
        positions=_positions(),
        metric="cascade_debt_share",
        targets=[0.50, 0.25, 0.75, 0.25],
        market_depth_usd=_market_depth(),
        tolerance=0.003,
        max_rounds=10,
    )

    assert frontier["target"].tolist() == [0.25, 0.50, 0.75]
    assert frontier["found"].all()

    critical = frontier["critical_decline"].tolist()
    assert critical == sorted(critical)


def test_asset_stress_weights_change_realized_shock_vector():
    point = evaluate_reverse_stress_point(
        positions=_positions(),
        decline=0.20,
        market_depth_usd=_market_depth(),
        asset_stress_weights={
            "ETH": 1.0,
            "BTC": 0.5,
            "SOL": 1.5,
        },
    )

    assert point.shocks["ETH"] == pytest.approx(-0.20)
    assert point.shocks["BTC"] == pytest.approx(-0.10)
    assert point.shocks["SOL"] == pytest.approx(-0.30)


def test_invalid_reverse_stress_inputs_are_rejected():
    positions = _positions()

    with pytest.raises(ValueError, match="metric must be one of"):
        find_critical_decline(
            positions=positions,
            metric="unknown",  # type: ignore[arg-type]
            target=0.25,
            market_depth_usd=_market_depth(),
        )

    with pytest.raises(ValueError, match="target must be in"):
        find_critical_decline(
            positions=positions,
            metric="cascade_debt_share",
            target=1.5,
            market_depth_usd=_market_depth(),
        )

    with pytest.raises(ValueError, match="missing assets"):
        evaluate_reverse_stress_point(
            positions=positions,
            decline=0.20,
            market_depth_usd={"ETH": 1_000_000.0},
        )

    with pytest.raises(ValueError, match="stress weight"):
        evaluate_reverse_stress_point(
            positions=positions,
            decline=0.20,
            market_depth_usd=_market_depth(),
            asset_stress_weights={
                "ETH": 1.0,
                "BTC": -1.0,
                "SOL": 1.0,
            },
        )
