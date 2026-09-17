import numpy as np
import pandas as pd
import pytest

from src.cascade_simulation import (
    run_cascade_aware_monte_carlo,
    run_market_depth_sensitivity,
    summarize_cascade_monte_carlo,
)
from src.data_generator import generate_protocol_positions


def _positions() -> pd.DataFrame:
    return generate_protocol_positions(
        n_positions=80,
        seed=17,
    )


def _market_depth() -> dict[str, float]:
    return {
        "ETH": 750_000.0,
        "BTC": 1_000_000.0,
        "SOL": 400_000.0,
    }


def test_cascade_aware_monte_carlo_is_reproducible():
    positions = _positions()

    first = run_cascade_aware_monte_carlo(
        positions=positions,
        market_depth_usd=_market_depth(),
        n_simulations=10,
        horizon_days=30,
        seed=11,
        max_rounds=8,
    )
    second = run_cascade_aware_monte_carlo(
        positions=positions,
        market_depth_usd=_market_depth(),
        n_simulations=10,
        horizon_days=30,
        seed=11,
        max_rounds=8,
    )

    pd.testing.assert_frame_equal(first, second)


def test_cascade_exposure_never_falls_below_first_order_exposure():
    results = run_cascade_aware_monte_carlo(
        positions=_positions(),
        market_depth_usd=_market_depth(),
        n_simulations=12,
        seed=31,
        max_rounds=8,
    )

    assert np.all(
        results["cascade_debt_share"]
        + 1e-12
        >= results["first_order_debt_share"]
    )
    assert np.all(results["amplification_debt_share"] >= 0)
    assert np.all(results["bad_debt_share"] >= 0)
    assert np.all(results["bad_debt_share"] <= 1)


def test_zero_price_impact_reduces_to_first_order_exposure():
    results = run_cascade_aware_monte_carlo(
        positions=_positions(),
        market_depth_usd=_market_depth(),
        n_simulations=10,
        seed=7,
        price_impact_factor=0.0,
        max_rounds=8,
    )

    assert np.allclose(
        results["cascade_debt_share"],
        results["first_order_debt_share"],
    )
    assert np.allclose(results["amplification_debt_share"], 0.0)
    assert (results["cascade_created_positions"] == 0).all()
    assert np.allclose(results["max_endogenous_price_decline"], 0.0)


def test_summary_reports_ordered_tail_metrics_and_bounded_probabilities():
    results = run_cascade_aware_monte_carlo(
        positions=_positions(),
        market_depth_usd=_market_depth(),
        n_simulations=15,
        seed=23,
        max_rounds=8,
    )
    summary = summarize_cascade_monte_carlo(results)

    assert summary["simulations"] == 15
    assert (
        summary["p90_cascade_debt_share"]
        <= summary["p95_cascade_debt_share"]
        <= summary["p99_cascade_debt_share"]
    )
    assert (
        summary["mean_cascade_debt_share"]
        + 1e-12
        >= summary["mean_first_order_debt_share"]
    )
    assert 0 <= summary["probability_of_cascade_amplification"] <= 1
    assert 0 <= summary["probability_cascade_exposure_gt_25"] <= 1
    assert 0 <= summary["probability_cascade_exposure_gt_50"] <= 1
    assert 0 <= summary["probability_cascade_exposure_gt_75"] <= 1


def test_shallower_liquidity_increases_endogenous_price_stress():
    sensitivity = run_market_depth_sensitivity(
        positions=_positions(),
        base_market_depth_usd=_market_depth(),
        depth_multipliers=[0.5, 1.0, 2.0],
        n_simulations=12,
        horizon_days=30,
        seed=29,
        max_rounds=8,
    )

    shallow = sensitivity.loc[
        sensitivity["depth_multiplier"] == 0.5
    ].iloc[0]
    deep = sensitivity.loc[
        sensitivity["depth_multiplier"] == 2.0
    ].iloc[0]

    assert (
        shallow["p95_endogenous_price_decline"]
        + 1e-12
        >= deep["p95_endogenous_price_decline"]
    )
    assert (
        shallow["mean_cascade_debt_share"]
        + 1e-12
        >= deep["mean_cascade_debt_share"]
    )


def test_invalid_market_depth_is_rejected():
    positions = _positions()

    with pytest.raises(ValueError, match="missing assets"):
        run_cascade_aware_monte_carlo(
            positions=positions,
            market_depth_usd={"ETH": 1_000_000.0},
            n_simulations=5,
        )

    with pytest.raises(ValueError, match="positive and finite"):
        run_cascade_aware_monte_carlo(
            positions=positions,
            market_depth_usd={
                "ETH": 1_000_000.0,
                "BTC": 0.0,
                "SOL": 1_000_000.0,
            },
            n_simulations=5,
        )
