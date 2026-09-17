import pandas as pd
import pytest

from src.data_generator import generate_protocol_positions
from src.tail_risk_attribution import (
    attribute_market_scenario,
    run_tail_risk_attribution,
)


MARKET_DEPTH = {
    "ETH": 40_000_000.0,
    "BTC": 50_000_000.0,
    "SOL": 15_000_000.0,
}


def test_zero_shock_has_zero_leave_one_out_contribution():
    positions = generate_protocol_positions(n_positions=120, seed=7)

    full, attribution = attribute_market_scenario(
        positions=positions,
        shocks={"ETH": 0.0, "BTC": 0.0, "SOL": 0.0},
        market_depth_usd=MARKET_DEPTH,
    )

    assert full["cascade_debt_share"] == pytest.approx(0.0)
    assert full["bad_debt_share"] == pytest.approx(0.0)
    assert attribution["contribution_cascade_debt_share"].abs().max() == pytest.approx(0.0)
    assert attribution["contribution_bad_debt_share"].abs().max() == pytest.approx(0.0)


def test_isolated_eth_crash_has_positive_eth_contribution():
    positions = generate_protocol_positions(n_positions=300, seed=42)
    deep_liquidity = {
        "ETH": 1_000_000_000.0,
        "BTC": 1_000_000_000.0,
        "SOL": 1_000_000_000.0,
    }

    full, attribution = attribute_market_scenario(
        positions=positions,
        shocks={"ETH": -0.45, "BTC": 0.0, "SOL": 0.0},
        market_depth_usd=deep_liquidity,
        price_impact_factor=0.0,
    )

    eth = attribution.loc[attribution["asset"] == "ETH"].iloc[0]
    btc = attribution.loc[attribution["asset"] == "BTC"].iloc[0]
    sol = attribution.loc[attribution["asset"] == "SOL"].iloc[0]

    assert full["cascade_debt_share"] > 0
    assert eth["contribution_cascade_debt_share"] > 0
    assert btc["contribution_cascade_debt_share"] == pytest.approx(0.0)
    assert sol["contribution_cascade_debt_share"] == pytest.approx(0.0)


def test_market_scenario_attribution_is_deterministic():
    positions = generate_protocol_positions(n_positions=150, seed=11)
    shocks = {"ETH": -0.30, "BTC": -0.20, "SOL": -0.40}

    full_a, attribution_a = attribute_market_scenario(
        positions=positions,
        shocks=shocks,
        market_depth_usd=MARKET_DEPTH,
    )
    full_b, attribution_b = attribute_market_scenario(
        positions=positions,
        shocks=shocks,
        market_depth_usd=MARKET_DEPTH,
    )

    assert full_a == full_b
    pd.testing.assert_frame_equal(attribution_a, attribution_b)


def test_tail_risk_attribution_is_reproducible_and_complete():
    positions = generate_protocol_positions(n_positions=100, seed=21)

    result_a = run_tail_risk_attribution(
        positions=positions,
        market_depth_usd=MARKET_DEPTH,
        n_simulations=6,
        horizon_days=30,
        seed=19,
        tail_quantile=0.90,
    )
    result_b = run_tail_risk_attribution(
        positions=positions,
        market_depth_usd=MARKET_DEPTH,
        n_simulations=6,
        horizon_days=30,
        seed=19,
        tail_quantile=0.90,
    )

    pd.testing.assert_frame_equal(result_a.scenario_results, result_b.scenario_results)
    pd.testing.assert_frame_equal(result_a.attribution_results, result_b.attribution_results)
    pd.testing.assert_frame_equal(result_a.asset_summary, result_b.asset_summary)

    assert len(result_a.scenario_results) == 6
    assert len(result_a.attribution_results) == 18
    assert set(result_a.asset_summary["asset"]) == {"ETH", "BTC", "SOL"}
    assert result_a.tail_quantile == pytest.approx(0.90)


def test_summary_tail_contributions_match_tail_cohort_means():
    positions = generate_protocol_positions(n_positions=120, seed=5)
    result = run_tail_risk_attribution(
        positions=positions,
        market_depth_usd=MARKET_DEPTH,
        n_simulations=8,
        horizon_days=45,
        seed=31,
        tail_quantile=0.75,
    )

    cutoff = result.scenario_results["cascade_debt_share"].quantile(0.75)
    tail_ids = set(
        result.scenario_results.loc[
            result.scenario_results["cascade_debt_share"] >= cutoff,
            "simulation_id",
        ]
    )

    for _, summary in result.asset_summary.iterrows():
        rows = result.attribution_results.loc[
            (result.attribution_results["asset"] == summary["asset"])
            & (result.attribution_results["simulation_id"].isin(tail_ids))
        ]
        expected = rows["contribution_cascade_debt_share"].mean()
        assert summary["tail_mean_cascade_exposure_contribution"] == pytest.approx(
            expected
        )


def test_invalid_attribution_inputs_raise():
    positions = generate_protocol_positions(n_positions=50, seed=9)

    with pytest.raises(ValueError, match="tail_quantile"):
        run_tail_risk_attribution(
            positions=positions,
            market_depth_usd=MARKET_DEPTH,
            n_simulations=2,
            tail_quantile=1.0,
        )

    with pytest.raises(ValueError, match="market_depth_usd"):
        run_tail_risk_attribution(
            positions=positions,
            market_depth_usd={"ETH": 1_000_000.0},
            n_simulations=2,
        )
