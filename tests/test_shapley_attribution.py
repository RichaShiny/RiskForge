import pandas as pd
import pytest

from src.data_generator import generate_protocol_positions
from src.shapley_attribution import (
    ATTRIBUTION_METRICS,
    run_shapley_tail_risk_attribution,
    shapley_market_scenario,
)


MARKET_DEPTH = {
    "ETH": 40_000_000.0,
    "BTC": 50_000_000.0,
    "SOL": 15_000_000.0,
}


def test_shapley_efficiency_reconciles_every_metric():
    positions = generate_protocol_positions(n_positions=180, seed=42)
    result = shapley_market_scenario(
        positions=positions,
        shocks={"ETH": -0.35, "BTC": -0.25, "SOL": -0.45},
        market_depth_usd=MARKET_DEPTH,
    )

    assert len(result.coalition_values) == 8

    for metric in ATTRIBUTION_METRICS:
        allocated = result.shapley_values[f"shapley_{metric}"].sum()
        expected = (
            float(result.full_metrics[metric])
            - float(result.baseline_metrics[metric])
        )
        assert allocated == pytest.approx(expected, abs=1e-10)
        assert result.shapley_values[
            f"efficiency_residual_{metric}"
        ].abs().max() <= 1e-10


def test_zero_shock_has_zero_shapley_contribution():
    positions = generate_protocol_positions(n_positions=100, seed=7)
    result = shapley_market_scenario(
        positions=positions,
        shocks={"ETH": 0.0, "BTC": 0.0, "SOL": 0.0},
        market_depth_usd=MARKET_DEPTH,
    )

    assert result.full_metrics == result.baseline_metrics
    for metric in ATTRIBUTION_METRICS:
        assert result.shapley_values[f"shapley_{metric}"].abs().max() == pytest.approx(0.0)


def test_isolated_eth_shock_assigns_risk_to_eth_only():
    positions = generate_protocol_positions(n_positions=300, seed=42)
    deep_liquidity = {
        "ETH": 1_000_000_000.0,
        "BTC": 1_000_000_000.0,
        "SOL": 1_000_000_000.0,
    }
    result = shapley_market_scenario(
        positions=positions,
        shocks={"ETH": -0.45, "BTC": 0.0, "SOL": 0.0},
        market_depth_usd=deep_liquidity,
        price_impact_factor=0.0,
    )

    values = result.shapley_values.set_index("asset")
    assert values.loc["ETH", "shapley_cascade_debt_share"] > 0
    assert values.loc["BTC", "shapley_cascade_debt_share"] == pytest.approx(0.0)
    assert values.loc["SOL", "shapley_cascade_debt_share"] == pytest.approx(0.0)


def test_shapley_market_scenario_is_deterministic():
    positions = generate_protocol_positions(n_positions=120, seed=11)
    kwargs = dict(
        positions=positions,
        shocks={"ETH": -0.30, "BTC": -0.20, "SOL": -0.40},
        market_depth_usd=MARKET_DEPTH,
    )

    a = shapley_market_scenario(**kwargs)
    b = shapley_market_scenario(**kwargs)

    assert a.full_metrics == b.full_metrics
    assert a.baseline_metrics == b.baseline_metrics
    pd.testing.assert_frame_equal(a.coalition_values, b.coalition_values)
    pd.testing.assert_frame_equal(a.shapley_values, b.shapley_values)


def test_stochastic_shapley_is_reproducible_and_efficient():
    positions = generate_protocol_positions(n_positions=80, seed=21)
    kwargs = dict(
        positions=positions,
        market_depth_usd=MARKET_DEPTH,
        n_simulations=4,
        horizon_days=30,
        seed=19,
        tail_quantile=0.75,
    )

    a = run_shapley_tail_risk_attribution(**kwargs)
    b = run_shapley_tail_risk_attribution(**kwargs)

    pd.testing.assert_frame_equal(a.scenario_results, b.scenario_results)
    pd.testing.assert_frame_equal(a.shapley_results, b.shapley_results)
    pd.testing.assert_frame_equal(a.asset_summary, b.asset_summary)

    assert len(a.scenario_results) == 4
    assert len(a.shapley_results) == 12
    assert set(a.asset_summary["asset"]) == {"ETH", "BTC", "SOL"}

    for metric in ATTRIBUTION_METRICS:
        assert a.scenario_results[
            f"shapley_efficiency_error_{metric}"
        ].abs().max() <= 1e-10


def test_tail_summary_matches_selected_cohort():
    positions = generate_protocol_positions(n_positions=90, seed=5)
    result = run_shapley_tail_risk_attribution(
        positions=positions,
        market_depth_usd=MARKET_DEPTH,
        n_simulations=6,
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
        rows = result.shapley_results.loc[
            (result.shapley_results["asset"] == summary["asset"])
            & (result.shapley_results["simulation_id"].isin(tail_ids))
        ]
        expected = rows["shapley_cascade_debt_share"].mean()
        assert summary["tail_mean_shapley_cascade_exposure"] == pytest.approx(expected)


def test_invalid_shapley_inputs_raise():
    positions = generate_protocol_positions(n_positions=50, seed=9)

    with pytest.raises(ValueError, match="shock for ETH"):
        shapley_market_scenario(
            positions=positions,
            shocks={"ETH": -1.0},
            market_depth_usd=MARKET_DEPTH,
        )

    with pytest.raises(ValueError, match="n_simulations"):
        run_shapley_tail_risk_attribution(
            positions=positions,
            market_depth_usd=MARKET_DEPTH,
            n_simulations=0,
        )
