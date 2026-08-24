import numpy as np

from src.data_generator import generate_protocol_positions
from src.simulation import (
    run_monte_carlo_simulation,
    simulate_market_returns,
)


def test_market_simulation_shape():
    returns = simulate_market_returns(
        n_simulations=250,
        horizon_days=30,
        seed=42,
    )

    assert returns.shape == (250, 3)
    assert list(returns.columns) == ["ETH", "BTC", "SOL"]


def test_market_simulation_is_reproducible():
    first = simulate_market_returns(
        n_simulations=100,
        horizon_days=30,
        seed=42,
    )

    second = simulate_market_returns(
        n_simulations=100,
        horizon_days=30,
        seed=42,
    )

    assert np.allclose(first, second)


def test_longer_horizon_has_more_dispersion():
    short = simulate_market_returns(
        n_simulations=5000,
        horizon_days=7,
        seed=42,
    )

    long = simulate_market_returns(
        n_simulations=5000,
        horizon_days=90,
        seed=42,
    )

    assert long["ETH"].std() > short["ETH"].std()
    assert long["BTC"].std() > short["BTC"].std()
    assert long["SOL"].std() > short["SOL"].std()


def test_monte_carlo_exposure_is_bounded():
    positions = generate_protocol_positions(
        n_positions=100,
        seed=42,
    )

    results = run_monte_carlo_simulation(
        positions,
        n_simulations=100,
        horizon_days=30,
        seed=42,
    )

    exposure = results["liquidatable_debt_share"]

    assert (exposure >= 0).all()
    assert (exposure <= 1).all()


def test_monte_carlo_output_size():
    positions = generate_protocol_positions(
        n_positions=100,
        seed=42,
    )

    results = run_monte_carlo_simulation(
        positions,
        n_simulations=125,
        horizon_days=30,
        seed=42,
    )

    assert len(results) == 125