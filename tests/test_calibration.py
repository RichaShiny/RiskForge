import numpy as np
import pandas as pd

from src.calibration import (
    calculate_log_returns,
    calibrate_market_parameters,
)


def test_log_returns_shape():
    prices = pd.DataFrame(
        {
            "ETH": [100, 110, 121],
            "BTC": [200, 220, 242],
            "SOL": [50, 55, 60.5],
        }
    )

    returns = calculate_log_returns(prices)

    assert returns.shape == (2, 3)


def test_log_returns_are_finite():
    prices = pd.DataFrame(
        {
            "ETH": [100, 110, 121],
            "BTC": [200, 220, 242],
            "SOL": [50, 55, 60.5],
        }
    )

    returns = calculate_log_returns(prices)

    assert np.isfinite(returns.to_numpy()).all()


def test_calibration_returns_expected_assets():
    prices = pd.DataFrame(
        {
            "ETH": [100, 105, 110, 108, 115],
            "BTC": [200, 205, 210, 208, 215],
            "SOL": [50, 53, 55, 52, 58],
        }
    )

    volatility, correlation = calibrate_market_parameters(
        prices
    )

    assert list(volatility.index) == [
        "ETH",
        "BTC",
        "SOL",
    ]

    assert list(correlation.columns) == [
        "ETH",
        "BTC",
        "SOL",
    ]


def test_correlation_matrix_is_symmetric():
    prices = pd.DataFrame(
        {
            "ETH": [100, 105, 110, 108, 115],
            "BTC": [200, 205, 210, 208, 215],
            "SOL": [50, 53, 55, 52, 58],
        }
    )

    _, correlation = calibrate_market_parameters(
        prices
    )

    assert np.allclose(
        correlation.to_numpy(),
        correlation.to_numpy().T,
    )


def test_correlation_diagonal_is_one():
    prices = pd.DataFrame(
        {
            "ETH": [100, 105, 110, 108, 115],
            "BTC": [200, 205, 210, 208, 215],
            "SOL": [50, 53, 55, 52, 58],
        }
    )

    _, correlation = calibrate_market_parameters(
        prices
    )

    assert np.allclose(
        np.diag(correlation),
        np.ones(3),
    )


def test_volatility_is_non_negative():
    prices = pd.DataFrame(
        {
            "ETH": [100, 105, 110, 108, 115],
            "BTC": [200, 205, 210, 208, 215],
            "SOL": [50, 53, 55, 52, 58],
        }
    )

    volatility, _ = calibrate_market_parameters(
        prices
    )

    assert (volatility >= 0).all()