import numpy as np
import pandas as pd

from src.stress_engine import run_asset_specific_stress_test


ASSETS = ["ETH", "BTC", "SOL"]

ANNUAL_VOLATILITY = {
    "ETH": 0.75,
    "BTC": 0.60,
    "SOL": 1.00,
}

CORRELATION_MATRIX = np.array(
    [
        [1.00, 0.75, 0.65],
        [0.75, 1.00, 0.55],
        [0.65, 0.55, 1.00],
    ]
)


def simulate_market_returns(
    n_simulations: int = 1000,
    horizon_days: int = 30,
    seed: int = 42,
    distribution: str = "student_t",
    degrees_of_freedom: int = 5,
    annual_volatility: dict[str, float] | None = None,
    correlation_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Simulate correlated crypto returns.

    Supported distributions:
        - "normal"
        - "student_t"
    """
    if annual_volatility is None:
        annual_volatility = ANNUAL_VOLATILITY

    if correlation_matrix is None:
        correlation_matrix = CORRELATION_MATRIX

    rng = np.random.default_rng(seed)

    time_fraction = horizon_days / 365

    vol_vector = np.array(
        [
            annual_volatility[asset]
            for asset in ASSETS
        ]
    )

    horizon_vol = (
        vol_vector * np.sqrt(time_fraction)
    )

    covariance_matrix = (
        np.outer(
            horizon_vol,
            horizon_vol,
        )
        * correlation_matrix
    )

    normal_draws = rng.multivariate_normal(
        mean=np.zeros(len(ASSETS)),
        cov=covariance_matrix,
        size=n_simulations,
    )

    if distribution == "normal":
        simulated_returns = normal_draws

    elif distribution == "student_t":
        chi_square_draws = rng.chisquare(
            degrees_of_freedom,
            size=n_simulations,
        )

        scaling = np.sqrt(
            (degrees_of_freedom - 2)
            / chi_square_draws
        )

        simulated_returns = (
            normal_draws
            * scaling[:, None]
        )

    else:
        raise ValueError(
            "distribution must be 'normal' or 'student_t'"
        )

    simulated_returns = np.clip(
        simulated_returns,
        -0.99,
        None,
    )

    return pd.DataFrame(
        simulated_returns,
        columns=ASSETS,
    )


def run_monte_carlo_simulation(
    positions: pd.DataFrame,
    n_simulations: int = 1000,
    horizon_days: int = 30,
    seed: int = 42,
    distribution: str = "student_t",
    annual_volatility: dict[str, float] | None = None,
    correlation_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Simulate market outcomes and measure liquidation exposure
    for the protocol under each scenario.
    """
    market_returns = simulate_market_returns(
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        seed=seed,
        distribution=distribution,
        annual_volatility=annual_volatility,
        correlation_matrix=correlation_matrix,
    )

    total_debt = positions["debt_usd"].sum()

    results = []

    for simulation_id, row in market_returns.iterrows():
        shocks = {
            asset: max(row[asset], -0.99)
            for asset in ASSETS
        }

        stressed = run_asset_specific_stress_test(
            positions,
            shocks,
        )

        liquidatable = stressed[
            stressed["liquidatable"]
        ]

        liquidatable_debt = (
            liquidatable["debt_usd"].sum()
        )

        results.append(
            {
                "simulation_id": simulation_id,
                "eth_return": shocks["ETH"],
                "btc_return": shocks["BTC"],
                "sol_return": shocks["SOL"],
                "liquidatable_positions": len(
                    liquidatable
                ),
                "liquidatable_debt": (
                    liquidatable_debt
                ),
                "liquidatable_debt_share": (
                    liquidatable_debt
                    / total_debt
                ),
            }
        )

    return pd.DataFrame(results)


def run_threshold_sensitivity(
    positions: pd.DataFrame,
    threshold_adjustments: list[float],
    market_shock: float = -0.20,
) -> pd.DataFrame:
    """
    Measure how changes to liquidation thresholds affect
    protocol liquidation exposure under a fixed market shock.
    """
    results = []

    total_debt = positions["debt_usd"].sum()

    for adjustment in threshold_adjustments:
        adjusted_positions = positions.copy()

        adjusted_positions[
            "liquidation_threshold"
        ] = (
            adjusted_positions[
                "liquidation_threshold"
            ]
            + adjustment
        ).clip(
            lower=0.10,
            upper=0.95,
        )

        stressed = run_asset_specific_stress_test(
            adjusted_positions,
            {
                "ETH": market_shock,
                "BTC": market_shock,
                "SOL": market_shock,
            },
        )

        liquidatable = stressed[
            stressed["liquidatable"]
        ]

        liquidatable_debt = (
            liquidatable["debt_usd"].sum()
        )

        results.append(
            {
                "threshold_adjustment": (
                    adjustment
                ),
                "liquidatable_positions": len(
                    liquidatable
                ),
                "liquidation_rate": (
                    len(liquidatable)
                    / len(positions)
                ),
                "liquidatable_debt": (
                    liquidatable_debt
                ),
                "liquidatable_debt_share": (
                    liquidatable_debt
                    / total_debt
                ),
            }
        )

    return pd.DataFrame(results)