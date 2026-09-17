from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.liquidation_cascade import run_liquidation_cascade
from src.simulation import simulate_market_returns
from src.stress_engine import run_asset_specific_stress_test


EXPOSURE_THRESHOLDS = (0.25, 0.50, 0.75)


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_market_depth(
    positions: pd.DataFrame,
    market_depth_usd: dict[str, float],
) -> None:
    assets = set(positions["asset"].astype(str))
    missing = sorted(assets.difference(market_depth_usd))
    if missing:
        raise ValueError(
            "market_depth_usd is missing assets: " + ", ".join(missing)
        )

    for asset in assets:
        depth = market_depth_usd[asset]
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError(
                f"market depth for {asset} must be positive and finite"
            )


def run_cascade_aware_monte_carlo(
    positions: pd.DataFrame,
    market_depth_usd: dict[str, float],
    n_simulations: int = 250,
    horizon_days: int = 30,
    seed: int = 42,
    distribution: str = "student_t",
    degrees_of_freedom: int = 5,
    annual_volatility: dict[str, float] | None = None,
    correlation_matrix: np.ndarray | None = None,
    close_factor: float = 0.50,
    liquidation_bonus: float = 0.05,
    price_impact_factor: float = 1.0,
    max_rounds: int = 20,
    min_liquidation_usd: float = 1.0,
) -> pd.DataFrame:
    """Run paired first-order and endogenous-cascade Monte Carlo stress tests.

    Every simulated market-return draw is evaluated twice:

    1. A first-order stress test that marks positions liquidatable immediately
       after the exogenous market move.
    2. The liquidation-cascade engine, which allows liquidations to create
       collateral sell pressure, endogenous price impact, and second-order
       liquidations.

    The paired design isolates amplification caused by the modeled liquidation
    feedback loop. It does not treat the price-impact model as a prediction of
    realized market execution.
    """
    if positions.empty:
        raise ValueError("positions must contain at least one lending position")

    _validate_positive_integer(n_simulations, "n_simulations")
    _validate_positive_integer(horizon_days, "horizon_days")
    _validate_positive_integer(max_rounds, "max_rounds")
    _validate_market_depth(positions, market_depth_usd)

    total_debt = float(positions["debt_usd"].sum())
    if total_debt <= 0:
        raise ValueError("positions must contain positive protocol debt")

    market_returns = simulate_market_returns(
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        seed=seed,
        distribution=distribution,
        degrees_of_freedom=degrees_of_freedom,
        annual_volatility=annual_volatility,
        correlation_matrix=correlation_matrix,
    )

    results: list[dict[str, float | int]] = []

    for simulation_id, row in market_returns.iterrows():
        shocks = {
            "ETH": max(float(row["ETH"]), -0.99),
            "BTC": max(float(row["BTC"]), -0.99),
            "SOL": max(float(row["SOL"]), -0.99),
        }

        first_order = run_asset_specific_stress_test(positions, shocks)
        first_order_mask = first_order["liquidatable"]
        first_order_debt = float(
            first_order.loc[first_order_mask, "debt_usd"].sum()
        )
        first_order_positions = int(first_order_mask.sum())

        cascade = run_liquidation_cascade(
            positions=positions,
            initial_shocks=shocks,
            market_depth_usd=market_depth_usd,
            close_factor=close_factor,
            liquidation_bonus=liquidation_bonus,
            price_impact_factor=price_impact_factor,
            max_rounds=max_rounds,
            min_liquidation_usd=min_liquidation_usd,
        )

        final_positions = cascade.final_positions
        ever_liquidatable = final_positions["ever_liquidatable"].astype(bool)
        cascade_exposed_debt = float(
            final_positions.loc[
                ever_liquidatable,
                "initial_debt_usd",
            ].sum()
        )

        cascade_debt_share = cascade_exposed_debt / total_debt
        first_order_debt_share = first_order_debt / total_debt
        amplification_debt_share = max(
            0.0,
            cascade_debt_share - first_order_debt_share,
        )
        relative_amplification = (
            max(0.0, cascade_exposed_debt - first_order_debt)
            / first_order_debt
            if first_order_debt > 0
            else 0.0
        )

        bad_debt_usd = float(cascade.summary["bad_debt_usd"])
        bad_debt_share = bad_debt_usd / total_debt

        if cascade.asset_summary.empty:
            max_endogenous_price_decline = 0.0
        else:
            min_impact = float(
                cascade.asset_summary["endogenous_price_impact_pct"].min()
            )
            max_endogenous_price_decline = max(0.0, -min_impact)

        results.append(
            {
                "simulation_id": int(simulation_id),
                "eth_return": shocks["ETH"],
                "btc_return": shocks["BTC"],
                "sol_return": shocks["SOL"],
                "first_order_liquidatable_positions": first_order_positions,
                "first_order_liquidatable_debt_usd": first_order_debt,
                "first_order_debt_share": first_order_debt_share,
                "cascade_exposed_positions": int(ever_liquidatable.sum()),
                "cascade_created_positions": int(
                    cascade.summary["cascade_liquidated_positions"]
                ),
                "cascade_exposed_debt_usd": cascade_exposed_debt,
                "cascade_debt_share": cascade_debt_share,
                "amplification_debt_share": amplification_debt_share,
                "relative_amplification": relative_amplification,
                "debt_repaid_usd": float(cascade.summary["debt_repaid_usd"]),
                "bad_debt_usd": bad_debt_usd,
                "bad_debt_share": bad_debt_share,
                "rounds_executed": int(cascade.summary["rounds_executed"]),
                "max_endogenous_price_decline": max_endogenous_price_decline,
            }
        )

    return pd.DataFrame(results)


def summarize_cascade_monte_carlo(
    results: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize first-order risk, cascade risk, and tail amplification."""
    if results.empty:
        raise ValueError("results must contain at least one simulation")

    required_columns = {
        "first_order_debt_share",
        "cascade_debt_share",
        "amplification_debt_share",
        "relative_amplification",
        "cascade_created_positions",
        "bad_debt_share",
        "max_endogenous_price_decline",
    }
    missing = required_columns.difference(results.columns)
    if missing:
        raise ValueError(
            "results is missing required columns: "
            + ", ".join(sorted(missing))
        )

    first_order = results["first_order_debt_share"]
    cascade = results["cascade_debt_share"]
    amplification = results["amplification_debt_share"]
    bad_debt = results["bad_debt_share"]

    summary: dict[str, float | int] = {
        "simulations": int(len(results)),
        "mean_first_order_debt_share": float(first_order.mean()),
        "mean_cascade_debt_share": float(cascade.mean()),
        "mean_amplification_debt_share": float(amplification.mean()),
        "median_cascade_debt_share": float(cascade.median()),
        "p90_cascade_debt_share": float(cascade.quantile(0.90)),
        "p95_cascade_debt_share": float(cascade.quantile(0.95)),
        "p99_cascade_debt_share": float(cascade.quantile(0.99)),
        "p95_first_order_debt_share": float(first_order.quantile(0.95)),
        "p99_first_order_debt_share": float(first_order.quantile(0.99)),
        "p95_tail_amplification": float(
            cascade.quantile(0.95) - first_order.quantile(0.95)
        ),
        "p99_tail_amplification": float(
            cascade.quantile(0.99) - first_order.quantile(0.99)
        ),
        "probability_of_cascade_amplification": float((amplification > 0).mean()),
        "mean_relative_amplification": float(
            results["relative_amplification"].mean()
        ),
        "mean_cascade_created_positions": float(
            results["cascade_created_positions"].mean()
        ),
        "mean_bad_debt_share": float(bad_debt.mean()),
        "p95_bad_debt_share": float(bad_debt.quantile(0.95)),
        "p99_bad_debt_share": float(bad_debt.quantile(0.99)),
        "p95_endogenous_price_decline": float(
            results["max_endogenous_price_decline"].quantile(0.95)
        ),
    }

    for threshold in EXPOSURE_THRESHOLDS:
        label = int(threshold * 100)
        summary[f"probability_first_order_exposure_gt_{label}"] = float(
            (first_order > threshold).mean()
        )
        summary[f"probability_cascade_exposure_gt_{label}"] = float(
            (cascade > threshold).mean()
        )

    return summary


def run_market_depth_sensitivity(
    positions: pd.DataFrame,
    base_market_depth_usd: dict[str, float],
    depth_multipliers: list[float],
    n_simulations: int = 100,
    horizon_days: int = 30,
    seed: int = 42,
    distribution: str = "student_t",
    **cascade_kwargs,
) -> pd.DataFrame:
    """Measure cascade tail risk across market-liquidity assumptions.

    Each depth multiplier uses the same seed and simulation settings, so the
    exogenous market draws remain paired across liquidity assumptions.
    """
    if not depth_multipliers:
        raise ValueError("depth_multipliers must contain at least one value")

    rows: list[dict[str, float]] = []

    for multiplier in depth_multipliers:
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("depth multipliers must be positive and finite")

        market_depth = {
            asset: depth * multiplier
            for asset, depth in base_market_depth_usd.items()
        }
        results = run_cascade_aware_monte_carlo(
            positions=positions,
            market_depth_usd=market_depth,
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            seed=seed,
            distribution=distribution,
            **cascade_kwargs,
        )
        summary = summarize_cascade_monte_carlo(results)

        rows.append(
            {
                "depth_multiplier": float(multiplier),
                "mean_cascade_debt_share": float(
                    summary["mean_cascade_debt_share"]
                ),
                "mean_amplification_debt_share": float(
                    summary["mean_amplification_debt_share"]
                ),
                "p95_cascade_debt_share": float(
                    summary["p95_cascade_debt_share"]
                ),
                "p99_cascade_debt_share": float(
                    summary["p99_cascade_debt_share"]
                ),
                "p95_tail_amplification": float(
                    summary["p95_tail_amplification"]
                ),
                "probability_of_cascade_amplification": float(
                    summary["probability_of_cascade_amplification"]
                ),
                "p95_bad_debt_share": float(summary["p95_bad_debt_share"]),
                "p95_endogenous_price_decline": float(
                    summary["p95_endogenous_price_decline"]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("depth_multiplier").reset_index(drop=True)
