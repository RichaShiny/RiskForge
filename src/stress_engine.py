from dataclasses import replace

import pandas as pd

from src.risk_engine import LendingPosition


def apply_price_shock(
    position: LendingPosition,
    shock_pct: float,
) -> LendingPosition:
    """
    Apply a percentage price shock to a lending position.

    Example:
        shock_pct = -0.20 means the collateral price falls by 20%.
    """
    shocked_price = position.collateral_price * (1 + shock_pct)

    if shocked_price < 0:
        raise ValueError("Price shock cannot result in a negative asset price.")

    return replace(
        position,
        collateral_price=shocked_price,
    )


def run_stress_test(
    position: LendingPosition,
    shocks: list[float],
) -> list[dict]:
    """
    Evaluate a lending position across multiple market shocks.
    """
    results = []

    for shock in shocks:
        stressed_position = apply_price_shock(position, shock)

        results.append(
            {
                "shock_pct": shock,
                "collateral_price": stressed_position.collateral_price,
                "collateral_value": stressed_position.collateral_value,
                "ltv": stressed_position.ltv,
                "health_factor": stressed_position.health_factor,
                "risk_status": stressed_position.risk_status,
                "liquidatable": stressed_position.health_factor < 1,
            }
        )

    return results


def run_protocol_stress_test(
    positions: pd.DataFrame,
    shock_pct: float,
) -> pd.DataFrame:
    """
    Apply the same market shock to every position
    and recalculate protocol-level risk metrics.
    """
    stressed = positions.copy()

    stressed["stressed_price"] = (
        stressed["collateral_price"] * (1 + shock_pct)
    )

    stressed["stressed_collateral_value"] = (
        stressed["collateral_amount"]
        * stressed["stressed_price"]
    )

    stressed["stressed_ltv"] = (
        stressed["debt_usd"]
        / stressed["stressed_collateral_value"]
    )

    stressed["stressed_health_factor"] = (
        stressed["stressed_collateral_value"]
        * stressed["liquidation_threshold"]
        / stressed["debt_usd"]
    )

    stressed["liquidatable"] = (
        stressed["stressed_health_factor"] < 1
    )

    return stressed

def run_protocol_shock_ladder(
    positions: pd.DataFrame,
    shocks: list[float],
) -> pd.DataFrame:
    """
    Run protocol-wide stress tests across multiple market shocks
    and summarize liquidation exposure at each shock level.
    """
    results = []

    total_debt = positions["debt_usd"].sum()

    for shock in shocks:
        stressed = run_protocol_stress_test(
            positions=positions,
            shock_pct=shock,
        )

        liquidatable = stressed["liquidatable"]

        liquidatable_positions = int(liquidatable.sum())

        liquidatable_debt = stressed.loc[
            liquidatable,
            "debt_usd",
        ].sum()

        results.append(
            {
                "shock_pct": shock,
                "liquidatable_positions": liquidatable_positions,
                "liquidation_rate": liquidatable.mean(),
                "liquidatable_debt": liquidatable_debt,
                "liquidatable_debt_share": (
                    liquidatable_debt / total_debt
                ),
            }
        )

    return pd.DataFrame(results)

def run_asset_specific_stress_test(
    positions: pd.DataFrame,
    asset_shocks: dict[str, float],
) -> pd.DataFrame:
    """
    Apply different price shocks by collateral asset.

    Example:
        {
            "ETH": -0.25,
            "BTC": -0.15,
            "SOL": -0.40,
        }
    """
    stressed = positions.copy()

    stressed["shock_pct"] = stressed["asset"].map(asset_shocks).fillna(0.0)

    stressed["stressed_price"] = (
        stressed["collateral_price"]
        * (1 + stressed["shock_pct"])
    )

    stressed["stressed_collateral_value"] = (
        stressed["collateral_amount"]
        * stressed["stressed_price"]
    )

    stressed["stressed_ltv"] = (
        stressed["debt_usd"]
        / stressed["stressed_collateral_value"]
    )

    stressed["stressed_health_factor"] = (
        stressed["stressed_collateral_value"]
        * stressed["liquidation_threshold"]
        / stressed["debt_usd"]
    )

    stressed["liquidatable"] = (
        stressed["stressed_health_factor"] < 1
    )

    return stressed