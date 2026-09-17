from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


REQUIRED_COLUMNS = {
    "position_id",
    "asset",
    "collateral_amount",
    "collateral_price",
    "debt_usd",
    "liquidation_threshold",
}


@dataclass
class LiquidationCascadeResult:
    """Outputs from an endogenous liquidation-cascade simulation."""

    final_positions: pd.DataFrame
    round_history: pd.DataFrame
    asset_summary: pd.DataFrame
    summary: dict[str, float | int | bool]


def _validate_inputs(
    positions: pd.DataFrame,
    initial_shocks: dict[str, float],
    market_depth_usd: dict[str, float],
    close_factor: float,
    liquidation_bonus: float,
    price_impact_factor: float,
    max_rounds: int,
    min_liquidation_usd: float,
) -> None:
    missing = REQUIRED_COLUMNS.difference(positions.columns)
    if missing:
        raise ValueError(
            "positions is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if positions.empty:
        raise ValueError("positions must contain at least one lending position")

    if not 0 < close_factor <= 1:
        raise ValueError("close_factor must be in the interval (0, 1]")

    if not math.isfinite(liquidation_bonus) or liquidation_bonus < 0:
        raise ValueError("liquidation_bonus must be finite and non-negative")

    if not math.isfinite(price_impact_factor) or price_impact_factor < 0:
        raise ValueError("price_impact_factor must be finite and non-negative")

    if not isinstance(max_rounds, int) or max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer")

    if not math.isfinite(min_liquidation_usd) or min_liquidation_usd < 0:
        raise ValueError("min_liquidation_usd must be finite and non-negative")

    assets = set(positions["asset"].astype(str))

    for asset in assets:
        asset_prices = positions.loc[
            positions["asset"].astype(str) == asset,
            "collateral_price",
        ]
        if asset_prices.nunique(dropna=False) != 1:
            raise ValueError(
                f"all positions for {asset} must use the same collateral_price"
            )

        depth = market_depth_usd.get(asset)
        if depth is None or not math.isfinite(depth) or depth <= 0:
            raise ValueError(
                f"market_depth_usd must contain a positive finite value for {asset}"
            )

    for asset, shock in initial_shocks.items():
        if not math.isfinite(shock) or shock <= -1:
            raise ValueError(
                f"initial shock for {asset} must be finite and greater than -1"
            )

    numeric_checks = {
        "collateral_amount": (0, None),
        "collateral_price": (0, None),
        "debt_usd": (0, None),
        "liquidation_threshold": (0, 1),
    }
    for column, (lower, upper) in numeric_checks.items():
        values = pd.to_numeric(positions[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} must contain only finite numeric values")
        if column == "collateral_price":
            invalid_lower = values <= lower
        elif column == "liquidation_threshold":
            invalid_lower = values <= lower
        else:
            invalid_lower = values < lower
        if invalid_lower.any():
            raise ValueError(f"{column} contains values below its valid range")
        if upper is not None and (values > upper).any():
            raise ValueError(f"{column} contains values above its valid range")


def _recalculate_state(
    positions: pd.DataFrame,
    prices: dict[str, float],
) -> None:
    positions["current_price"] = positions["asset"].map(prices).astype(float)
    positions["collateral_value"] = (
        positions["collateral_amount"] * positions["current_price"]
    )

    positions["ltv"] = 0.0
    positive_collateral = positions["collateral_value"] > 0
    positions.loc[positive_collateral, "ltv"] = (
        positions.loc[positive_collateral, "debt_usd"]
        / positions.loc[positive_collateral, "collateral_value"]
    )

    positions["health_factor"] = float("inf")
    positive_debt = positions["debt_usd"] > 0
    positions.loc[positive_debt, "health_factor"] = (
        positions.loc[positive_debt, "collateral_value"]
        * positions.loc[positive_debt, "liquidation_threshold"]
        / positions.loc[positive_debt, "debt_usd"]
    )

    positions["liquidatable"] = (
        positive_debt & (positions["health_factor"] < 1.0)
    )
    positions["bad_debt_usd"] = (
        positions["debt_usd"] - positions["collateral_value"]
    ).clip(lower=0.0)


def _asset_prices(positions: pd.DataFrame) -> dict[str, float]:
    return {
        str(asset): float(group["collateral_price"].iloc[0])
        for asset, group in positions.groupby("asset", sort=True)
    }


def run_liquidation_cascade(
    positions: pd.DataFrame,
    initial_shocks: dict[str, float],
    market_depth_usd: dict[str, float],
    close_factor: float = 0.50,
    liquidation_bonus: float = 0.05,
    price_impact_factor: float = 1.0,
    max_rounds: int = 20,
    min_liquidation_usd: float = 1.0,
) -> LiquidationCascadeResult:
    """
    Simulate a liquidation cascade with endogenous collateral price impact.

    The simulation starts from an exogenous asset-price shock. Positions with a
    health factor below one are partially liquidated up to ``close_factor``.
    Seized collateral is treated as market sell pressure. Each asset's price is
    then reduced according to an exponential market-impact curve:

        next_price = price * exp(-impact_factor * sold_usd / market_depth_usd)

    The lower the configured market depth, the larger the endogenous price
    impact from a given amount of collateral sales. The process repeats until
    there are no liquidatable positions, liquidation activity falls below the
    minimum size, or ``max_rounds`` is reached.

    This is a stress-testing model, not a prediction of realized liquidation
    execution or a recommendation for protocol parameters.
    """
    _validate_inputs(
        positions=positions,
        initial_shocks=initial_shocks,
        market_depth_usd=market_depth_usd,
        close_factor=close_factor,
        liquidation_bonus=liquidation_bonus,
        price_impact_factor=price_impact_factor,
        max_rounds=max_rounds,
        min_liquidation_usd=min_liquidation_usd,
    )

    state = positions.copy(deep=True).reset_index(drop=True)
    state["asset"] = state["asset"].astype(str)
    state["initial_debt_usd"] = state["debt_usd"].astype(float)
    state["initial_collateral_amount"] = state["collateral_amount"].astype(float)

    base_prices = _asset_prices(state)
    shocked_prices = {
        asset: base_price * (1 + initial_shocks.get(asset, 0.0))
        for asset, base_price in base_prices.items()
    }
    prices = shocked_prices.copy()

    _recalculate_state(state, prices)

    initial_liquidatable = state["liquidatable"].copy()
    initial_liquidatable_debt = float(
        state.loc[initial_liquidatable, "debt_usd"].sum()
    )

    state["ever_liquidatable"] = initial_liquidatable
    state["first_liquidation_round"] = pd.Series(
        [0 if flag else pd.NA for flag in initial_liquidatable],
        dtype="Int64",
    )

    history_rows: list[dict[str, float | int | str]] = []
    rounds_executed = 0

    for round_number in range(1, max_rounds + 1):
        _recalculate_state(state, prices)
        active = (
            state["liquidatable"]
            & (state["debt_usd"] > min_liquidation_usd)
            & (state["collateral_amount"] > 0)
        )

        if not active.any():
            break

        starting_prices = prices.copy()
        sold_usd_by_asset = {asset: 0.0 for asset in prices}
        repaid_usd_by_asset = {asset: 0.0 for asset in prices}
        executions_by_asset = {asset: 0 for asset in prices}

        for index in state.index[active]:
            asset = str(state.at[index, "asset"])
            price = prices[asset]
            debt = float(state.at[index, "debt_usd"])
            collateral_amount = float(state.at[index, "collateral_amount"])

            desired_repayment = debt * close_factor
            max_repayment_from_collateral = (
                collateral_amount * price / (1 + liquidation_bonus)
            )
            debt_repaid = min(
                desired_repayment,
                max_repayment_from_collateral,
            )

            if debt_repaid < min_liquidation_usd:
                continue

            collateral_sold_usd = debt_repaid * (1 + liquidation_bonus)
            collateral_seized = collateral_sold_usd / price

            state.at[index, "debt_usd"] = max(0.0, debt - debt_repaid)
            state.at[index, "collateral_amount"] = max(
                0.0,
                collateral_amount - collateral_seized,
            )

            sold_usd_by_asset[asset] += collateral_sold_usd
            repaid_usd_by_asset[asset] += debt_repaid
            executions_by_asset[asset] += 1

        total_repaid = sum(repaid_usd_by_asset.values())
        if total_repaid < min_liquidation_usd:
            break

        for asset, starting_price in starting_prices.items():
            depth = market_depth_usd[asset]
            sold_usd = sold_usd_by_asset[asset]
            impact_load = price_impact_factor * sold_usd / depth
            impact_load = min(impact_load, 50.0)
            prices[asset] = starting_price * math.exp(-impact_load)

        _recalculate_state(state, prices)

        newly_liquidatable = state["liquidatable"] & ~state["ever_liquidatable"]
        state.loc[newly_liquidatable, "first_liquidation_round"] = round_number
        state.loc[newly_liquidatable, "ever_liquidatable"] = True

        for asset in sorted(prices):
            asset_mask = state["asset"] == asset
            newly_for_asset = int((newly_liquidatable & asset_mask).sum())
            ending_price = prices[asset]
            starting_price = starting_prices[asset]
            price_impact_pct = ending_price / starting_price - 1

            history_rows.append(
                {
                    "round": round_number,
                    "asset": asset,
                    "starting_price": starting_price,
                    "ending_price": ending_price,
                    "market_depth_usd": market_depth_usd[asset],
                    "collateral_sold_usd": sold_usd_by_asset[asset],
                    "debt_repaid_usd": repaid_usd_by_asset[asset],
                    "liquidation_executions": executions_by_asset[asset],
                    "price_impact_pct": price_impact_pct,
                    "newly_liquidatable_positions": newly_for_asset,
                    "liquidatable_positions_after": int(
                        (state["liquidatable"] & asset_mask).sum()
                    ),
                    "bad_debt_usd_after": float(
                        state.loc[asset_mask, "bad_debt_usd"].sum()
                    ),
                }
            )

        rounds_executed = round_number

    _recalculate_state(state, prices)

    state["cascade_liquidated"] = (
        state["first_liquidation_round"].fillna(0).astype(int) > 0
    )

    round_history = pd.DataFrame(history_rows)
    total_collateral_sold = (
        float(round_history["collateral_sold_usd"].sum())
        if not round_history.empty
        else 0.0
    )
    total_debt_repaid = float(
        state["initial_debt_usd"].sum() - state["debt_usd"].sum()
    )

    asset_rows = []
    for asset in sorted(prices):
        asset_mask = state["asset"] == asset
        asset_history = (
            round_history.loc[round_history["asset"] == asset]
            if not round_history.empty
            else pd.DataFrame()
        )
        initial_debt = float(state.loc[asset_mask, "initial_debt_usd"].sum())
        final_debt = float(state.loc[asset_mask, "debt_usd"].sum())
        collateral_sold = (
            float(asset_history["collateral_sold_usd"].sum())
            if not asset_history.empty
            else 0.0
        )

        asset_rows.append(
            {
                "asset": asset,
                "base_price": base_prices[asset],
                "shocked_price": shocked_prices[asset],
                "final_price": prices[asset],
                "endogenous_price_impact_pct": (
                    prices[asset] / shocked_prices[asset] - 1
                ),
                "market_depth_usd": market_depth_usd[asset],
                "initial_debt_usd": initial_debt,
                "debt_repaid_usd": initial_debt - final_debt,
                "final_debt_usd": final_debt,
                "collateral_sold_usd": collateral_sold,
                "initial_liquidatable_positions": int(
                    (initial_liquidatable & asset_mask).sum()
                ),
                "cascade_liquidated_positions": int(
                    state.loc[asset_mask, "cascade_liquidated"].sum()
                ),
                "ever_liquidatable_positions": int(
                    state.loc[asset_mask, "ever_liquidatable"].sum()
                ),
                "remaining_liquidatable_positions": int(
                    state.loc[asset_mask, "liquidatable"].sum()
                ),
                "bad_debt_usd": float(
                    state.loc[asset_mask, "bad_debt_usd"].sum()
                ),
            }
        )

    asset_summary = pd.DataFrame(asset_rows)
    max_rounds_reached = bool(
        rounds_executed == max_rounds and state["liquidatable"].any()
    )

    summary: dict[str, float | int | bool] = {
        "positions": int(len(state)),
        "rounds_executed": int(rounds_executed),
        "max_rounds_reached": max_rounds_reached,
        "initial_total_debt_usd": float(state["initial_debt_usd"].sum()),
        "initial_liquidatable_positions": int(initial_liquidatable.sum()),
        "initial_liquidatable_debt_usd": initial_liquidatable_debt,
        "cascade_liquidated_positions": int(state["cascade_liquidated"].sum()),
        "ever_liquidatable_positions": int(state["ever_liquidatable"].sum()),
        "remaining_liquidatable_positions": int(state["liquidatable"].sum()),
        "debt_repaid_usd": total_debt_repaid,
        "remaining_debt_usd": float(state["debt_usd"].sum()),
        "collateral_sold_usd": total_collateral_sold,
        "bad_debt_usd": float(state["bad_debt_usd"].sum()),
    }

    return LiquidationCascadeResult(
        final_positions=state,
        round_history=round_history,
        asset_summary=asset_summary,
        summary=summary,
    )
