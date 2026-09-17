from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.liquidation_cascade import run_liquidation_cascade
from src.simulation import simulate_market_returns
from src.stress_engine import run_asset_specific_stress_test


@dataclass
class TailRiskAttributionResult:
    """Paired scenario-level and asset-level tail-risk attribution outputs."""

    scenario_results: pd.DataFrame
    attribution_results: pd.DataFrame
    asset_summary: pd.DataFrame
    tail_quantile: float


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_inputs(
    positions: pd.DataFrame,
    market_depth_usd: dict[str, float],
    tail_quantile: float,
) -> list[str]:
    required = {
        "position_id",
        "asset",
        "collateral_amount",
        "collateral_price",
        "debt_usd",
        "liquidation_threshold",
    }
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(
            "positions is missing required columns: " + ", ".join(sorted(missing))
        )
    if positions.empty:
        raise ValueError("positions must contain at least one lending position")

    total_debt = float(positions["debt_usd"].sum())
    if not math.isfinite(total_debt) or total_debt <= 0:
        raise ValueError("positions must contain positive finite protocol debt")

    if not math.isfinite(tail_quantile) or not 0 < tail_quantile < 1:
        raise ValueError("tail_quantile must be finite and in the interval (0, 1)")

    assets = sorted(positions["asset"].astype(str).unique().tolist())
    for asset in assets:
        depth = market_depth_usd.get(asset)
        if depth is None or not math.isfinite(depth) or depth <= 0:
            raise ValueError(
                f"market_depth_usd must contain a positive finite value for {asset}"
            )

    return assets


def _evaluate_scenario(
    positions: pd.DataFrame,
    shocks: dict[str, float],
    market_depth_usd: dict[str, float],
    close_factor: float,
    liquidation_bonus: float,
    price_impact_factor: float,
    max_rounds: int,
    min_liquidation_usd: float,
) -> dict[str, float | int]:
    total_debt = float(positions["debt_usd"].sum())

    first_order = run_asset_specific_stress_test(positions, shocks)
    first_order_mask = first_order["liquidatable"]
    first_order_debt = float(
        first_order.loc[first_order_mask, "debt_usd"].sum()
    )

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
        final_positions.loc[ever_liquidatable, "initial_debt_usd"].sum()
    )

    first_order_share = first_order_debt / total_debt
    cascade_share = cascade_exposed_debt / total_debt
    amplification_share = max(0.0, cascade_share - first_order_share)
    bad_debt_share = float(cascade.summary["bad_debt_usd"]) / total_debt

    if cascade.asset_summary.empty:
        max_endogenous_price_decline = 0.0
    else:
        max_endogenous_price_decline = max(
            0.0,
            -float(cascade.asset_summary["endogenous_price_impact_pct"].min()),
        )

    return {
        "first_order_debt_share": first_order_share,
        "cascade_debt_share": cascade_share,
        "amplification_debt_share": amplification_share,
        "bad_debt_share": bad_debt_share,
        "cascade_created_positions": int(
            cascade.summary["cascade_liquidated_positions"]
        ),
        "max_endogenous_price_decline": max_endogenous_price_decline,
    }


def attribute_market_scenario(
    positions: pd.DataFrame,
    shocks: dict[str, float],
    market_depth_usd: dict[str, float],
    close_factor: float = 0.50,
    liquidation_bonus: float = 0.05,
    price_impact_factor: float = 1.0,
    max_rounds: int = 20,
    min_liquidation_usd: float = 1.0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Attribute one deterministic market scenario by neutralizing one asset at a time.

    The full scenario is evaluated once. For each collateral asset, the exact same
    scenario is re-evaluated with only that asset's exogenous return set to zero.
    The paired difference is that asset's leave-one-out contribution under the
    model. Contributions can overlap because cascade interactions are nonlinear,
    so they should not be interpreted as an additive decomposition.
    """
    assets = _validate_inputs(
        positions=positions,
        market_depth_usd=market_depth_usd,
        tail_quantile=0.95,
    )

    normalized_shocks = {
        asset: float(shocks.get(asset, 0.0))
        for asset in assets
    }
    for asset, shock in normalized_shocks.items():
        if not math.isfinite(shock) or shock <= -1:
            raise ValueError(
                f"shock for {asset} must be finite and greater than -1"
            )

    full = _evaluate_scenario(
        positions=positions,
        shocks=normalized_shocks,
        market_depth_usd=market_depth_usd,
        close_factor=close_factor,
        liquidation_bonus=liquidation_bonus,
        price_impact_factor=price_impact_factor,
        max_rounds=max_rounds,
        min_liquidation_usd=min_liquidation_usd,
    )

    rows: list[dict[str, float | int | str]] = []
    metrics = (
        "first_order_debt_share",
        "cascade_debt_share",
        "amplification_debt_share",
        "bad_debt_share",
        "cascade_created_positions",
        "max_endogenous_price_decline",
    )

    for asset in assets:
        muted_shocks = normalized_shocks.copy()
        muted_shocks[asset] = 0.0
        muted = _evaluate_scenario(
            positions=positions,
            shocks=muted_shocks,
            market_depth_usd=market_depth_usd,
            close_factor=close_factor,
            liquidation_bonus=liquidation_bonus,
            price_impact_factor=price_impact_factor,
            max_rounds=max_rounds,
            min_liquidation_usd=min_liquidation_usd,
        )

        row: dict[str, float | int | str] = {
            "asset": asset,
            "asset_shock": normalized_shocks[asset],
        }
        for metric in metrics:
            row[f"full_{metric}"] = full[metric]
            row[f"muted_{metric}"] = muted[metric]
            row[f"contribution_{metric}"] = (
                float(full[metric]) - float(muted[metric])
            )
        rows.append(row)

    return full, pd.DataFrame(rows)


def run_tail_risk_attribution(
    positions: pd.DataFrame,
    market_depth_usd: dict[str, float],
    n_simulations: int = 50,
    horizon_days: int = 30,
    seed: int = 42,
    distribution: str = "student_t",
    degrees_of_freedom: int = 5,
    annual_volatility: dict[str, float] | None = None,
    correlation_matrix: np.ndarray | None = None,
    tail_quantile: float = 0.95,
    close_factor: float = 0.50,
    liquidation_bonus: float = 0.05,
    price_impact_factor: float = 1.0,
    max_rounds: int = 20,
    min_liquidation_usd: float = 1.0,
) -> TailRiskAttributionResult:
    """Run paired leave-one-asset-out attribution over identical market draws."""
    assets = _validate_inputs(
        positions=positions,
        market_depth_usd=market_depth_usd,
        tail_quantile=tail_quantile,
    )
    _validate_positive_integer(n_simulations, "n_simulations")
    _validate_positive_integer(horizon_days, "horizon_days")
    _validate_positive_integer(max_rounds, "max_rounds")

    returns = simulate_market_returns(
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        seed=seed,
        distribution=distribution,
        degrees_of_freedom=degrees_of_freedom,
        annual_volatility=annual_volatility,
        correlation_matrix=correlation_matrix,
    )

    scenario_rows: list[dict[str, float | int]] = []
    attribution_frames: list[pd.DataFrame] = []

    for simulation_id, row in returns.iterrows():
        shocks = {
            asset: max(float(row.get(asset, 0.0)), -0.99)
            for asset in assets
        }
        full, attribution = attribute_market_scenario(
            positions=positions,
            shocks=shocks,
            market_depth_usd=market_depth_usd,
            close_factor=close_factor,
            liquidation_bonus=liquidation_bonus,
            price_impact_factor=price_impact_factor,
            max_rounds=max_rounds,
            min_liquidation_usd=min_liquidation_usd,
        )

        scenario_row: dict[str, float | int] = {
            "simulation_id": int(simulation_id),
            **{f"{asset.lower()}_return": shocks[asset] for asset in assets},
            **full,
        }
        scenario_rows.append(scenario_row)

        attribution = attribution.copy()
        attribution.insert(0, "simulation_id", int(simulation_id))
        attribution_frames.append(attribution)

    scenario_results = pd.DataFrame(scenario_rows)
    attribution_results = pd.concat(attribution_frames, ignore_index=True)

    exposure_cutoff = float(
        scenario_results["cascade_debt_share"].quantile(tail_quantile)
    )
    bad_debt_cutoff = float(
        scenario_results["bad_debt_share"].quantile(tail_quantile)
    )
    exposure_tail_ids = set(
        scenario_results.loc[
            scenario_results["cascade_debt_share"] >= exposure_cutoff,
            "simulation_id",
        ].astype(int)
    )
    bad_debt_tail_ids = set(
        scenario_results.loc[
            scenario_results["bad_debt_share"] >= bad_debt_cutoff,
            "simulation_id",
        ].astype(int)
    )

    summary_rows: list[dict[str, float | int | str]] = []
    for asset in assets:
        rows = attribution_results.loc[attribution_results["asset"] == asset]
        exposure_tail = rows.loc[rows["simulation_id"].isin(exposure_tail_ids)]
        bad_debt_tail = rows.loc[rows["simulation_id"].isin(bad_debt_tail_ids)]

        exposure_contribution = rows["contribution_cascade_debt_share"]
        bad_debt_contribution = rows["contribution_bad_debt_share"]

        summary_rows.append(
            {
                "asset": asset,
                "simulations": int(len(rows)),
                "mean_cascade_exposure_contribution": float(
                    exposure_contribution.mean()
                ),
                "p95_cascade_exposure_contribution": float(
                    exposure_contribution.quantile(0.95)
                ),
                "tail_mean_cascade_exposure_contribution": float(
                    exposure_tail["contribution_cascade_debt_share"].mean()
                ),
                "probability_positive_exposure_contribution": float(
                    (exposure_contribution > 0).mean()
                ),
                "mean_bad_debt_contribution": float(bad_debt_contribution.mean()),
                "p95_bad_debt_contribution": float(
                    bad_debt_contribution.quantile(0.95)
                ),
                "tail_mean_bad_debt_contribution": float(
                    bad_debt_tail["contribution_bad_debt_share"].mean()
                ),
                "mean_amplification_contribution": float(
                    rows["contribution_amplification_debt_share"].mean()
                ),
                "mean_secondary_liquidation_contribution": float(
                    rows["contribution_cascade_created_positions"].mean()
                ),
                "mean_endogenous_price_decline_contribution": float(
                    rows["contribution_max_endogenous_price_decline"].mean()
                ),
            }
        )

    asset_summary = pd.DataFrame(summary_rows).sort_values(
        "tail_mean_cascade_exposure_contribution",
        ascending=False,
    ).reset_index(drop=True)

    return TailRiskAttributionResult(
        scenario_results=scenario_results,
        attribution_results=attribution_results,
        asset_summary=asset_summary,
        tail_quantile=tail_quantile,
    )
