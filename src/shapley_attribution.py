from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd

from src.simulation import simulate_market_returns
from src.tail_risk_attribution import _evaluate_scenario, _validate_inputs


ATTRIBUTION_METRICS = (
    "first_order_debt_share",
    "cascade_debt_share",
    "amplification_debt_share",
    "bad_debt_share",
    "cascade_created_positions",
    "max_endogenous_price_decline",
)


@dataclass
class ShapleyScenarioResult:
    """Exact coalition values and Shapley contributions for one market scenario."""

    full_metrics: dict[str, float | int]
    baseline_metrics: dict[str, float | int]
    coalition_values: pd.DataFrame
    shapley_values: pd.DataFrame


@dataclass
class ShapleyTailRiskResult:
    """Paired scenario and asset-level exact Shapley attribution outputs."""

    scenario_results: pd.DataFrame
    shapley_results: pd.DataFrame
    asset_summary: pd.DataFrame
    tail_quantile: float


def _coalitions(assets: list[str]):
    for size in range(len(assets) + 1):
        yield from combinations(assets, size)


def _coalition_label(coalition: tuple[str, ...]) -> str:
    return "+".join(coalition) if coalition else "NONE"


def _validate_shocks(assets: list[str], shocks: dict[str, float]) -> dict[str, float]:
    normalized = {asset: float(shocks.get(asset, 0.0)) for asset in assets}
    for asset, shock in normalized.items():
        if not math.isfinite(shock) or shock <= -1:
            raise ValueError(
                f"shock for {asset} must be finite and greater than -1"
            )
    return normalized


def shapley_market_scenario(
    positions: pd.DataFrame,
    shocks: dict[str, float],
    market_depth_usd: dict[str, float],
    close_factor: float = 0.50,
    liquidation_bonus: float = 0.05,
    price_impact_factor: float = 1.0,
    max_rounds: int = 20,
    min_liquidation_usd: float = 1.0,
) -> ShapleyScenarioResult:
    """Compute an exact Shapley decomposition across collateral-asset shocks.

    For every subset of collateral assets, assets inside the coalition retain
    their scenario shock and assets outside the coalition are set to a zero
    exogenous return. The exact Shapley value averages each asset's marginal
    contribution over all possible coalition orderings.

    With ETH, BTC, and SOL this requires only 2^3 = 8 deterministic scenario
    evaluations, so no Monte Carlo approximation of the Shapley values is
    necessary.
    """
    assets = _validate_inputs(
        positions=positions,
        market_depth_usd=market_depth_usd,
        tail_quantile=0.95,
    )
    normalized_shocks = _validate_shocks(assets, shocks)

    coalition_metrics: dict[frozenset[str], dict[str, float | int]] = {}
    coalition_rows: list[dict[str, float | int | str]] = []

    for coalition_tuple in _coalitions(assets):
        coalition = frozenset(coalition_tuple)
        coalition_shocks = {
            asset: normalized_shocks[asset] if asset in coalition else 0.0
            for asset in assets
        }
        metrics = _evaluate_scenario(
            positions=positions,
            shocks=coalition_shocks,
            market_depth_usd=market_depth_usd,
            close_factor=close_factor,
            liquidation_bonus=liquidation_bonus,
            price_impact_factor=price_impact_factor,
            max_rounds=max_rounds,
            min_liquidation_usd=min_liquidation_usd,
        )
        coalition_metrics[coalition] = metrics
        coalition_rows.append(
            {
                "coalition": _coalition_label(coalition_tuple),
                "coalition_size": len(coalition),
                **{f"{asset.lower()}_active": int(asset in coalition) for asset in assets},
                **metrics,
            }
        )

    n_assets = len(assets)
    factorial_n = math.factorial(n_assets)
    shapley_rows: list[dict[str, float | str]] = []

    for asset in assets:
        values = {metric: 0.0 for metric in ATTRIBUTION_METRICS}

        others = [candidate for candidate in assets if candidate != asset]
        for subset_size in range(len(others) + 1):
            weight = (
                math.factorial(subset_size)
                * math.factorial(n_assets - subset_size - 1)
                / factorial_n
            )
            for subset_tuple in combinations(others, subset_size):
                subset = frozenset(subset_tuple)
                with_asset = subset | {asset}
                without_metrics = coalition_metrics[subset]
                with_metrics = coalition_metrics[with_asset]

                for metric in ATTRIBUTION_METRICS:
                    values[metric] += weight * (
                        float(with_metrics[metric]) - float(without_metrics[metric])
                    )

        shapley_rows.append(
            {
                "asset": asset,
                "asset_shock": normalized_shocks[asset],
                **{f"shapley_{metric}": value for metric, value in values.items()},
            }
        )

    baseline = coalition_metrics[frozenset()]
    full = coalition_metrics[frozenset(assets)]
    shapley_values = pd.DataFrame(shapley_rows)

    for metric in ATTRIBUTION_METRICS:
        allocated = float(shapley_values[f"shapley_{metric}"].sum())
        total_change = float(full[metric]) - float(baseline[metric])
        shapley_values[f"efficiency_residual_{metric}"] = allocated - total_change

    return ShapleyScenarioResult(
        full_metrics=full,
        baseline_metrics=baseline,
        coalition_values=pd.DataFrame(coalition_rows).sort_values(
            ["coalition_size", "coalition"]
        ).reset_index(drop=True),
        shapley_values=shapley_values,
    )


def run_shapley_tail_risk_attribution(
    positions: pd.DataFrame,
    market_depth_usd: dict[str, float],
    n_simulations: int = 25,
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
) -> ShapleyTailRiskResult:
    """Run exact Shapley attribution over paired stochastic market draws."""
    assets = _validate_inputs(
        positions=positions,
        market_depth_usd=market_depth_usd,
        tail_quantile=tail_quantile,
    )
    if not isinstance(n_simulations, int) or isinstance(n_simulations, bool) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")
    if not isinstance(horizon_days, int) or isinstance(horizon_days, bool) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")
    if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer")

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
    shapley_frames: list[pd.DataFrame] = []

    for simulation_id, row in returns.iterrows():
        shocks = {
            asset: max(float(row.get(asset, 0.0)), -0.99)
            for asset in assets
        }
        result = shapley_market_scenario(
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
            **result.full_metrics,
        }
        for metric in ATTRIBUTION_METRICS:
            allocated = float(result.shapley_values[f"shapley_{metric}"].sum())
            total_change = (
                float(result.full_metrics[metric])
                - float(result.baseline_metrics[metric])
            )
            scenario_row[f"shapley_efficiency_error_{metric}"] = (
                allocated - total_change
            )
        scenario_rows.append(scenario_row)

        frame = result.shapley_values.copy()
        frame.insert(0, "simulation_id", int(simulation_id))
        shapley_frames.append(frame)

    scenario_results = pd.DataFrame(scenario_rows)
    shapley_results = pd.concat(shapley_frames, ignore_index=True)

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
        rows = shapley_results.loc[shapley_results["asset"] == asset]
        exposure_tail = rows.loc[rows["simulation_id"].isin(exposure_tail_ids)]
        bad_debt_tail = rows.loc[rows["simulation_id"].isin(bad_debt_tail_ids)]

        exposure = rows["shapley_cascade_debt_share"]
        bad_debt = rows["shapley_bad_debt_share"]

        summary_rows.append(
            {
                "asset": asset,
                "simulations": int(len(rows)),
                "mean_shapley_cascade_exposure": float(exposure.mean()),
                "p95_shapley_cascade_exposure": float(exposure.quantile(0.95)),
                "tail_mean_shapley_cascade_exposure": float(
                    exposure_tail["shapley_cascade_debt_share"].mean()
                ),
                "probability_positive_shapley_exposure": float((exposure > 0).mean()),
                "mean_shapley_bad_debt": float(bad_debt.mean()),
                "p95_shapley_bad_debt": float(bad_debt.quantile(0.95)),
                "tail_mean_shapley_bad_debt": float(
                    bad_debt_tail["shapley_bad_debt_share"].mean()
                ),
                "mean_shapley_amplification": float(
                    rows["shapley_amplification_debt_share"].mean()
                ),
                "mean_shapley_secondary_liquidations": float(
                    rows["shapley_cascade_created_positions"].mean()
                ),
                "mean_shapley_endogenous_price_decline": float(
                    rows["shapley_max_endogenous_price_decline"].mean()
                ),
            }
        )

    asset_summary = pd.DataFrame(summary_rows).sort_values(
        "tail_mean_shapley_cascade_exposure",
        ascending=False,
    ).reset_index(drop=True)

    return ShapleyTailRiskResult(
        scenario_results=scenario_results,
        shapley_results=shapley_results,
        asset_summary=asset_summary,
        tail_quantile=tail_quantile,
    )
