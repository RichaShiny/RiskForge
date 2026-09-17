from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import pandas as pd

from src.liquidation_cascade import run_liquidation_cascade
from src.stress_engine import run_asset_specific_stress_test


ReverseStressMetric = Literal[
    "first_order_debt_share",
    "cascade_debt_share",
    "amplification_debt_share",
    "bad_debt_share",
]

SUPPORTED_METRICS: tuple[ReverseStressMetric, ...] = (
    "first_order_debt_share",
    "cascade_debt_share",
    "amplification_debt_share",
    "bad_debt_share",
)


@dataclass(frozen=True)
class ReverseStressPoint:
    decline: float
    shocks: dict[str, float]
    first_order_debt_share: float
    cascade_debt_share: float
    amplification_debt_share: float
    bad_debt_share: float
    cascade_created_positions: int
    rounds_executed: int


@dataclass(frozen=True)
class ReverseStressResult:
    metric: ReverseStressMetric
    target: float
    found: bool
    critical_decline: float | None
    metric_value: float | None
    lower_safe_decline: float | None
    lower_safe_value: float | None
    iterations: int
    tolerance: float
    point: ReverseStressPoint | None


def _validate_fraction(value: float, name: str, *, allow_zero: bool = True) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    lower_valid = value >= 0 if allow_zero else value > 0
    if not lower_valid or value > 1:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")


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


def _validate_weights(
    positions: pd.DataFrame,
    asset_stress_weights: dict[str, float] | None,
) -> dict[str, float]:
    assets = sorted(set(positions["asset"].astype(str)))
    if asset_stress_weights is None:
        return {asset: 1.0 for asset in assets}

    missing = sorted(set(assets).difference(asset_stress_weights))
    if missing:
        raise ValueError(
            "asset_stress_weights is missing assets: " + ", ".join(missing)
        )

    weights: dict[str, float] = {}
    for asset in assets:
        weight = asset_stress_weights[asset]
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"stress weight for {asset} must be finite and non-negative"
            )
        weights[asset] = float(weight)
    return weights


def _shocks_from_decline(
    decline: float,
    weights: dict[str, float],
) -> dict[str, float]:
    return {
        asset: max(-0.99, -decline * weight)
        for asset, weight in weights.items()
    }


def evaluate_reverse_stress_point(
    positions: pd.DataFrame,
    decline: float,
    market_depth_usd: dict[str, float],
    asset_stress_weights: dict[str, float] | None = None,
    close_factor: float = 0.50,
    liquidation_bonus: float = 0.05,
    price_impact_factor: float = 1.0,
    max_rounds: int = 20,
    min_liquidation_usd: float = 1.0,
) -> ReverseStressPoint:
    """Evaluate first-order and cascade risk at one common stress magnitude.

    ``decline`` is expressed as a positive fraction. A value of 0.20 means a
    20% base market decline. Optional asset stress weights scale that decline by
    asset, e.g. {"ETH": 1.0, "BTC": 0.8, "SOL": 1.3}.
    """
    if positions.empty:
        raise ValueError("positions must contain at least one lending position")
    _validate_fraction(decline, "decline")
    _validate_market_depth(positions, market_depth_usd)
    weights = _validate_weights(positions, asset_stress_weights)

    total_debt = float(positions["debt_usd"].sum())
    if total_debt <= 0:
        raise ValueError("positions must contain positive protocol debt")

    shocks = _shocks_from_decline(decline, weights)

    first_order = run_asset_specific_stress_test(
        positions=positions,
        asset_shocks=shocks,
    )
    first_order_mask = first_order["liquidatable"].astype(bool)
    first_order_debt = float(
        first_order.loc[first_order_mask, "debt_usd"].sum()
    )
    first_order_share = first_order_debt / total_debt

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

    ever_liquidatable = cascade.final_positions["ever_liquidatable"].astype(bool)
    cascade_exposed_debt = float(
        cascade.final_positions.loc[
            ever_liquidatable,
            "initial_debt_usd",
        ].sum()
    )
    cascade_share = cascade_exposed_debt / total_debt
    amplification_share = max(0.0, cascade_share - first_order_share)
    bad_debt_share = float(cascade.summary["bad_debt_usd"]) / total_debt

    return ReverseStressPoint(
        decline=float(decline),
        shocks=shocks,
        first_order_debt_share=first_order_share,
        cascade_debt_share=cascade_share,
        amplification_debt_share=amplification_share,
        bad_debt_share=bad_debt_share,
        cascade_created_positions=int(
            cascade.summary["cascade_liquidated_positions"]
        ),
        rounds_executed=int(cascade.summary["rounds_executed"]),
    )


def find_critical_decline(
    positions: pd.DataFrame,
    metric: ReverseStressMetric,
    target: float,
    market_depth_usd: dict[str, float],
    asset_stress_weights: dict[str, float] | None = None,
    max_decline: float = 0.90,
    tolerance: float = 0.001,
    max_iterations: int = 40,
    **cascade_kwargs,
) -> ReverseStressResult:
    """Find the smallest modeled market decline that reaches a risk target.

    The search evaluates the same deterministic protocol and cascade mechanics
    at progressively tighter shock brackets. Results should be interpreted as
    model-based reverse stress thresholds, not forecasts of future market moves.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            "metric must be one of: " + ", ".join(SUPPORTED_METRICS)
        )
    _validate_fraction(target, "target")
    _validate_fraction(max_decline, "max_decline", allow_zero=False)
    if not math.isfinite(tolerance) or tolerance <= 0 or tolerance >= max_decline:
        raise ValueError("tolerance must be positive and smaller than max_decline")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")

    safe_point = evaluate_reverse_stress_point(
        positions=positions,
        decline=0.0,
        market_depth_usd=market_depth_usd,
        asset_stress_weights=asset_stress_weights,
        **cascade_kwargs,
    )
    safe_value = float(getattr(safe_point, metric))

    if safe_value >= target:
        return ReverseStressResult(
            metric=metric,
            target=target,
            found=True,
            critical_decline=0.0,
            metric_value=safe_value,
            lower_safe_decline=None,
            lower_safe_value=None,
            iterations=0,
            tolerance=tolerance,
            point=safe_point,
        )

    breach_point = evaluate_reverse_stress_point(
        positions=positions,
        decline=max_decline,
        market_depth_usd=market_depth_usd,
        asset_stress_weights=asset_stress_weights,
        **cascade_kwargs,
    )
    breach_value = float(getattr(breach_point, metric))

    if breach_value < target:
        return ReverseStressResult(
            metric=metric,
            target=target,
            found=False,
            critical_decline=None,
            metric_value=None,
            lower_safe_decline=max_decline,
            lower_safe_value=breach_value,
            iterations=0,
            tolerance=tolerance,
            point=None,
        )

    lower_decline = 0.0
    upper_decline = max_decline
    lower_point = safe_point
    upper_point = breach_point
    iterations = 0

    while (
        upper_decline - lower_decline > tolerance
        and iterations < max_iterations
    ):
        midpoint = (lower_decline + upper_decline) / 2
        point = evaluate_reverse_stress_point(
            positions=positions,
            decline=midpoint,
            market_depth_usd=market_depth_usd,
            asset_stress_weights=asset_stress_weights,
            **cascade_kwargs,
        )
        value = float(getattr(point, metric))

        if value >= target:
            upper_decline = midpoint
            upper_point = point
        else:
            lower_decline = midpoint
            lower_point = point

        iterations += 1

    return ReverseStressResult(
        metric=metric,
        target=target,
        found=True,
        critical_decline=upper_decline,
        metric_value=float(getattr(upper_point, metric)),
        lower_safe_decline=lower_decline,
        lower_safe_value=float(getattr(lower_point, metric)),
        iterations=iterations,
        tolerance=tolerance,
        point=upper_point,
    )


def build_reverse_stress_frontier(
    positions: pd.DataFrame,
    metric: ReverseStressMetric,
    targets: list[float],
    market_depth_usd: dict[str, float],
    asset_stress_weights: dict[str, float] | None = None,
    max_decline: float = 0.90,
    tolerance: float = 0.001,
    **cascade_kwargs,
) -> pd.DataFrame:
    """Solve a sequence of reverse-stress targets for one risk metric."""
    if not targets:
        raise ValueError("targets must contain at least one risk threshold")

    normalized_targets = sorted(set(float(target) for target in targets))
    rows: list[dict[str, float | int | bool | str | None]] = []

    for target in normalized_targets:
        result = find_critical_decline(
            positions=positions,
            metric=metric,
            target=target,
            market_depth_usd=market_depth_usd,
            asset_stress_weights=asset_stress_weights,
            max_decline=max_decline,
            tolerance=tolerance,
            **cascade_kwargs,
        )

        rows.append(
            {
                "metric": metric,
                "target": target,
                "found": result.found,
                "critical_decline": result.critical_decline,
                "metric_value": result.metric_value,
                "lower_safe_decline": result.lower_safe_decline,
                "lower_safe_value": result.lower_safe_value,
                "iterations": result.iterations,
                "cascade_created_positions": (
                    result.point.cascade_created_positions
                    if result.point is not None
                    else None
                ),
                "bad_debt_share_at_breach": (
                    result.point.bad_debt_share
                    if result.point is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)
