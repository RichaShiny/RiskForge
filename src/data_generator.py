import numpy as np
import pandas as pd


ASSET_CONFIG = {
    "ETH": {
        "price": 4000.0,
        "liquidation_threshold": 0.80,
    },
    "BTC": {
        "price": 65000.0,
        "liquidation_threshold": 0.75,
    },
    "SOL": {
        "price": 180.0,
        "liquidation_threshold": 0.70,
    },
}


def generate_protocol_positions(
    n_positions: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    records = []

    assets = list(ASSET_CONFIG.keys())

    for position_id in range(1, n_positions + 1):
        asset = rng.choice(
            assets,
            p=[0.55, 0.30, 0.15],
        )

        config = ASSET_CONFIG[asset]

        collateral_value = rng.lognormal(
            mean=10.5,
            sigma=1.0,
        )

        initial_ltv = rng.uniform(
            0.25,
            config["liquidation_threshold"] * 0.98,
        )

        debt_usd = collateral_value * initial_ltv

        collateral_amount = (
            collateral_value / config["price"]
        )

        health_factor = (
            collateral_value
            * config["liquidation_threshold"]
            / debt_usd
        )

        records.append(
            {
                "position_id": position_id,
                "asset": asset,
                "collateral_amount": collateral_amount,
                "collateral_price": config["price"],
                "collateral_value": collateral_value,
                "debt_usd": debt_usd,
                "liquidation_threshold": config[
                    "liquidation_threshold"
                ],
                "ltv": initial_ltv,
                "health_factor": health_factor,
            }
        )

    return pd.DataFrame(records)