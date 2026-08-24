from pathlib import Path

import pandas as pd

from src.calibration import (
    calibrate_market_parameters,
    download_historical_prices,
)


prices = download_historical_prices(
    start="2022-01-01",
)

volatility, correlation = (
    calibrate_market_parameters(prices)
)

rows = []

for asset in volatility.index:
    rows.append(
        {
            "parameter": "annualized_volatility",
            "asset": asset,
            "asset_2": "",
            "value": volatility[asset],
        }
    )

for asset_1 in correlation.index:
    for asset_2 in correlation.columns:
        rows.append(
            {
                "parameter": "correlation",
                "asset": asset_1,
                "asset_2": asset_2,
                "value": correlation.loc[
                    asset_1,
                    asset_2,
                ],
            }
        )

snapshot = pd.DataFrame(rows)

output_path = Path(
    "data/calibration_snapshot.csv"
)

snapshot.to_csv(
    output_path,
    index=False,
)

print(
    f"Saved calibration snapshot to "
    f"{output_path}"
)

print("\nAnnualized volatility:")
print(volatility)

print("\nCorrelation matrix:")
print(correlation)