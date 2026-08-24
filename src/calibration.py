import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = {
    "ETH": "ETH-USD",
    "BTC": "BTC-USD",
    "SOL": "SOL-USD",
}


def download_historical_prices(
    start: str = "2022-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download historical daily crypto prices one asset at a time.
    """
    series = []

    for asset, ticker in TICKERS.items():
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            raise RuntimeError(
                f"Failed to download historical prices for {asset} ({ticker})."
            )

        close = data["Close"].copy()

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close.name = asset
        series.append(close)

    prices = pd.concat(
        series,
        axis=1,
        join="inner",
    )

    return prices.dropna()


def calculate_log_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily log returns.
    """
    return np.log(
        prices / prices.shift(1)
    ).dropna()


def calibrate_market_parameters(
    prices: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Estimate annualized volatility and the return
    correlation matrix from historical daily prices.
    """
    returns = calculate_log_returns(prices)

    annualized_volatility = (
        returns.std() * np.sqrt(365)
    )

    correlation_matrix = returns.corr()

    return (
        annualized_volatility,
        correlation_matrix,
    )