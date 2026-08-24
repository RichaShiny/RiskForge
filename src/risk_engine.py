from dataclasses import dataclass


@dataclass
class LendingPosition:
    collateral_asset: str
    collateral_amount: float
    collateral_price: float
    debt_usd: float
    liquidation_threshold: float

    @property
    def collateral_value(self) -> float:
        return self.collateral_amount * self.collateral_price

    @property
    def ltv(self) -> float:
        if self.collateral_value == 0:
            return 0.0

        return self.debt_usd / self.collateral_value

    @property
    def health_factor(self) -> float:
        if self.debt_usd == 0:
            return float("inf")

        return (
            self.collateral_value * self.liquidation_threshold
        ) / self.debt_usd

    @property
    def liquidation_price(self) -> float:
        if self.collateral_amount == 0:
            return 0.0

        return self.debt_usd / (
            self.collateral_amount * self.liquidation_threshold
        )

    @property
    def distance_to_liquidation(self) -> float:
        if self.collateral_price == 0:
            return 0.0

        return (
            self.collateral_price - self.liquidation_price
        ) / self.collateral_price

    @property
    def risk_status(self) -> str:
        hf = self.health_factor

        if hf < 1.0:
            return "LIQUIDATABLE"
        elif hf < 1.1:
            return "CRITICAL"
        elif hf < 1.25:
            return "HIGH RISK"
        elif hf < 1.5:
            return "MODERATE RISK"

        return "HEALTHY"