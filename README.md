# RiskForge

A quantitative protocol risk and stress-testing engine for simulating market shocks, estimating liquidation exposure, and analyzing systemic risk under uncertainty.

## Why I Built This

A lending position can look healthy under current market conditions while becoming vulnerable after relatively small changes in collateral prices.

That led to a broader question:

> How can we move from evaluating individual lending positions to measuring protocol-level risk under deterministic and stochastic market stress?

RiskForge explores that question through position-level risk modeling, deterministic stress testing, asset-specific scenarios, Monte Carlo simulation, historical market calibration, tail-risk analysis, and parameter sensitivity.

## Questions I Wanted to Answer

Rather than starting with a dashboard, I built the project around a sequence of risk questions:

1. When does an individual lending position become liquidatable?

2. How quickly does protocol exposure increase as markets decline?

3. Which collateral assets contribute most to liquidation exposure?

4. What happens when ETH, BTC, and SOL experience different shocks instead of moving identically?

5. Deterministic scenarios tell us what happens under a chosen crash, but what is the distribution of possible outcomes?

6. How often do severe outcomes occur in the tail of that distribution?

7. How sensitive are those results to assumptions about liquidation thresholds?

8. How much do simulation results change when volatility and cross-asset correlations are calibrated from historical market data rather than assumed?

These questions drove the architecture of RiskForge.

## Evaluation Flow

Position Data
↓
Position-Level Risk Metrics
↓
Deterministic Market Stress
↓
Protocol Liquidation Exposure
↓
Asset-Specific Stress Testing
↓
Monte Carlo Market Simulation
↓
Historical Volatility + Correlation Calibration
↓
Tail-Risk Analysis
↓
Liquidation-Threshold Sensitivity
↓
Validation and Automated Tests

Each stage answers a different question rather than simply adding another visualization.

## Methodology

### 1. Position-Level Risk

Each lending position contains collateral value, debt, and a liquidation threshold.

RiskForge derives metrics including:

- Loan-to-value ratio
- Health factor
- Liquidation price
- Distance to liquidation
- Risk status

This establishes the position-level mechanics used throughout the rest of the system.

### 2. Deterministic Stress Testing

The stress engine applies controlled collateral-price declines and recalculates position health.

A protocol shock ladder evaluates increasingly severe market declines and measures:

- Liquidatable positions
- Liquidation rate
- Liquidatable debt
- Share of protocol debt exposed to liquidation

This answers:

> How does protocol risk evolve as market conditions deteriorate?

### 3. Asset-Specific Scenarios

A single protocol-wide shock assumes every collateral asset moves identically.

RiskForge therefore supports independent ETH, BTC, and SOL shocks.

This allows scenarios such as:

- ETH: -25%
- BTC: -15%
- SOL: -40%

The resulting exposure can then be decomposed by collateral asset.

### 4. Monte Carlo Risk Simulation

Deterministic scenarios answer "what if this happens?"

Monte Carlo simulation asks:

> What range of outcomes could occur, and how frequently do severe outcomes appear?

RiskForge simulates correlated ETH, BTC, and SOL market returns and maps each simulated market state into protocol liquidation exposure.

Both Normal and Student-t return models are available so the effect of heavier-tailed return assumptions can be explored.

### 5. Historical Calibration

Rather than relying exclusively on assumed market parameters, RiskForge can estimate volatility and cross-asset dependence from historical ETH, BTC, and SOL prices.

The current calibration snapshot uses historical data beginning in January 2022.

Estimated annualized volatility:

| Asset | Volatility |
| --- | ---: |
| ETH | 69.1% |
| BTC | 51.0% |
| SOL | 94.3% |

Estimated return correlations:

| | ETH | BTC | SOL |
| --- | ---: | ---: | ---: |
| ETH | 1.000 | 0.847 | 0.755 |
| BTC | 0.847 | 1.000 | 0.753 |
| SOL | 0.755 | 0.753 | 1.000 |

The calibrated correlation matrix was also checked for positive semidefiniteness before being used in simulation.

### 6. Tail-Risk Evaluation

Mean exposure alone can hide severe but less frequent outcomes.

RiskForge therefore evaluates:

- Median exposure
- P(Exposure > 25%)
- P(Exposure > 50%)
- P(Exposure > 75%)
- Maximum simulated exposure
- 50th, 75th, 90th, 95th, and 99th percentile exposure

This makes the shape and tail of the simulated risk distribution visible.

### 7. Parameter Sensitivity

Risk estimates depend on model and protocol assumptions.

RiskForge therefore performs counterfactual liquidation-threshold sensitivity analysis under a fixed market shock.

This is intended as sensitivity analysis, not parameter optimization or a recommendation for protocol settings.

## Validation

The project includes automated tests covering:

- Position-level risk calculations
- Price-shock behavior
- Zero-shock invariance
- Monotonic liquidation exposure under increasingly severe shocks
- Asset-specific shock isolation
- Monte Carlo output dimensions
- Simulation reproducibility
- Horizon-dependent dispersion
- Exposure bounds
- Historical return construction
- Calibration output
- Correlation symmetry
- Unit correlation diagonal
- Non-negative volatility

Current test suite:

**17 tests passing**

## Dashboard

The Streamlit dashboard provides interactive access to:

- Protocol stress-test ladders
- Interactive asset-specific shocks
- Liquidation exposure by collateral asset
- Monte Carlo simulations
- Normal vs Student-t return assumptions
- Historical vs assumed calibration
- Tail-risk metrics
- Liquidation-threshold sensitivity

## Project Structure

```text
riskforge/
├── app.py
├── README.md
├── requirements.txt
├── pytest.ini
├── assets/
├── data/
│   └── calibration_snapshot.csv
├── scripts/
│   └── save_calibration.py
├── src/
│   ├── __init__.py
│   ├── calibration.py
│   ├── data_generator.py
│   ├── risk_engine.py
│   ├── simulation.py
│   └── stress_engine.py
└── tests/
    ├── test_calibration.py
    ├── test_risk_engine.py
    ├── test_simulation.py
    └── test_stress_engine.py