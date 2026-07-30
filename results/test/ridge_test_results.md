# Ridge Test Replication Report

## Protocol Lock & Isolation Confirmation

- Selected Non-Terminal Horizon ($h^*$): locked from Validation as **$h = 1$**
- Terminal Baseline Horizon: **$h = 5$**
- Models evaluated: Exactly two already-fitted Ridge models (`alpha=1.0` trained on Train partition only)
- Test Partition: Test decisions (2023-01-04 through 2025-12-23)
- Fixed Evaluation Target: 5-trading-day forward log return ($\delta = 5$)
- One-shot evaluation: No model refitting, hyperparameter tuning, or post-hoc alterations were performed.

## Test Results ($\delta = 5$ Evaluation Target)

| Model | Mean Daily IC | Total Dates | Valid Dates | Excluded Dates |
|---|---|---:|---:|---:|
| Selected Non-Terminal ($h = 1$) | 0.092776 | 741 | 741 | 0 |
| Terminal Baseline ($h = 5$) | 0.085270 | 741 | 741 | 0 |
| **Difference ($h^* - h=5$)** | **0.007506** | - | - | - |

## Primary Replication Outcome

# Primary Outcome: **Supported**

- **Classification Criteria**:
  - **Supported**: Test Mean IC of $h^*$ > Test Mean IC of $h = 5$.
  - **Unsupported**: Test Mean IC of $h^*$ \le Test Mean IC of $h = 5$.
  - **Inconclusive**: Test comparison cannot be computed.
- **Finding**: The selected non-terminal label horizon model ($h = 1$) achieved a Test mean daily IC of **0.092776** versus **0.085270** for the terminal-label model ($h = 5$), resulting in a difference of **0.007506**.

## Reproducibility Metadata

- Python Version: `3.13.5`
- pandas Version: `2.2.3`
- numpy Version: `2.1.3`
- scikit-learn Version: `1.6.1`
- yfinance Version: `1.5.2`
- Execution Timestamp (UTC): `2026-07-30T23:37:34.177944+00:00`
- Git Commit SHA: `8ca53dce357e51a546aa49d9f71a58e2d942f6bc`
- Ticker Universe: `AAPL, ABBV, AMZN, BAC, BRK-B, COST, GOOGL, HD, JNJ, JPM, KO, MA, META, MSFT, NVDA, PG, TSLA, UNH, V, XOM`
- Data Range: `2014-09-01 to 2025-12-31`
- Ridge Alpha: `1.0`
