# Ridge Validation Model Selection Report

## Overview & Protocol Safeguards

- Model family: Ridge Regression (`alpha=1.0`)
- Training partition: Train decisions (2015-01-05 through 2020-12-15; 1494 dates / 29880 stock-date rows)
- Validation partition: Validation decisions (2021-01-05 through 2022-12-15)
- Fixed Evaluation Target: 5-trading-day forward log return ($\delta = 5$)
- Primary Metric: Mean daily cross-sectional Pearson IC
- Test Partition Status: Untouched during validation and model selection

## Validation Results (Terminal Evaluation Target $\delta = 5$)

| Training Horizon | Mean Daily IC | Std Daily IC | Total Dates | Valid Dates | Excluded Dates |
|---|---:|---:|---:|---:|---:|
| h = 1 (Selected non-terminal) | -0.031555 | 0.429820 | 487 | 487 | 0 |
| h = 2 | -0.033697 | 0.425279 | 487 | 487 | 0 |
| h = 3 | -0.038839 | 0.426166 | 487 | 487 | 0 |
| h = 4 | -0.040428 | 0.431620 | 487 | 487 | 0 |
| h = 5 (Terminal baseline) | -0.039677 | 0.438365 | 487 | 487 | 0 |

## Model Selection Decision

- **Candidate Non-Terminal Horizons Evaluated**: $h \in \{1, 2, 3, 4\}$
- **Selected Non-Terminal Horizon ($h^*$)**: **$h = 1$** with Validation Mean Daily IC = **-0.031555**
- **Terminal Baseline Horizon ($h = 5$)**: Validation Mean Daily IC = **-0.039677**
- **Validation Delta ($h^* - h=5$)**: **0.008121**

The selected non-terminal model ($h = 1$) is locked for one-shot evaluation on the Test partition.

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
