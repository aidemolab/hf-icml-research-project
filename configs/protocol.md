# Locked proxy protocol

## Universe and source

- Universe: AAPL, ABBV, AMZN, BAC, BRK-B, COST, GOOGL, HD, JNJ, JPM, KO, MA, META, MSFT, NVDA, PG, TSLA, UNH, V, XOM.
- Source: Yahoo Finance daily prices.
- Raw range: 2014-09-01 through 2025-12-31 inclusive.
- Price field: Yahoo Finance `Adj Close`, which incorporates split and dividend adjustments.
- Calendar: dates on which all 20 stocks have a valid adjusted close.

## Locked future experiment

- Evaluation horizon: 5 common trading dates.
- Training horizons: 1, 2, 3, 4, and 5 common trading dates.
- Inputs: 1-, 5-, 10-, and 20-day log returns plus 5-, 10-, 20-, and 60-day rolling standard deviations of one-day log returns (`ddof=1`).
- One ridge-regression model per training horizon; every model is evaluated against the same terminal 5-day return.
- Primary metric: mean daily cross-sectional Pearson IC.
- No model work is authorized during M3.

## Partitions

- Train decisions: 2015-01-05 through 2020-12-15.
- Purge 1: 2020-12-16 through 2021-01-04.
- Validation decisions: 2021-01-05 through 2022-12-15.
- Purge 2: 2022-12-16 through 2023-01-03.
- Test decisions: 2023-01-04 through 2025-12-23.

A decision date is retained only when its complete 5-common-trading-day forward target remains inside the same partition. Purge dates are never retained.
