# M4 structural smoke test

- Stocks: AAPL, MSFT, NVDA
- Retained Train dates: 2015-01-05, 2015-01-06, 2015-01-07, 2015-01-08, 2015-01-09, 2015-01-12, 2015-01-13, 2015-01-14
- Structural rows: 24
- Features: 8 (log_return_1d, log_return_5d, log_return_10d, log_return_20d, volatility_5d, volatility_10d, volatility_20d, volatility_60d)
- Training-label horizons constructed: [1, 2, 3, 4, 5]
- Evaluation horizon locked to: 5
- All dates inside Train: True
- Purge rows: {'Purge 1': 0, 'Purge 2': 0}
- Validation/Test rows: 0
- All feature values finite: True
- Maximum absolute target group mean after standardisation: 1.554e-15
- Maximum absolute target group variance error after standardisation: 3.331e-16

Features at date `t` use adjusted closes no later than `t`. Target `delta` uses the common-calendar close exactly `delta` positions after `t`. Target standardisation in this smoke artifact is performed independently for each date and horizon using only AAPL, MSFT, and NVDA, as required for M4. The eventual locked experiment must standardise across the complete 20-stock universe.

No models, coefficients, predictions, alpha selections, IC values, or comparative horizon results are present.
