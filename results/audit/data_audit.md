# M3 data audit

## Dataset and grain

Yahoo Finance daily observations for the locked 20-stock universe, stored at one row per date and ticker. `adjusted_close` is Yahoo's split- and dividend-adjusted field; dividend and split event columns are retained as adjustment evidence.

## Coverage

- First common date: 2014-09-02
- Last common date: 2025-12-31
- Common dates: 2851
- Exact universe returned: True
- Duplicate date-ticker rows: 0
- Positive adjusted closes on common calendar: True
- Required 2025-12-31 terminal target price available: True
- Complete 60-day volatility history for every retained stock-date: True

## Missing adjusted closes before common-calendar filtering

Missingness is measured against the union of dates returned for any locked ticker.

| Ticker | Missing dates |
|---|---:|
| AAPL | 0 |
| ABBV | 0 |
| AMZN | 0 |
| BAC | 0 |
| BRK-B | 0 |
| COST | 0 |
| GOOGL | 0 |
| HD | 0 |
| JNJ | 0 |
| JPM | 0 |
| KO | 0 |
| MA | 0 |
| META | 0 |
| MSFT | 0 |
| NVDA | 0 |
| PG | 0 |
| TSLA | 0 |
| UNH | 0 |
| V | 0 |
| XOM | 0 |

## Retained sample after the within-partition five-day target rule

| Partition | Decision dates | Stock rows |
|---|---:|---:|
| Train | 1494 | 29880 |
| Validation | 487 | 9740 |
| Test | 741 | 14820 |

- Purge 1 retained decisions: 0
- Purge 2 retained decisions: 0

## Yahoo and corporate-action checks

- Yahoo symbol for Berkshire Hathaway Class B: `BRK-B`.
- BRK-B rows: 2851; coverage 2014-09-02 through 2025-12-31.
- BRK-B missing adjusted closes before common filtering: 0.
- Adjustment evidence, including counts of dividend events, split events, and rows where adjusted and unadjusted close differ, is preserved in `data_audit.json`.

## Findings

- No protocol-blocking issue detected.

This audit does not fit a model, tune regularisation, or calculate validation/test IC.
