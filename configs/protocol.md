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
- One ridge-regression model per training horizon ($h \in \{1, 2, 3, 4, 5\}$); every model is trained strictly on the Train partition and evaluated against the same terminal 5-day return ($\delta = 5$).
- Ridge regularization parameter: fixed at `alpha = 1.0` across all five training horizons.
- Hyperparameter rules: No alpha tuning, grid search, or cross-validation is permitted in the primary run.
- Primary metric: mean daily cross-sectional Pearson IC.
- Partition usage:
  - **Train**: Model fitting only.
  - **Validation**: Model selection of the optimal non-terminal horizon $h^* \in \{1, 2, 3, 4\}$ against the terminal 5-day evaluation target.
  - **Test**: One-shot evaluation of the selected non-terminal model ($h^*$) versus the terminal baseline ($h = 5$). Untouched during model development.
- No model work is authorized during M3.

## Partitions

- Train decisions: 2015-01-05 through 2020-12-15.
- Purge 1: 2020-12-16 through 2021-01-04.
- Validation decisions: 2021-01-05 through 2022-12-15.
- Purge 2: 2022-12-16 through 2023-01-03.
- Test decisions: 2023-01-04 through 2025-12-23.

A decision date is retained only when its complete 5-common-trading-day forward target remains inside the same partition. Purge dates are never retained.

## Model Selection & Test Evaluation Protocol

1. **Non-terminal horizon selection rule**:
   - On the Validation partition, select the non-terminal horizon $h^* \in \{1, 2, 3, 4\}$ that achieves the highest mean daily cross-sectional Pearson IC against the 5-day terminal target.
   - Deterministic tie-break rule: If two non-terminal horizons have equal mean IC at reported precision (6 decimal places), select the shorter horizon (smaller $h$).
   - The $h = 5$ model is the fixed terminal-label baseline and is excluded from the non-terminal candidate selection pool.

2. **Final controlled Test evaluation**:
   - Train all five models using the Train partition only.
   - Use Validation only to select the single non-terminal horizon $h^*$.
   - Once selection is locked, evaluate exactly two already-fitted models once on the untouched Test partition:
     a. The selected non-terminal-horizon model ($h^*$).
     b. The terminal-label baseline model ($h = 5$).
   - Both models are evaluated against the same fixed 5-day terminal target.
   - Models must not be refitted, retuned, or altered after viewing Validation results.
   - Test results must not be used to alter any modeling or selection decisions.

3. **Primary replication outcome categories**:
   - **Supported**: The selected non-terminal model ($h^*$) achieves a strictly higher mean daily cross-sectional Pearson IC than the $h = 5$ baseline on the Test partition.
   - **Unsupported**: The selected non-terminal model ($h^*$) does not achieve a higher Test mean IC than the $h = 5$ baseline.
   - **Inconclusive**: The final comparison cannot be computed reliably due to documented data, variance, or execution failure.

4. **Metric safeguards**:
   - Pearson IC is calculated daily across the cross-section of stocks.
   - Dates where Pearson IC is undefined (e.g. constant predictions or constant target returns with zero standard deviation) are excluded from the daily mean IC calculation.
   - All validation and test outputs must explicitly report the count of total decision dates, valid decision dates, and excluded decision dates.

5. **Reporting structure**:
   - Validation selection results must be saved separately in [`results/validation/ridge_validation_results.md`](results/validation/ridge_validation_results.md).
   - Final Test replication evaluation results must be written separately in [`results/test/ridge_test_results.md`](results/test/ridge_test_results.md).
   - Reports must clearly distinguish intermediate model selection on Validation from the final one-shot replication outcome on Test.

## Protocol Amendments (Pre-Modeling)

- **Alpha Specification Note**: The Ridge regularization parameter $\alpha$ was previously unspecified in the initial protocol draft. Prior to executing any model fitting, training, or evaluation, $\alpha$ was locked to $1.0$ for all five training horizons ($h = 1, 2, 3, 4, 5$) to preserve a simple, low-cost, and controlled proxy experiment without introducing hyperparameter tuning or additional researcher degrees of freedom.
- **Selection & Test Procedure Note**: Detailed rules for non-terminal model selection, deterministic tie-breaking, one-shot Test evaluation, primary outcome classification, metric safeguards for zero-variance dates, and separate reporting were explicitly locked prior to writing or executing any modeling code.
