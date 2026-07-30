# Replication Decision Report: Label-Horizon Proxy Study

## 1. Primary Replication Outcome & Tested Research Claim

- **Primary Outcome Classification**: **Supported under the locked proxy protocol**.
- **Exact Research Claim Tested**: Under the locked proxy protocol, determine whether a Ridge model trained on a selected non-terminal label horizon $h \in \{1, 2, 3, 4\}$ achieves a higher mean daily cross-sectional Pearson IC against the fixed 5-day evaluation target than a Ridge model trained on the matching 5-day label.

---

## 2. Experimental Summary & Model Selection

### Validation Model Selection (Descriptive)
- **Partition**: Validation decisions (`2021-01-05` through `2022-12-15`, 487 decision dates).
- **Candidate Pool**: $h \in \{1, 2, 3, 4\}$.
- **Validation Results ($\delta = 5$ Evaluation Target)**:
  - $h = 1$: **-0.031555** *(Selected non-terminal $h^*$)*
  - $h = 2$: -0.033697
  - $h = 3$: -0.038839
  - $h = 4$: -0.040428
  - $h = 5$: -0.039677 *(Terminal baseline)*
- **Selection Outcome**: $h = 1$ was selected from $h \in \{1, 2, 3, 4\}$ using the Validation partition. The Validation comparison with $h = 5$ was descriptive; the confirmatory outcome was determined exclusively from the one-shot Test comparison.

### One-Shot Test Replication (Confirmatory)
- **Partition**: Test decisions (`2023-01-04` through `2025-12-23`, 741 decision dates).
- **Evaluated Models**: Exactly two already-fitted Ridge models ($h^* = 1$ and $h = 5$, trained on Train partition `2015-01-05` to `2020-12-15`).
- **Test Results ($\delta = 5$ Evaluation Target)**:
  - Selected Non-Terminal ($h^* = 1$): **0.092776**
  - Terminal Baseline ($h = 5$): **0.085270**
  - Difference ($h^* - h=5$): **+0.007506**

---

## 3. Three Levels of Evidence

1. **Supported Proxy Result**:
   - The locked proxy protocol produced a higher Test mean daily Pearson IC for $h = 1$ than for $h = 5$.
2. **Partial Reproduction Evidence**:
   - The result is directionally consistent with the paper's central label-horizon claim within this restricted proxy experiment.
3. **Not a Full Replication**:
   - The experiment differs from the original paper in universe, model class, feature set, scale, and other design details.

---

## 4. Mechanism Hypotheses

- Theoretical mechanisms proposed in the literature—such as shorter training labels acting as structural regularizers, noise filters, or reducers of label-overlap correlation—were not directly tested or established by this experiment.
- These mechanisms remain unverified hypotheses outside the empirical scope of this proxy study.

---

## 5. Magnitude of Test Result

- **Absolute Difference**: **+0.007506 IC points**.
- **Relative Difference**: Approximately **+8.80%** compared with the $h = 5$ Test IC.
- **Interpretation**: The observed improvement is positive but modest. Its economic or statistical significance was not evaluated in this proxy study.

---

## 6. Methodological & Practical Limitations

1. **Universe Size**: Restricted to a 20-stock universe.
2. **Universe Selection**: Fixed large-cap universe with potential survivorship or selection bias.
3. **Model Specification**: Single linear model class (Ridge regression).
4. **Hyperparameters**: Fixed $\alpha = 1.0$ without hyperparameter tuning.
5. **Data Source**: Dependence on Yahoo Finance daily adjusted closes.
6. **Feature Engineering**: Limited to eight technical return and volatility features.
7. **Evaluation Horizon**: Restricted to one evaluation horizon ($\delta = 5$).
8. **Partitioning**: Single temporal split structure.
9. **Candidate Pool**: Model selection conducted across four candidate non-terminal horizons.
10. **Scope**: Results reflect one specific proxy specification without statistical uncertainty or economic-value analysis.

---

## 7. Post-Hoc Robustness Recommendation

- A single exploratory temporal-stability analysis may be justified to examine whether the observed Test IC difference is concentrated in one sub-period. Because this analysis was proposed after the Test result was known, it must be labelled post hoc and cannot alter the primary Supported classification.
- **Status**: No robustness check has yet been executed.

---

## 8. Reproducibility Metadata

- Python Version: `3.13.5`
- pandas Version: `2.2.3`
- numpy Version: `2.1.3`
- scikit-learn Version: `1.6.1`
- yfinance Version: `1.5.2`
- Execution Timestamp (UTC): `2026-07-30T23:35:40Z`
- Implementation Commit SHA: `8ca53dce357e51a546aa49d9f71a58e2d942f6bc`
- Results Commit SHA: `a0b195bdd03683006657850acb9c1a1cf0bbdf5e`
- Ticker Universe: `AAPL, ABBV, AMZN, BAC, BRK-B, COST, GOOGL, HD, JNJ, JPM, KO, MA, META, MSFT, NVDA, PG, TSLA, UNH, V, XOM`
- Raw Data Range: `2014-09-01 to 2025-12-31`
- Ridge Alpha: `1.0`
