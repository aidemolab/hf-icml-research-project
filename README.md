# label-horizon-reproduction

A controlled, low-cost proxy reproduction of the label-horizon claim in *The Label Horizon Paradox*, using daily US equity data and Ridge regression.

## Disclaimer

This study is a **controlled proxy reproduction** designed for a CPU-friendly, low-cost experimental setup, **not a full replication** of the original paper's entire empirical universe or production architecture.

## Project Status

- **Milestones 1 to 6**: **Completed**.
  - **M1 (Scope & Claim Locking)**: Controlled proxy reproduction scope established.
  - **M2 (Protocol Specification)**: Locked experimental protocol recorded in [`configs/protocol.md`](configs/protocol.md).
  - **M3 (Data Audit)**: Market data audited and documented in [`results/audit/data_audit.md`](results/audit/data_audit.md).
  - **M4 (Structural Smoke Test)**: Data preparation math verified in [`results/smoke/structural_smoke_test.md`](results/smoke/structural_smoke_test.md).
  - **M5 (Model Training & Evaluation)**: Single controlled primary run completed.
    - Candidate non-terminal horizon $h^* = 1$ selected on Validation partition.
    - One-shot evaluation on Test partition: $h^* = 1$ achieved mean daily cross-sectional Pearson IC of **0.092776** versus **0.085270** for terminal baseline $h = 5$ ($\Delta = +0.007506$).
    - Primary Replication Outcome: **Supported under the locked proxy protocol**.
  - **M6 (Replication Decision Report)**: Replication decision report finalized in [`results/replication_decision_report.md`](results/replication_decision_report.md).
    - **Note**: No post-hoc robustness check has yet been executed.

## Key Documentation & Reports

- Protocol specification: [`configs/protocol.md`](configs/protocol.md)
- Data audit report: [`results/audit/data_audit.md`](results/audit/data_audit.md)
- Validation model selection report: [`results/validation/ridge_validation_results.md`](results/validation/ridge_validation_results.md)
- Test replication report: [`results/test/ridge_test_results.md`](results/test/ridge_test_results.md)
- Replication decision report: [`results/replication_decision_report.md`](results/replication_decision_report.md)

Downloaded market data files (`data/raw/` and `data/processed/`) are intentionally excluded from version control. All scripts regenerate their authorized outputs locally.
