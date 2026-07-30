# label-horizon-reproduction
A controlled, low-cost proxy reproduction of the label-horizon claim in The Label Horizon Paradox, using daily US equity data and ridge regression.

## Current scope

This repository is staged for a controlled, CPU-friendly proxy reproduction of the paper's label-horizon claim. M1-M4 cover claim locking, protocol locking, data auditing, and a model-free structural smoke test. No model has been fitted and no validation or test performance has been inspected.

The authoritative protocol record is [`configs/protocol.md`](configs/protocol.md). Data-audit findings are documented in [`results/audit/data_audit.md`](results/audit/data_audit.md). The structural preparation smoke test is documented in [`results/smoke/structural_smoke_test.md`](results/smoke/structural_smoke_test.md).

Downloaded market data and the paper PDF are intentionally excluded from version control. The audit and smoke-test scripts regenerate their authorized outputs locally.
