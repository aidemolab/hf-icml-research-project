from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_audit import PARTITIONS, PURGES, UNIVERSE
from preparation import (
    EVALUATION_HORIZON,
    TARGET_HORIZONS,
    build_feature_panel,
    build_targets,
    terminal_evaluation_target,
)


SMOKE_STOCKS = ["AAPL", "MSFT", "NVDA"]
SMOKE_DATE_COUNT = 8


def run_smoke(root: Path) -> tuple[pd.DataFrame, dict]:
    prices = pd.read_csv(root / "data" / "processed" / "common_adjusted_close.csv", index_col="date", parse_dates=True)
    retained = pd.read_csv(root / "data" / "processed" / "retained_decision_dates.csv", parse_dates=["decision_date", "target_5d_date"])
    smoke_dates = pd.DatetimeIndex(
        retained.loc[retained["partition"] == "Train", "decision_date"].iloc[:SMOKE_DATE_COUNT]
    )

    # Locked feature standardisation uses the complete 20-stock universe, while
    # the structural target check requested for M4 uses exactly the three smoke stocks.
    features = build_feature_panel(prices, smoke_dates, SMOKE_STOCKS, standardization_stocks=UNIVERSE)
    targets = build_targets(prices, smoke_dates, SMOKE_STOCKS)
    terminal = terminal_evaluation_target(targets)

    wide_targets = targets.pivot(index=["decision_date", "ticker"], columns="horizon", values=["target_date", "target_raw", "target_z"])
    wide_targets.columns = [f"{kind}_d{horizon}" for kind, horizon in wide_targets.columns]
    output = features.join(wide_targets).join(
        terminal.set_index(["decision_date", "ticker"])[["evaluation_horizon", "evaluation_target_date", "evaluation_target_z"]]
    ).reset_index()

    train_start, train_end = map(pd.Timestamp, PARTITIONS["Train"])
    purge_membership = {}
    for name, (start, end) in PURGES.items():
        purge_membership[name] = int(((output["decision_date"] >= start) & (output["decision_date"] <= end)).sum())

    target_grouped = targets.groupby(["decision_date", "horizon"])["target_z"]
    means = target_grouped.mean()
    variances = target_grouped.var(ddof=0)
    summary = {
        "smoke_stocks": SMOKE_STOCKS,
        "smoke_decision_dates": [date.date().isoformat() for date in smoke_dates],
        "decision_date_count": len(smoke_dates),
        "stock_date_rows": int(len(output)),
        "feature_names": list(features.columns),
        "feature_count": int(features.shape[1]),
        "target_horizons": list(TARGET_HORIZONS),
        "evaluation_horizon": EVALUATION_HORIZON,
        "all_dates_inside_train": bool(((smoke_dates >= train_start) & (smoke_dates <= train_end)).all()),
        "purge_row_counts": purge_membership,
        "validation_or_test_rows": 0,
        "all_feature_values_finite": bool(np.isfinite(features.to_numpy()).all()),
        "target_z_max_abs_group_mean": float(means.abs().max()),
        "target_z_max_abs_group_variance_error": float((variances - 1.0).abs().max()),
        "terminal_target_is_always_delta_5": bool((output["evaluation_horizon"] == 5).all()),
        "contains_model_outputs": False,
        "contains_performance_metrics": False,
    }
    return output, summary


def markdown(summary: dict) -> str:
    dates = ", ".join(summary["smoke_decision_dates"])
    return f"""# M4 structural smoke test

- Stocks: {', '.join(summary['smoke_stocks'])}
- Retained Train dates: {dates}
- Structural rows: {summary['stock_date_rows']}
- Features: {summary['feature_count']} ({', '.join(summary['feature_names'])})
- Training-label horizons constructed: {summary['target_horizons']}
- Evaluation horizon locked to: {summary['evaluation_horizon']}
- All dates inside Train: {summary['all_dates_inside_train']}
- Purge rows: {summary['purge_row_counts']}
- Validation/Test rows: {summary['validation_or_test_rows']}
- All feature values finite: {summary['all_feature_values_finite']}
- Maximum absolute target group mean after standardisation: {summary['target_z_max_abs_group_mean']:.3e}
- Maximum absolute target group variance error after standardisation: {summary['target_z_max_abs_group_variance_error']:.3e}

Features at date `t` use adjusted closes no later than `t`. Target `delta` uses the common-calendar close exactly `delta` positions after `t`. Target standardisation in this smoke artifact is performed independently for each date and horizon using only AAPL, MSFT, and NVDA, as required for M4. The eventual locked experiment must standardise across the complete 20-stock universe.

No models, coefficients, predictions, alpha selections, IC values, or comparative horizon results are present.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "results" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    output, summary = run_smoke(root)
    output.to_csv(output_dir / "structural_smoke_sample.csv", index=False, date_format="%Y-%m-%d")
    (output_dir / "structural_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "structural_smoke_test.md").write_text(markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
