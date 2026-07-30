from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yfinance as yf
from sklearn.linear_model import Ridge

from data_audit import RAW_END_INCLUSIVE, RAW_START, UNIVERSE
from preparation import (
    EVALUATION_HORIZON,
    TARGET_HORIZONS,
    build_feature_panel,
    build_targets,
    terminal_evaluation_target,
)


FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_10d",
    "log_return_20d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
]
RIDGE_ALPHA = 1.0


def get_reproducibility_metadata(root: Path) -> dict:
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.resolve(), text=True).strip()
    except Exception as e:
        git_sha = f"unknown: {e}"

    return {
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "yfinance_version": yf.__version__,
        "git_commit_sha": git_sha,
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ticker_universe": UNIVERSE,
        "data_range": f"{RAW_START} to {RAW_END_INCLUSIVE}",
        "alpha": RIDGE_ALPHA,
    }


@dataclass(frozen=True)
class PartitionData:
    features: pd.DataFrame
    targets: pd.DataFrame
    terminal_target: pd.DataFrame
    decision_dates: pd.DatetimeIndex


def load_partition_data(
    prices: pd.DataFrame,
    retained: pd.DataFrame,
    partition_name: str,
) -> PartitionData:
    partition_retained = retained[retained["partition"] == partition_name].copy()
    decision_dates = pd.DatetimeIndex(partition_retained["decision_date"].unique()).sort_values()
    features = build_feature_panel(prices, decision_dates, UNIVERSE, standardization_stocks=UNIVERSE)
    targets = build_targets(prices, decision_dates, UNIVERSE)
    terminal_target = terminal_evaluation_target(targets)
    return PartitionData(
        features=features,
        targets=targets,
        terminal_target=terminal_target,
        decision_dates=decision_dates,
    )


def compute_daily_ic(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    prediction_col: str = "prediction",
    actual_col: str = "evaluation_target_z",
) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = predictions.merge(actuals, on=["decision_date", "ticker"], how="inner")
    
    daily_records = []
    excluded_count = 0
    valid_count = 0

    for decision_date, group in merged.groupby("decision_date"):
        preds = group[prediction_col].to_numpy()
        acts = group[actual_col].to_numpy()

        if len(preds) < 2 or np.isclose(np.std(preds, ddof=0), 0.0) or np.isclose(np.std(acts, ddof=0), 0.0):
            excluded_count += 1
            continue

        valid_count += 1
        r = float(np.corrcoef(preds, acts)[0, 1])
        daily_records.append({"decision_date": decision_date, "pearson_ic": r})

    daily_df = pd.DataFrame.from_records(daily_records)
    counts = {
        "total_dates": int(merged["decision_date"].nunique()),
        "valid_dates": valid_count,
        "excluded_dates": excluded_count,
    }
    return daily_df, counts


def train_ridge_models(
    train_data: PartitionData,
    alpha: float = RIDGE_ALPHA,
) -> dict[int, Ridge]:
    models: dict[int, Ridge] = {}
    
    for h in TARGET_HORIZONS:
        target_h = train_data.targets[train_data.targets["horizon"] == h].set_index(["decision_date", "ticker"])["target_z"]
        aligned = train_data.features.join(target_h, how="inner")
        
        X_train = aligned[FEATURE_COLUMNS].to_numpy()
        y_train = aligned["target_z"].to_numpy()
        
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train)
        models[h] = model

    return models


def predict_partition(
    model: Ridge,
    partition_data: PartitionData,
) -> pd.DataFrame:
    X = partition_data.features[FEATURE_COLUMNS].to_numpy()
    preds = model.predict(X)
    df = partition_data.features.reset_index()[["decision_date", "ticker"]].copy()
    df["prediction"] = preds
    return df


def select_non_terminal_horizon(
    val_metrics: dict[int, dict],
) -> tuple[int, bool]:
    candidate_horizons = [h for h in TARGET_HORIZONS if h != EVALUATION_HORIZON]
    
    sorted_candidates = sorted(
        candidate_horizons,
        key=lambda h: (-round(val_metrics[h]["mean_ic"], 6), h),
    )
    
    selected = sorted_candidates[0]
    
    top_mean_rounded = round(val_metrics[selected]["mean_ic"], 6)
    ties = [h for h in candidate_horizons if round(val_metrics[h]["mean_ic"], 6) == top_mean_rounded]
    is_tie_broken = len(ties) > 1

    return selected, is_tie_broken


def run_experiment(root: Path) -> dict:
    prices_path = root / "data" / "processed" / "common_adjusted_close.csv"
    retained_path = root / "data" / "processed" / "retained_decision_dates.csv"
    
    if not prices_path.exists() or not retained_path.exists():
        raise FileNotFoundError("Processed datasets missing. Run data_audit.py first.")

    prices = pd.read_csv(prices_path, index_col="date", parse_dates=True)
    retained = pd.read_csv(retained_path, parse_dates=["decision_date", "target_5d_date"])

    train_data = load_partition_data(prices, retained, "Train")
    val_data = load_partition_data(prices, retained, "Validation")
    test_data = load_partition_data(prices, retained, "Test")

    models = train_ridge_models(train_data, alpha=RIDGE_ALPHA)

    val_metrics: dict[int, dict] = {}
    val_daily_ics: dict[int, pd.DataFrame] = {}

    for h in TARGET_HORIZONS:
        val_preds = predict_partition(models[h], val_data)
        daily_ic, counts = compute_daily_ic(val_preds, val_data.terminal_target)
        mean_ic = float(daily_ic["pearson_ic"].mean()) if not daily_ic.empty else 0.0
        std_ic = float(daily_ic["pearson_ic"].std()) if not daily_ic.empty else 0.0
        val_metrics[h] = {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "counts": counts,
        }
        val_daily_ics[h] = daily_ic

    selected_h, tie_broken = select_non_terminal_horizon(val_metrics)

    selected_test_preds = predict_partition(models[selected_h], test_data)
    selected_test_ic_df, test_counts_selected = compute_daily_ic(selected_test_preds, test_data.terminal_target)
    test_mean_ic_selected = float(selected_test_ic_df["pearson_ic"].mean()) if not selected_test_ic_df.empty else 0.0

    terminal_test_preds = predict_partition(models[EVALUATION_HORIZON], test_data)
    terminal_test_ic_df, test_counts_terminal = compute_daily_ic(terminal_test_preds, test_data.terminal_target)
    test_mean_ic_terminal = float(terminal_test_ic_df["pearson_ic"].mean()) if not terminal_test_ic_df.empty else 0.0

    ic_diff = test_mean_ic_selected - test_mean_ic_terminal
    if selected_test_ic_df.empty or terminal_test_ic_df.empty:
        outcome = "Inconclusive"
    elif test_mean_ic_selected > test_mean_ic_terminal:
        outcome = "Supported"
    else:
        outcome = "Unsupported"

    metadata = get_reproducibility_metadata(root)

    return {
        "metadata": metadata,
        "alpha": RIDGE_ALPHA,
        "train_rows": len(train_data.features),
        "train_decision_dates": len(train_data.decision_dates),
        "validation_metrics": val_metrics,
        "selected_horizon": selected_h,
        "tie_broken": tie_broken,
        "test_metrics": {
            "selected_h": selected_h,
            "selected_mean_ic": test_mean_ic_selected,
            "selected_counts": test_counts_selected,
            "terminal_mean_ic": test_mean_ic_terminal,
            "terminal_counts": test_counts_terminal,
            "ic_difference": ic_diff,
            "replication_outcome": outcome,
        },
    }


def generate_validation_markdown(results: dict) -> str:
    val_m = results["validation_metrics"]
    sel_h = results["selected_horizon"]
    tie = results["tie_broken"]
    meta = results["metadata"]
    
    rows = []
    for h in TARGET_HORIZONS:
        m = val_m[h]
        tag = " (Selected non-terminal)" if h == sel_h else (" (Terminal baseline)" if h == 5 else "")
        rows.append(
            f"| h = {h}{tag} | {m['mean_ic']:.6f} | {m['std_ic']:.6f} | {m['counts']['total_dates']} | {m['counts']['valid_dates']} | {m['counts']['excluded_dates']} |"
        )
    table_str = "\n".join(rows)

    tie_note = " (Tie-break applied: selected shorter horizon)." if tie else ""

    candidate_str = r"\{1, 2, 3, 4\}"
    delta_str = r"$\delta = 5$"
    in_str = r"\in"
    universe_str = ", ".join(meta["ticker_universe"])

    return f"""# Ridge Validation Model Selection Report

## Overview & Protocol Safeguards

- Model family: Ridge Regression (`alpha={results['alpha']}`)
- Training partition: Train decisions (2015-01-05 through 2020-12-15; {results['train_decision_dates']} dates / {results['train_rows']} stock-date rows)
- Validation partition: Validation decisions (2021-01-05 through 2022-12-15)
- Fixed Evaluation Target: 5-trading-day forward log return ({delta_str})
- Primary Metric: Mean daily cross-sectional Pearson IC
- Test Partition Status: Untouched during validation and model selection

## Validation Results (Terminal Evaluation Target {delta_str})

| Training Horizon | Mean Daily IC | Std Daily IC | Total Dates | Valid Dates | Excluded Dates |
|---|---:|---:|---:|---:|---:|
{table_str}

## Model Selection Decision

- **Candidate Non-Terminal Horizons Evaluated**: $h {in_str} {candidate_str}$
- **Selected Non-Terminal Horizon ($h^*$)**: **$h = {sel_h}$** with Validation Mean Daily IC = **{val_m[sel_h]['mean_ic']:.6f}**{tie_note}
- **Terminal Baseline Horizon ($h = 5$)**: Validation Mean Daily IC = **{val_m[5]['mean_ic']:.6f}**
- **Validation Delta ($h^* - h=5$)**: **{val_m[sel_h]['mean_ic'] - val_m[5]['mean_ic']:.6f}**

The selected non-terminal model ($h = {sel_h}$) is locked for one-shot evaluation on the Test partition.

## Reproducibility Metadata

- Python Version: `{meta['python_version']}`
- pandas Version: `{meta['pandas_version']}`
- numpy Version: `{meta['numpy_version']}`
- scikit-learn Version: `{meta['sklearn_version']}`
- yfinance Version: `{meta['yfinance_version']}`
- Execution Timestamp (UTC): `{meta['execution_timestamp']}`
- Git Commit SHA: `{meta['git_commit_sha']}`
- Ticker Universe: `{universe_str}`
- Data Range: `{meta['data_range']}`
- Ridge Alpha: `{meta['alpha']}`
"""


def generate_test_markdown(results: dict) -> str:
    tm = results["test_metrics"]
    sel_h = tm["selected_h"]
    outcome = tm["replication_outcome"]
    meta = results["metadata"]
    delta_str = r"$\delta = 5$"
    le_str = r"\le"
    universe_str = ", ".join(meta["ticker_universe"])

    return f"""# Ridge Test Replication Report

## Protocol Lock & Isolation Confirmation

- Selected Non-Terminal Horizon ($h^*$): locked from Validation as **$h = {sel_h}$**
- Terminal Baseline Horizon: **$h = 5$**
- Models evaluated: Exactly two already-fitted Ridge models (`alpha={results['alpha']}` trained on Train partition only)
- Test Partition: Test decisions (2023-01-04 through 2025-12-23)
- Fixed Evaluation Target: 5-trading-day forward log return ({delta_str})
- One-shot evaluation: No model refitting, hyperparameter tuning, or post-hoc alterations were performed.

## Test Results ({delta_str} Evaluation Target)

| Model | Mean Daily IC | Total Dates | Valid Dates | Excluded Dates |
|---|---|---:|---:|---:|
| Selected Non-Terminal ($h = {sel_h}$) | {tm['selected_mean_ic']:.6f} | {tm['selected_counts']['total_dates']} | {tm['selected_counts']['valid_dates']} | {tm['selected_counts']['excluded_dates']} |
| Terminal Baseline ($h = 5$) | {tm['terminal_mean_ic']:.6f} | {tm['terminal_counts']['total_dates']} | {tm['terminal_counts']['valid_dates']} | {tm['terminal_counts']['excluded_dates']} |
| **Difference ($h^* - h=5$)** | **{tm['ic_difference']:.6f}** | - | - | - |

## Primary Replication Outcome

# Primary Outcome: **{outcome}**

- **Classification Criteria**:
  - **Supported**: Test Mean IC of $h^*$ > Test Mean IC of $h = 5$.
  - **Unsupported**: Test Mean IC of $h^*$ {le_str} Test Mean IC of $h = 5$.
  - **Inconclusive**: Test comparison cannot be computed.
- **Finding**: The selected non-terminal label horizon model ($h = {sel_h}$) achieved a Test mean daily IC of **{tm['selected_mean_ic']:.6f}** versus **{tm['terminal_mean_ic']:.6f}** for the terminal-label model ($h = 5$), resulting in a difference of **{tm['ic_difference']:.6f}**.

## Reproducibility Metadata

- Python Version: `{meta['python_version']}`
- pandas Version: `{meta['pandas_version']}`
- numpy Version: `{meta['numpy_version']}`
- scikit-learn Version: `{meta['sklearn_version']}`
- yfinance Version: `{meta['yfinance_version']}`
- Execution Timestamp (UTC): `{meta['execution_timestamp']}`
- Git Commit SHA: `{meta['git_commit_sha']}`
- Ticker Universe: `{universe_str}`
- Data Range: `{meta['data_range']}`
- Ridge Alpha: `{meta['alpha']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    results = run_experiment(root)

    val_dir = root / "results" / "validation"
    test_dir = root / "results" / "test"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    (val_dir / "ridge_validation_results.json").write_text(json.dumps(results["validation_metrics"], indent=2), encoding="utf-8")
    (val_dir / "ridge_validation_results.md").write_text(generate_validation_markdown(results), encoding="utf-8")

    (test_dir / "ridge_test_results.json").write_text(json.dumps(results["test_metrics"], indent=2), encoding="utf-8")
    (test_dir / "ridge_test_results.md").write_text(generate_test_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
