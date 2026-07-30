from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


UNIVERSE = [
    "AAPL", "ABBV", "AMZN", "BAC", "BRK-B", "COST", "GOOGL", "HD", "JNJ", "JPM",
    "KO", "MA", "META", "MSFT", "NVDA", "PG", "TSLA", "UNH", "V", "XOM",
]
RAW_START = "2014-09-01"
RAW_END_INCLUSIVE = "2025-12-31"
YAHOO_END_EXCLUSIVE = "2026-01-01"

PARTITIONS = {
    "Train": ("2015-01-05", "2020-12-15"),
    "Validation": ("2021-01-05", "2022-12-15"),
    "Test": ("2023-01-04", "2025-12-23"),
}
PURGES = {
    "Purge 1": ("2020-12-16", "2021-01-04"),
    "Purge 2": ("2022-12-16", "2023-01-03"),
}


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw_csv(self) -> Path:
        return self.root / "data" / "raw" / "yahoo_daily_prices.csv"

    @property
    def common_csv(self) -> Path:
        return self.root / "data" / "processed" / "common_adjusted_close.csv"

    @property
    def retained_csv(self) -> Path:
        return self.root / "data" / "processed" / "retained_decision_dates.csv"

    @property
    def audit_json(self) -> Path:
        return self.root / "results" / "audit" / "data_audit.json"

    @property
    def audit_md(self) -> Path:
        return self.root / "results" / "audit" / "data_audit.md"


def download_yahoo() -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for ticker in UNIVERSE:
        frame = yf.Ticker(ticker).history(
            start=RAW_START,
            end=YAHOO_END_EXCLUSIVE,
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=False,
        )
        if frame.empty:
            raise RuntimeError(f"Yahoo Finance returned no rows for {ticker}")
        frame = frame.reset_index()
        date_column = "Date" if "Date" in frame.columns else frame.columns[0]
        frame["date"] = pd.to_datetime(frame[date_column], utc=True).dt.tz_convert(None).dt.normalize()
        frame["ticker"] = ticker
        for column in ["Close", "Adj Close", "Dividends", "Stock Splits"]:
            if column not in frame:
                frame[column] = 0.0 if column in ["Dividends", "Stock Splits"] else np.nan
        frames.append(
            frame[["date", "ticker", "Close", "Adj Close", "Dividends", "Stock Splits"]].rename(
                columns={
                    "Close": "close",
                    "Adj Close": "adjusted_close",
                    "Dividends": "dividends",
                    "Stock Splits": "stock_splits",
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def common_price_matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.DatetimeIndex]:
    matrix = raw.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    matrix = matrix.reindex(columns=UNIVERSE)
    union_dates = matrix.index
    missing = matrix.isna().sum().astype(int).to_dict()
    common = matrix.dropna(how="any")
    return common, missing, union_dates


def retained_decision_dates(common_dates: pd.DatetimeIndex, horizon: int = 5) -> pd.DataFrame:
    date_set = set(common_dates)
    records = []
    for name, (start, end) in PARTITIONS.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        candidates = common_dates[(common_dates >= start_ts) & (common_dates <= end_ts)]
        for date in candidates:
            position = common_dates.get_loc(date)
            if position + horizon >= len(common_dates):
                continue
            target_date = common_dates[position + horizon]
            if target_date <= end_ts and target_date in date_set:
                records.append({"partition": name, "decision_date": date, "target_5d_date": target_date})
    return pd.DataFrame.from_records(records, columns=["partition", "decision_date", "target_5d_date"])


def feature_availability(common: pd.DataFrame, retained: pd.DataFrame) -> tuple[bool, int]:
    one_day_log_return = np.log(common / common.shift(1))
    vol_60 = one_day_log_return.rolling(60, min_periods=60).std(ddof=1)
    dates = pd.DatetimeIndex(retained["decision_date"])
    unavailable = int(vol_60.reindex(dates).isna().sum().sum())
    return unavailable == 0, unavailable


def audit(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    expected = set(UNIVERSE)
    observed = set(raw["ticker"].unique())
    duplicates = int(raw.duplicated(["date", "ticker"]).sum())
    common, missing, union_dates = common_price_matrix(raw)
    retained = retained_decision_dates(common.index)
    vol_ok, vol_missing = feature_availability(common, retained)

    retained_dates = set(pd.to_datetime(retained["decision_date"]))
    purge_counts = {}
    for name, (start, end) in PURGES.items():
        purge_counts[name] = sum(pd.Timestamp(start) <= date <= pd.Timestamp(end) for date in retained_dates)

    partition_counts = {}
    for name, (start, end) in PARTITIONS.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        candidates = common.index[(common.index >= start_ts) & (common.index <= end_ts)]
        decision_count = int((retained["partition"] == name).sum())
        partition_counts[name] = {
            "first_common_decision_date": candidates.min().date().isoformat(),
            "last_common_decision_date": candidates.max().date().isoformat(),
            "common_candidate_dates": int(len(candidates)),
            "decision_dates": decision_count,
            "stock_rows": decision_count * len(UNIVERSE),
            "excluded_by_complete_target_within_partition_rule": int(len(candidates) - decision_count),
        }

    action_summary = {}
    for ticker in UNIVERSE:
        part = raw[raw["ticker"] == ticker]
        action_summary[ticker] = {
            "dividend_events": int((part["dividends"].fillna(0) != 0).sum()),
            "split_events": int((part["stock_splits"].fillna(0) != 0).sum()),
            "adjusted_differs_from_close_rows": int(
                (~np.isclose(part["adjusted_close"], part["close"], equal_nan=True)).sum()
            ),
        }

    requested_start = pd.Timestamp(RAW_START)
    requested_end = pd.Timestamp(RAW_END_INCLUSIVE)
    required_test_target = pd.Timestamp("2025-12-31")
    summary = {
        "source": "Yahoo Finance via yfinance",
        "raw_requested_range": {"start": RAW_START, "end_inclusive": RAW_END_INCLUSIVE},
        "universe": UNIVERSE,
        "universe_exact": observed == expected,
        "missing_tickers": sorted(expected - observed),
        "unexpected_tickers": sorted(observed - expected),
        "raw_rows": int(len(raw)),
        "duplicate_date_ticker_rows": duplicates,
        "union_first_date": union_dates.min().date().isoformat(),
        "union_last_date": union_dates.max().date().isoformat(),
        "first_common_date": common.index.min().date().isoformat(),
        "last_common_date": common.index.max().date().isoformat(),
        "common_date_count": int(len(common)),
        "missing_adjusted_close_by_stock_before_common_filter": missing,
        "common_prices_positive": bool((common > 0).all().all()),
        "requested_range_covered": bool(union_dates.min() <= requested_start + pd.Timedelta(days=2) and union_dates.max() == requested_end),
        "required_last_test_target_available": bool(required_test_target in common.index),
        "all_retained_decisions_have_common_5d_targets": bool(
            len(retained) > 0 and retained["target_5d_date"].isin(common.index).all()
        ),
        "partition_counts_after_complete_target_rule": partition_counts,
        "purge_retained_decision_counts": purge_counts,
        "all_purges_empty": all(count == 0 for count in purge_counts.values()),
        "volatility_60_available_for_every_retained_stock_row": vol_ok,
        "volatility_60_missing_stock_rows": vol_missing,
        "corporate_action_summary": action_summary,
        "brk_b": {
            "yahoo_symbol": "BRK-B",
            "raw_rows": int((raw["ticker"] == "BRK-B").sum()),
            "missing_adjusted_close_before_common_filter": int(missing["BRK-B"]),
            "first_date": raw.loc[raw["ticker"] == "BRK-B", "date"].min().date().isoformat(),
            "last_date": raw.loc[raw["ticker"] == "BRK-B", "date"].max().date().isoformat(),
        },
    }
    return summary, common, retained


def markdown_report(summary: dict) -> str:
    counts = summary["partition_counts_after_complete_target_rule"]
    missing_rows = "\n".join(
        f"| {ticker} | {count} |" for ticker, count in summary["missing_adjusted_close_by_stock_before_common_filter"].items()
    )
    count_rows = "\n".join(
        f"| {name} | {value['decision_dates']} | {value['stock_rows']} |" for name, value in counts.items()
    )
    brk = summary["brk_b"]
    issues = []
    if not summary["universe_exact"]:
        issues.append("The returned ticker set did not match the locked universe.")
    if summary["duplicate_date_ticker_rows"]:
        issues.append(f"Found {summary['duplicate_date_ticker_rows']} duplicate date-ticker rows.")
    if not summary["required_last_test_target_available"]:
        issues.append("The final required 2025-12-31 target price is unavailable on the common calendar.")
    if not summary["volatility_60_available_for_every_retained_stock_row"]:
        issues.append("Some retained rows lack a complete 60-day volatility window.")
    issue_text = "\n".join(f"- {item}" for item in issues) if issues else "- No protocol-blocking issue detected."
    return f"""# M3 data audit

## Dataset and grain

Yahoo Finance daily observations for the locked 20-stock universe, stored at one row per date and ticker. `adjusted_close` is Yahoo's split- and dividend-adjusted field; dividend and split event columns are retained as adjustment evidence.

## Coverage

- First common date: {summary['first_common_date']}
- Last common date: {summary['last_common_date']}
- Common dates: {summary['common_date_count']}
- Exact universe returned: {summary['universe_exact']}
- Duplicate date-ticker rows: {summary['duplicate_date_ticker_rows']}
- Positive adjusted closes on common calendar: {summary['common_prices_positive']}
- Required 2025-12-31 terminal target price available: {summary['required_last_test_target_available']}
- Complete 60-day volatility history for every retained stock-date: {summary['volatility_60_available_for_every_retained_stock_row']}

## Missing adjusted closes before common-calendar filtering

Missingness is measured against the union of dates returned for any locked ticker.

| Ticker | Missing dates |
|---|---:|
{missing_rows}

## Retained sample after the within-partition five-day target rule

| Partition | Decision dates | Stock rows |
|---|---:|---:|
{count_rows}

- Purge 1 retained decisions: {summary['purge_retained_decision_counts']['Purge 1']}
- Purge 2 retained decisions: {summary['purge_retained_decision_counts']['Purge 2']}

## Yahoo and corporate-action checks

- Yahoo symbol for Berkshire Hathaway Class B: `{brk['yahoo_symbol']}`.
- BRK-B rows: {brk['raw_rows']}; coverage {brk['first_date']} through {brk['last_date']}.
- BRK-B missing adjusted closes before common filtering: {brk['missing_adjusted_close_before_common_filter']}.
- Adjustment evidence, including counts of dividend events, split events, and rows where adjusted and unadjusted close differ, is preserved in `data_audit.json`.

## Findings

{issue_text}

This audit does not fit a model, tune regularisation, or calculate validation/test IC.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--use-existing-raw", action="store_true")
    args = parser.parse_args()
    paths = Paths(args.root.resolve())
    for directory in [paths.raw_csv.parent, paths.common_csv.parent, paths.audit_json.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.use_existing_raw:
        raw = pd.read_csv(paths.raw_csv, parse_dates=["date"])
    else:
        raw = download_yahoo()
        raw.to_csv(paths.raw_csv, index=False, date_format="%Y-%m-%d")
    summary, common, retained = audit(raw)
    common.to_csv(paths.common_csv, date_format="%Y-%m-%d")
    retained.to_csv(paths.retained_csv, index=False, date_format="%Y-%m-%d")
    paths.audit_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.audit_md.write_text(markdown_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
