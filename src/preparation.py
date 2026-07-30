from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


RETURN_WINDOWS = (1, 5, 10, 20)
VOLATILITY_WINDOWS = (5, 10, 20, 60)
TARGET_HORIZONS = (1, 2, 3, 4, 5)
EVALUATION_HORIZON = 5


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    result.index = pd.DatetimeIndex(result.index).tz_localize(None).normalize()
    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError("Price calendar contains duplicate dates")
    if result.isna().any().any() or (result <= 0).any().any():
        raise ValueError("Prices must be complete and positive")
    return result.astype(float)


def cross_sectional_zscore(values: pd.DataFrame) -> pd.DataFrame:
    """Standardize across columns independently on each date using ddof=0."""
    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1, ddof=0)
    if (standard_deviations == 0).any():
        raise ValueError("Cannot standardize a date with zero cross-sectional dispersion")
    return values.sub(means, axis=0).div(standard_deviations, axis=0)


def build_raw_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build features whose latest input is the adjusted close at the row's date."""
    prices = _validate_prices(prices)
    one_day = np.log(prices / prices.shift(1))
    features: dict[str, pd.DataFrame] = {}
    for window in RETURN_WINDOWS:
        features[f"log_return_{window}d"] = np.log(prices / prices.shift(window))
    for window in VOLATILITY_WINDOWS:
        features[f"volatility_{window}d"] = one_day.rolling(window, min_periods=window).std(ddof=1)
    return features


def build_feature_panel(
    prices: pd.DataFrame,
    decision_dates: Iterable[pd.Timestamp],
    output_stocks: list[str],
    standardization_stocks: list[str] | None = None,
) -> pd.DataFrame:
    """Return the eight locked, date-wise standardized features in long form."""
    prices = _validate_prices(prices)
    standardization_stocks = standardization_stocks or list(prices.columns)
    dates = pd.DatetimeIndex(decision_dates)
    frames = []
    for feature_name, raw_values in build_raw_features(prices).items():
        standardized = cross_sectional_zscore(raw_values.loc[dates, standardization_stocks])
        selected = standardized[output_stocks].stack().rename(feature_name)
        selected.index.names = ["decision_date", "ticker"]
        frames.append(selected)
    panel = pd.concat(frames, axis=1).sort_index()
    if panel.isna().any().any():
        raise ValueError("Feature panel contains unavailable values")
    return panel


def build_targets(
    prices: pd.DataFrame,
    decision_dates: Iterable[pd.Timestamp],
    stocks: list[str],
) -> pd.DataFrame:
    """Build raw and within-date standardized targets for horizons 1 through 5."""
    prices = _validate_prices(prices)
    dates = pd.DatetimeIndex(decision_dates)
    records = []
    for decision_date in dates:
        position = prices.index.get_loc(decision_date)
        decision_prices = prices.loc[decision_date, stocks]
        for horizon in TARGET_HORIZONS:
            target_date = prices.index[position + horizon]
            raw = np.log(prices.loc[target_date, stocks] / decision_prices)
            mean = float(raw.mean())
            std = float(raw.std(ddof=0))
            if std == 0:
                raise ValueError(f"Zero target dispersion on {decision_date} at horizon {horizon}")
            standardized = (raw - mean) / std
            for ticker in stocks:
                records.append(
                    {
                        "decision_date": decision_date,
                        "ticker": ticker,
                        "horizon": horizon,
                        "target_date": target_date,
                        "target_raw": float(raw[ticker]),
                        "target_z": float(standardized[ticker]),
                    }
                )
    return pd.DataFrame.from_records(records)


def terminal_evaluation_target(targets: pd.DataFrame) -> pd.DataFrame:
    """Return the sole evaluation target, which is locked to delta=5."""
    terminal = targets.loc[targets["horizon"] == EVALUATION_HORIZON].copy()
    terminal = terminal.rename(
        columns={"target_date": "evaluation_target_date", "target_raw": "evaluation_target_raw", "target_z": "evaluation_target_z"}
    )
    terminal["evaluation_horizon"] = EVALUATION_HORIZON
    return terminal.drop(columns="horizon")
