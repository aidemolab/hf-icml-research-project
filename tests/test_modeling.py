import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modeling import (
    compute_daily_ic,
    run_experiment,
    select_non_terminal_horizon,
    train_ridge_models,
)
from preparation import TARGET_HORIZONS


class ModelingPipelineTests(unittest.TestCase):
    def test_zero_variance_date_exclusion(self):
        dates = pd.date_range("2021-01-05", periods=3)
        tickers = ["AAPL", "MSFT", "NVDA"]
        records_pred = []
        records_act = []
        
        for i, date in enumerate(dates):
            for ticker in tickers:
                if i == 0:
                    pred = 0.05
                else:
                    pred = np.random.randn()
                act = np.random.randn()
                records_pred.append({"decision_date": date, "ticker": ticker, "prediction": pred})
                records_act.append({"decision_date": date, "ticker": ticker, "evaluation_target_z": act})

        preds_df = pd.DataFrame(records_pred)
        acts_df = pd.DataFrame(records_act)

        daily_df, counts = compute_daily_ic(preds_df, acts_df)
        self.assertEqual(counts["total_dates"], 3)
        self.assertEqual(counts["valid_dates"], 2)
        self.assertEqual(counts["excluded_dates"], 1)
        self.assertEqual(len(daily_df), 2)

    def test_tie_breaking_rule(self):
        val_metrics = {
            1: {"mean_ic": 0.0456123},
            2: {"mean_ic": 0.0456124},
            3: {"mean_ic": 0.0300000},
            4: {"mean_ic": 0.0100000},
            5: {"mean_ic": 0.0500000},
        }
        selected, tie_broken = select_non_terminal_horizon(val_metrics)
        self.assertEqual(selected, 1)
        self.assertTrue(tie_broken)

    def test_end_to_end_modeling_pipeline(self):
        results = run_experiment(ROOT)
        self.assertEqual(results["alpha"], 1.0)
        self.assertIn(results["selected_horizon"], [1, 2, 3, 4])
        self.assertIn(results["test_metrics"]["replication_outcome"], ["Supported", "Unsupported", "Inconclusive"])
        self.assertEqual(len(results["validation_metrics"]), 5)


if __name__ == "__main__":
    unittest.main()
