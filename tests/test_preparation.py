import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_audit import PURGES, UNIVERSE
from preparation import build_feature_panel, build_targets, terminal_evaluation_target
from smoke_test import SMOKE_DATE_COUNT, SMOKE_STOCKS


class PreparationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = pd.read_csv(ROOT / "data" / "processed" / "common_adjusted_close.csv", index_col="date", parse_dates=True)
        retained = pd.read_csv(ROOT / "data" / "processed" / "retained_decision_dates.csv", parse_dates=["decision_date"])
        cls.smoke_dates = pd.DatetimeIndex(retained.loc[retained["partition"] == "Train", "decision_date"].iloc[:SMOKE_DATE_COUNT])

    def test_feature_availability_and_no_future_dependency(self):
        original = build_feature_panel(self.prices, self.smoke_dates, SMOKE_STOCKS, UNIVERSE)
        changed = self.prices.copy()
        changed.loc[changed.index > self.smoke_dates.max(), :] *= 7.0
        rebuilt = build_feature_panel(changed, self.smoke_dates, SMOKE_STOCKS, UNIVERSE)
        self.assertEqual(original.shape, (SMOKE_DATE_COUNT * len(SMOKE_STOCKS), 8))
        self.assertTrue(np.isfinite(original.to_numpy()).all())
        pd.testing.assert_frame_equal(original, rebuilt)

    def test_target_forward_date_alignment(self):
        targets = build_targets(self.prices, self.smoke_dates, SMOKE_STOCKS)
        for row in targets.itertuples(index=False):
            position = self.prices.index.get_loc(row.decision_date)
            self.assertEqual(row.target_date, self.prices.index[position + row.horizon])
            expected = np.log(self.prices.loc[row.target_date, row.ticker] / self.prices.loc[row.decision_date, row.ticker])
            self.assertAlmostEqual(row.target_raw, expected, places=14)

    def test_target_cross_sectional_standardisation(self):
        targets = build_targets(self.prices, self.smoke_dates, SMOKE_STOCKS)
        groups = targets.groupby(["decision_date", "horizon"])["target_z"]
        for _, group in groups:
            self.assertAlmostEqual(float(group.mean()), 0.0, places=12)
            self.assertAlmostEqual(float(group.var(ddof=0)), 1.0, places=12)

    def test_terminal_target_and_split_isolation(self):
        targets = build_targets(self.prices, self.smoke_dates, SMOKE_STOCKS)
        terminal = terminal_evaluation_target(targets)
        self.assertTrue((terminal["evaluation_horizon"] == 5).all())
        self.assertGreaterEqual(self.smoke_dates.min(), pd.Timestamp("2015-01-05"))
        self.assertLessEqual(self.smoke_dates.max(), pd.Timestamp("2020-12-15"))
        for start, end in PURGES.values():
            self.assertFalse(((self.smoke_dates >= start) & (self.smoke_dates <= end)).any())


if __name__ == "__main__":
    unittest.main()
