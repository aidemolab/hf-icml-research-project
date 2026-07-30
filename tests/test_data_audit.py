import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_audit import PARTITIONS, PURGES, UNIVERSE, common_price_matrix, retained_decision_dates


class DataAuditTests(unittest.TestCase):
    def test_common_date_universe_filtering(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="D")
        rows = []
        for date in dates:
            for ticker in UNIVERSE:
                value = 100.0
                if date == dates[1] and ticker == "BRK-B":
                    value = np.nan
                rows.append({"date": date, "ticker": ticker, "adjusted_close": value})
        common, missing, _ = common_price_matrix(pd.DataFrame(rows))
        self.assertEqual(list(common.index), [dates[0], dates[2]])
        self.assertEqual(missing["BRK-B"], 1)
        self.assertEqual(list(common.columns), UNIVERSE)

    def test_forward_target_alignment(self):
        dates = pd.bdate_range("2023-01-04", periods=12)
        retained = retained_decision_dates(dates, horizon=5)
        first = retained.iloc[0]
        self.assertEqual(first["decision_date"], dates[0])
        self.assertEqual(first["target_5d_date"], dates[5])

    def test_partition_boundary_exclusion(self):
        dates = pd.bdate_range("2020-12-01", "2021-01-15")
        retained = retained_decision_dates(dates, horizon=5)
        train_end = pd.Timestamp(PARTITIONS["Train"][1])
        train = retained[retained["partition"] == "Train"]
        self.assertTrue((train["target_5d_date"] <= train_end).all())
        self.assertNotIn(pd.Timestamp("2020-12-15"), set(train["decision_date"]))

    def test_purge_date_exclusion(self):
        dates = pd.bdate_range("2020-12-01", "2023-01-20")
        retained = retained_decision_dates(dates, horizon=5)
        retained_dates = set(retained["decision_date"])
        for start, end in PURGES.values():
            for date in pd.bdate_range(start, end):
                self.assertNotIn(date, retained_dates)


if __name__ == "__main__":
    unittest.main()
