import importlib.util
import os
import unittest
from unittest.mock import patch

from live_data_feeder import LiveDataFeeder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_PATH = os.path.join(ROOT, "MACRO ENGINE.py")
ENGINE_SPEC = importlib.util.spec_from_file_location("macro_engine_mod", ENGINE_PATH)
ENGINE_MODULE = importlib.util.module_from_spec(ENGINE_SPEC)
ENGINE_SPEC.loader.exec_module(ENGINE_MODULE)


class LiveDataFeederFallbackTests(unittest.TestCase):
    def test_fetch_macro_inputs_falls_back_to_local_csv_without_fred_key(self):
        # Patch env so the .env-loaded FRED_API_KEY is hidden, forcing CSV fallback
        with patch.dict(os.environ, {}, clear=True):
            feeder = LiveDataFeeder(fred_api_key=None)
            feeder.fred = None  # Ensure FRED client is disabled

            current_macro, historical_df = feeder.fetch_macro_inputs()

        self.assertTrue(current_macro)
        # Engine-style names should be present via alias mapping
        self.assertIn("Core_CPI_YoY", current_macro)
        self.assertIn("Federal_Funds_Effective_Rate", current_macro)
        self.assertIn("ISM_Mfg_PMI", current_macro)
        self.assertFalse(historical_df.empty)
        self.assertIn("Core_CPI_YoY", historical_df.columns)
        self.assertIn("Federal_Funds_Effective_Rate", historical_df.columns)

    def test_no_duplicate_date_columns_in_historical_df(self):
        feeder = LiveDataFeeder(fred_api_key=None)
        _, historical_df = feeder.fetch_macro_inputs()
        date_cols = [c for c in historical_df.columns if c.startswith("date")]
        self.assertEqual(date_cols, [], f"Found leftover date columns: {date_cols}")

    def test_regime_bucket_scores_within_bounds(self):
        feeder = LiveDataFeeder(fred_api_key=None)
        current_macro, historical_df = feeder.fetch_macro_inputs()

        engine = ENGINE_MODULE.MacroEngine()
        regime, classification = engine.calculate_regime_scores(current_macro, historical_df)

        # All bucket scores should be bounded by winsorized z-scores
        for bucket, value in regime["Bucket_Scores"].items():
            self.assertLessEqual(abs(value), 3.0, f"Bucket '{bucket}' out of bounds: {value}")

        # Classification should be one of the four valid regimes
        valid_regimes = {"Reflationary Boom", "Disinflationary Bust", "Inflationary Boom", "Stagflation"}
        self.assertIn(classification, valid_regimes)

    def test_csv_write_does_not_accumulate_date_columns(self):
        """Write outputs twice and verify no duplicate date columns appear."""
        feeder = LiveDataFeeder(fred_api_key=None)
        current_macro, historical_df = feeder.fetch_macro_inputs()
        commodity_signals = {"Oil_Signal": 0.0, "BDI_Signal": 0.0, "CuGold_Signal": 0.0}
        technical_data = {t: {"Price_vs_200D": 0.0, "RSI": 50.0} for t in
                          ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU", "EEM", "IWM"]}

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the current CSV to a temp location
            historical_df.to_csv(os.path.join(tmpdir, "macro_historical_data.csv"))

            # Write twice
            ENGINE_MODULE.write_live_outputs(tmpdir, current_macro, historical_df, commodity_signals, technical_data)
            # Reload what was written
            reloaded = pd.read_csv(os.path.join(tmpdir, "macro_historical_data.csv"), index_col=0)
            ENGINE_MODULE.write_live_outputs(tmpdir, current_macro, reloaded, commodity_signals, technical_data)
            # Reload again
            reloaded2 = pd.read_csv(os.path.join(tmpdir, "macro_historical_data.csv"), index_col=0)

            date_cols = [c for c in reloaded2.columns if c.startswith("date")]
            self.assertEqual(date_cols, [], f"Date columns accumulated: {date_cols}")


import pandas as pd

if __name__ == "__main__":
    unittest.main()
