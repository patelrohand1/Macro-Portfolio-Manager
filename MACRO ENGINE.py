import os
import numpy as np
import pandas as pd

from live_data_feeder import LiveDataFeeder, apply_aliases_to_dict, apply_aliases_to_df


class MacroEngine:
    def __init__(self):
        self.tickers = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU", "EEM", "IWM"]

        self.neutral_weights = {
            "XLC": 0.09046, "XLY": 0.08479, "XLP": 0.04311, "XLE": 0.02925,
            "XLF": 0.10621, "XLV": 0.07885, "XLI": 0.07732, "XLK": 0.33681,
            "XLB": 0.01692, "XLRE": 0.01701, "XLU": 0.01926, "EEM": 0.05000,
            "IWM": 0.05000
        }

        self.sensitivity_matrix = pd.DataFrame({
            "XLC":  [1.5,  1.5,  1.5,   0.0,  0.0,  0.0],
            "XLY":  [3.0,  1.5,  1.5,   0.0,  0.0,  0.0],
            "XLP":  [-1.5, -0.75, 0.0,  -0.5,  0.0, -0.5],
            "XLE":  [2.0, -3.0,  0.75,  3.0,  0.5,  0.0],
            "XLF":  [2.5, -1.5,  3.0,   0.0,  0.0,  0.0],
            "XLV":  [-0.75, 0.75, 0.0,   0.0,  0.0,  0.0],
            "XLI":  [3.0,  1.5,  1.5,   1.0,  1.0,  0.5],
            "XLK":  [1.5,  3.0,  0.75,  0.0,  0.0,  0.0],
            "XLB":  [2.0, -2.0,  0.75,  0.0,  0.5,  0.5],
            "XLRE": [0.0,  3.0,  1.5,   0.0, -0.5, -0.5],
            "XLU":  [-2.0,  2.0,  0.0,   0.0, -0.5, -0.75],
            "EEM":  [2.0,  2.0,  3.0,   0.5,  0.5,  0.5],
            "IWM":  [2.0,  1.5,  2.0,  -0.5,  0.0,  0.0]
        }, index=["Real Economy", "Monetary Pulse", "Financial Cond", "Oil", "BDI", "CuGold"])
        self.conviction_flag = 0.75
        self.scalar = 0.0075

        # Indicators that flow through the z-score regime pipeline.
        # Commodity and FX metrics are excluded here — they flow directly
        # through build_macro_themes via Commodity_Signals.
        self.indicator_specs = [
            {"metric": "Real_GDP_Growth_Q", "bucket": "growth", "direction": "higher"},
            {"metric": "ISM_Mfg_PMI", "bucket": "growth", "direction": "higher"},
            {"metric": "ISM_Services_PMI", "bucket": "growth", "direction": "higher"},
            {"metric": "Industrial_Production_YoY", "bucket": "growth", "direction": "higher"},
            {"metric": "Retail_Sales_YoY", "bucket": "growth", "direction": "higher"},
            {"metric": "ISM_New_Orders_minus_Inventories", "bucket": "growth", "direction": "higher"},
            {"metric": "S_P_Global_Services_PMI", "bucket": "growth", "direction": "higher"},
            {"metric": "ISM_Services_New_Orders", "bucket": "growth", "direction": "higher"},
            {"metric": "Avg_Weekly_Hours_Manufacturing", "bucket": "growth", "direction": "higher"},
            {"metric": "Retail_Sales_Control_Group_MoM", "bucket": "growth", "direction": "higher"},
            {"metric": "Redbook_Same_Store_Sales_YoY", "bucket": "growth", "direction": "higher"},
            {"metric": "Core_Capital_Goods_Orders_3MMA_MoM_Change", "bucket": "growth", "direction": "higher"},
            {"metric": "New_Home_Sales_3M_3M_Change", "bucket": "growth", "direction": "higher"},
            {"metric": "Building_Permits_3MMA_MoM_Change", "bucket": "growth", "direction": "higher"},
            {"metric": "Cardboard_Containerboard_Production_3MMA_MoM_Change", "bucket": "growth", "direction": "higher"},
            {"metric": "BEDZ_Hotel_ETF_52W_High_Pct", "bucket": "growth", "direction": "higher"},
            {"metric": "Unemployment_Rate", "bucket": "labor", "direction": "lower"},
            {"metric": "Nonfarm_Payrolls_3M_Avg", "bucket": "labor", "direction": "higher"},
            {"metric": "Initial_Jobless_Claims_4Wk_Avg", "bucket": "labor", "direction": "lower"},
            {"metric": "Prime_Age_LFPR", "bucket": "labor", "direction": "higher"},
            {"metric": "Continuing_Claims_4Wk_Avg", "bucket": "labor", "direction": "lower"},
            {"metric": "Job_Openings_Hires_Ratio", "bucket": "labor", "direction": "lower"},
            {"metric": "Quits_Rate", "bucket": "labor", "direction": "higher"},
            {"metric": "Hires_Rate_JOLTS", "bucket": "labor", "direction": "higher"},
            {"metric": "Temporary_Help_Services_Employment_YoY", "bucket": "labor", "direction": "higher"},
            {"metric": "Core_CPI_YoY", "bucket": "inflation", "direction": "lower"},
            {"metric": "Core_PCE_YoY", "bucket": "inflation", "direction": "lower"},
            {"metric": "Headline_CPI_YoY", "bucket": "inflation", "direction": "lower"},
            {"metric": "Headline_PCE_YoY", "bucket": "inflation", "direction": "lower"},
            {"metric": "Trimmed_Mean_PCE", "bucket": "inflation", "direction": "lower"},
            {"metric": "Median_CPI", "bucket": "inflation", "direction": "lower"},
            {"metric": "Sticky_CPI", "bucket": "inflation", "direction": "lower"},
            {"metric": "Supercore_PCE", "bucket": "inflation", "direction": "lower"},
            {"metric": "PPI_Final_Demand_YoY", "bucket": "inflation", "direction": "lower"},
            {"metric": "Five_Year_Breakeven_Inflation", "bucket": "inflation", "direction": "lower"},
            {"metric": "ISM_Prices_Paid", "bucket": "inflation", "direction": "lower"},
            {"metric": "ISM_Services_Prices_Paid", "bucket": "inflation", "direction": "lower"},
            {"metric": "NFIB_Price_Plans", "bucket": "inflation", "direction": "lower"},
            {"metric": "PPI_Services_Intermediate_Demand_3M_Annualized", "bucket": "inflation", "direction": "lower"},
            {"metric": "PPI_Processed_Goods_Intermediate_Demand_3M_Annualized", "bucket": "inflation", "direction": "lower"},
            {"metric": "FiveY5Y_Forward_Inflation_Expectation", "bucket": "inflation", "direction": "lower"},
            {"metric": "Michigan_1Y_Inflation_Expectations", "bucket": "inflation", "direction": "lower"},
            {"metric": "Import_Price_Index_ex_Fuel_YoY", "bucket": "inflation", "direction": "lower"},
            {"metric": "Zillow_Observed_Rent_Index_YoY", "bucket": "inflation", "direction": "lower"},
            {"metric": "Federal_Funds_Effective_Rate", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "Real_Fed_Funds_Rate", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "FOMC_Score", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "FF_Futures_3M_Implied_Rate", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "FF_Futures_12M_Implied_Rate", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "Market_Implied_Cuts_Hikes_12M", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "FF_Futures_Curve_Slope_12M_minus_3M", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "Real_Fed_Funds_Rate_vs_5Y5Y_Fwd_Inflation", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "TwoY_Treasury_minus_Fed_Funds", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "OneY_minus_ThreeM_Treasury", "bucket": "fed_policy", "direction": "lower"},
            {"metric": "TwoY_Treasury_Yield", "bucket": "rates", "direction": "higher"},
            {"metric": "TwoY_Treasury_Yield_3M_Change", "bucket": "rates", "direction": "lower"},
            {"metric": "TenY_Treasury_Yield", "bucket": "rates", "direction": "higher"},
            {"metric": "TenY_Treasury_Yield_3M_Change", "bucket": "rates", "direction": "lower"},
            {"metric": "Two_s_Ten_s_Curve", "bucket": "rates", "direction": "positive"},
            {"metric": "TenY_minus_ThreeM_Treasury", "bucket": "rates", "direction": "positive"},
            {"metric": "TenY_Real_Yield", "bucket": "rates", "direction": "lower"},
            {"metric": "TenY_Real_Yield_3M_Change", "bucket": "rates", "direction": "lower"},
            {"metric": "TenY_Term_Premium", "bucket": "rates", "direction": "lower"},
            {"metric": "Mortgage_Rates", "bucket": "rates", "direction": "lower"},
            {"metric": "Mortgage_Rates_3M_Change", "bucket": "rates", "direction": "lower"},
            {"metric": "HY_Credit_Spreads", "bucket": "credit", "direction": "lower"},
            {"metric": "IG_Credit_Spreads", "bucket": "credit", "direction": "lower"},
            {"metric": "IG_Credit_Spreads_3M_Change", "bucket": "credit", "direction": "lower"},
            {"metric": "HY_Credit_Spreads_3M_Change", "bucket": "credit", "direction": "lower"},
            {"metric": "SLOOS_C_and_I_Lending_Standards", "bucket": "credit", "direction": "lower"},
            {"metric": "SLOOS_C_and_I_Loan_Demand", "bucket": "credit", "direction": "higher"},
            {"metric": "SLOOS_Consumer_Card_Lending_Standards", "bucket": "credit", "direction": "lower"},
            {"metric": "SLOOS_Consumer_Loan_Demand", "bucket": "credit", "direction": "higher"},
            {"metric": "HY_Issuance_3M_Trend_Pct", "bucket": "credit", "direction": "higher"},
            {"metric": "KBW_Bank_Index_vs_SPX_3M_Relative_Pct", "bucket": "credit", "direction": "higher"},
        ]

    def calculate_z_score(self, current_val, historical_series):
        """Calculates a strictly bounded Z-score to prevent math explosions."""
        
        # 1. Calculate Mean and Standard Deviation
        mean_val = historical_series.mean()
        std_val = historical_series.std()
        
        # 2. Safety check: If standard deviation is 0 (or missing), return 0 to avoid dividing by zero
        if pd.isna(std_val) or std_val == 0:
            return 0.0
            
        # 3. Calculate raw mathematical Z-score
        raw_z = (current_val - mean_val) / std_val
        
        # 4. WINSORIZATION: Cap the maximum and minimum values at +/- 3.0
        # This forces the engine to behave like the bounded Excel spreadsheet
        bounded_z = max(-3.0, min(3.0, raw_z))
        
        return bounded_z

    def transform_indicator_score(self, spec, current_val, historical_series):
        z_score = self.calculate_z_score(current_val, historical_series)
        direction = spec["direction"]
        if direction in {"lower", "negative", "loosening"}:
            return -z_score
        return z_score

    def calculate_regime_scores(self, current_data, historical_data):
        bucket_scores = {bucket: [] for bucket in ["growth", "labor", "inflation", "fed_policy", "credit", "rates"]}

        for spec in self.indicator_specs:
            metric = spec["metric"]
            bucket = spec["bucket"]
            if bucket not in bucket_scores:
                continue
            if metric not in current_data or metric not in historical_data.columns:
                continue
            current_val = current_data[metric]
            if pd.isna(current_val):
                continue
            score = self.transform_indicator_score(spec, current_val, historical_data[metric])
            bucket_scores[bucket].append(score)

        aggregate = {}
        for bucket, scores in bucket_scores.items():
            aggregate[bucket] = sum(scores) / len(scores) if scores else 0.0

        growth_score = (aggregate["growth"] + aggregate["labor"]) / 2.0
        inflation_score = aggregate["inflation"]
        financial_score = (aggregate["fed_policy"] + aggregate["credit"] + aggregate["rates"]) / 3.0

        if growth_score > 0 and inflation_score <= 0 and financial_score > 0:
            regime_class = "Reflationary Boom"
        elif growth_score <= 0 and inflation_score <= 0:
            regime_class = "Disinflationary Bust"
        elif growth_score > 0 and inflation_score > 0:
            regime_class = "Inflationary Boom"
        else:
            regime_class = "Stagflation"

        return {
            "Growth": growth_score,
            "Inflation": inflation_score,
            "Financial_Conditions": financial_score,
            "Bucket_Scores": aggregate,
        }, regime_class

    def build_macro_themes(self, regime_scores, commodity_data):
        return {
            "Real Economy": np.clip((regime_scores["Growth"] * 0.75) + 0.285, -3.0, 3.0),
            "Monetary Pulse": np.clip((regime_scores["Inflation"] * 1.2) - 0.024, -3.0, 3.0),
            "Financial Cond": np.clip((regime_scores["Financial_Conditions"] * 0.9) - 0.005, -3.0, 3.0),
            "Oil": commodity_data.get("Oil_Signal", 0.0),
            "BDI": commodity_data.get("BDI_Signal", 0.0),
            "CuGold": commodity_data.get("CuGold_Signal", 0.0)
        }

    def process_mrc(self, technical_inputs, raw_tilts):
        mrc_multipliers = {}
        for t in self.tickers:
            tech = technical_inputs[t]
            price_vs_200d = tech.get("Price_vs_200D")
            rsi = tech.get("RSI")
            price_vs_200d = float(price_vs_200d) if price_vs_200d is not None else 0.0
            rsi = float(rsi) if rsi is not None else 50.0
            tech_health = (price_vs_200d * 0.5) + (rsi - 50) * 0.1
            tilt = raw_tilts[t]

            if tilt > 0 and tech_health < -5:
                mult = 0.20
            elif tilt > 0 and tech_health > 5:
                mult = 1.75
            elif tilt < 0 and tech_health > 5:
                mult = 0.50
            else:
                mult = 1.00

            mrc_multipliers[t] = mult

        return mrc_multipliers

    def apply_risk_controls(self, final_weights, prev_weights):
        weights = pd.Series(final_weights)
        neutral = pd.Series(self.neutral_weights)

        weights = weights.clip(np.maximum(0.0, neutral - 0.15), neutral + 0.15)

        if weights["EEM"] > 0.12:
            weights["EEM"] = 0.12
        if weights["IWM"] > 0.12:
            weights["IWM"] = 0.12

        if weights["XLE"] + weights["XLB"] > 0.15:
            excess = (weights["XLE"] + weights["XLB"]) - 0.15
            weights["XLE"] -= (excess / 2)
            weights["XLB"] -= (excess / 2)

        defensive_sum = weights["XLP"] + weights["XLV"] + weights["XLU"]
        if defensive_sum > 0.30:
            excess = defensive_sum - 0.30
            weights["XLP"] -= excess * (weights["XLP"] / defensive_sum)
            weights["XLV"] -= excess * (weights["XLV"] / defensive_sum)
            weights["XLU"] -= excess * (weights["XLU"] / defensive_sum)

        weights = weights / weights.sum()

        turnover = sum(abs(weights[t] - prev_weights[t]) for t in self.tickers) / 2.0
        if turnover > 0.25:
            shrink_factor = 0.25 / turnover
            for t in self.tickers:
                delta = weights[t] - prev_weights[t]
                weights[t] = prev_weights[t] + (delta * shrink_factor)

        weights = weights / weights.sum()
        return weights.to_dict(), turnover

    def run_pipeline(self, current_data, historical_data, comm_data, tech_data, prev_weights):
        regime, classification = self.calculate_regime_scores(current_data, historical_data)
        themes = self.build_macro_themes(regime, comm_data)
        M = np.array([
            themes["Real Economy"], themes["Monetary Pulse"], themes["Financial Cond"],
            themes["Oil"], themes["BDI"], themes["CuGold"]
        ])

        raw_tilts = {}
        for t in self.tickers:
            raw_tilts[t] = np.dot(self.sensitivity_matrix[t].values, M) * self.scalar * self.conviction_flag

        mrc_mults = self.process_mrc(tech_data, raw_tilts)
        prelim_weights = {t: self.neutral_weights[t] + (raw_tilts[t] * mrc_mults[t]) for t in self.tickers}
        final_weights, turnover = self.apply_risk_controls(prelim_weights, prev_weights)

        results = []
        for t in self.tickers:
            results.append({
                "Ticker": t,
                "Neutral": self.neutral_weights[t],
                "Raw_Tilt": raw_tilts[t],
                "MRC_x": mrc_mults[t],
                "Final_Wt": final_weights[t]
            })

        df = pd.DataFrame(results).set_index("Ticker")
        
        return {
            "regime": regime,
            "classification": classification,
            "themes": themes,
            "raw_tilts": raw_tilts,
            "mrc_mults": mrc_mults,
            "final_weights": final_weights,
            "turnover": turnover,
            "results_df": df
        }

def print_pipeline_results(results):
    classification = results["classification"]
    turnover = results["turnover"]
    regime = results["regime"]
    df = results["results_df"].copy()

    print(f"\n>> Regime: {classification.upper()} | Turnover: {turnover*100:.2f}%\n")
    print("Bucket scores:")
    for bucket, value in regime["Bucket_Scores"].items():
        print(f"  - {bucket}: {value:.3f}")

    df["Neutral"] = (df["Neutral"] * 100).map("{:.2f}%".format)
    df["Raw_Tilt"] = (df["Raw_Tilt"] * 100).map("{:+.2f}%".format)
    df["Final_Wt"] = (df["Final_Wt"] * 100).map("{:.2f}%".format)
    print(df.to_string())



def write_live_outputs(base_dir, current_macro, historical_df, commodity_signals, technical_data):
    """Write engine state to CSV files for persistence and debugging."""
    # Current macro snapshot
    current_df = pd.DataFrame([
        {"metric": metric, "value": value}
        for metric, value in sorted(current_macro.items())
        if value is not None
    ])
    current_df.to_csv(os.path.join(base_dir, "macro_current_data.csv"), index=False)

    # Historical data — clean before writing to prevent column accumulation
    historical_out = historical_df.copy()
    # Drop any duplicate date.* columns that may have crept in
    date_cols = [c for c in historical_out.columns if c.startswith("date")]
    if date_cols:
        historical_out = historical_out.drop(columns=date_cols)
    # Ensure index name is set and write with index
    if historical_out.index.name is None:
        historical_out.index.name = "date"
    historical_out.to_csv(os.path.join(base_dir, "macro_historical_data.csv"))

    # Commodity signals
    commodity_df = pd.DataFrame([
        {"metric": metric, "value": value}
        for metric, value in sorted(commodity_signals.items())
        if value is not None
    ])
    commodity_df.to_csv(os.path.join(base_dir, "macro_commodity_data.csv"), index=False)

    # Technical MRC data
    technical_df = pd.DataFrame([
        {"ticker": ticker, "Price_vs_200D": data.get("Price_vs_200D"), "RSI": data.get("RSI")}
        for ticker, data in sorted(technical_data.items())
    ])
    technical_df.to_csv(os.path.join(base_dir, "macro_technical_data.csv"), index=False)


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Initialize the Feeder
    feeder = LiveDataFeeder() 
    
    # 2. Attempt to pull live data
    try:
        current_macro, historical_df = feeder.fetch_macro_inputs()
        market_data = feeder.fetch_market_data()
        technical_mrc = feeder.fetch_technical_inputs(["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU", "EEM", "IWM"])
        
        # Merge signals
        commodity_signals = market_data.get("Commodity_Signals", {})
        
        print("\n=== SUCCESS: LIVE DATA PULLED ===")
        print("\n1. current_macro = ")
        for k, v in sorted(current_macro.items()): print(f"   '{k}': {v}")
        print("\n2. historical_df sample = ")
        print(historical_df.tail().to_string())
        print("\n3. market_data = ")
        for k, v in sorted(market_data.items()): print(f"   '{k}': {v}")

    except Exception as exc:
        print(f"\nLive Feed Unavailable ({exc}). Falling back to static CSVs.")
        # Fallback to local files if API fails
        current_macro = {row["metric"]: float(row["value"]) for _, row in pd.read_csv(os.path.join(base_dir, "macro_current_data.csv")).iterrows()}
        historical_df = pd.read_csv(os.path.join(base_dir, "macro_historical_data.csv"), index_col=0)
        historical_df = historical_df.apply(pd.to_numeric, errors="coerce")
        # Drop any leftover date columns and apply aliases
        date_cols = [c for c in historical_df.columns if c.startswith("date")]
        if date_cols:
            historical_df = historical_df.drop(columns=date_cols)
        current_macro = apply_aliases_to_dict(current_macro)
        historical_df = apply_aliases_to_df(historical_df)
        commodity_signals = {row["metric"]: float(row["value"]) for _, row in pd.read_csv(os.path.join(base_dir, "macro_commodity_data.csv")).iterrows()}
        technical_mrc = {row["ticker"]: {"Price_vs_200D": row["Price_vs_200D"], "RSI": row["RSI"]} for _, row in pd.read_csv(os.path.join(base_dir, "macro_technical_data.csv")).iterrows()}

    # 3. Run the Engine with the ingested data
    engine = MacroEngine()
    prev_weights = engine.neutral_weights # Or load from your 'Portfolio.csv'
    
    # Execute the Pipeline
    results = engine.run_pipeline(current_macro, historical_df, commodity_signals, technical_mrc, prev_weights)
    print_pipeline_results(results)

    write_live_outputs(base_dir, current_macro, historical_df, commodity_signals, technical_mrc)
