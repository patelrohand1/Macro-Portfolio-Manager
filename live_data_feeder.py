import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
import warnings
import datetime
warnings.filterwarnings('ignore')

# Load .env file so FRED_API_KEY is available via os.getenv
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass

# Canonical alias mapping: FRED/CSV short names → engine indicator_spec names.
# Both directions are populated so lookups work either way.
COLUMN_ALIASES = {
    "Real_GDP_Growth":    "Real_GDP_Growth_Q",
    "Mfg_PMI_Proxy":      "ISM_Mfg_PMI",
    "Ind_Prod_YoY":       "Industrial_Production_YoY",
    "NFP_3Mo_Avg":        "Nonfarm_Payrolls_3M_Avg",
    "Initial_Claims_4Wk": "Initial_Jobless_Claims_4Wk_Avg",
    "5Y_Breakeven":       "Five_Year_Breakeven_Inflation",
    "Fed_Funds_Rate":     "Federal_Funds_Effective_Rate",
    "2s10s_Curve":        "Two_s_Ten_s_Curve",
    "10Y_Real_Yield":     "TenY_Real_Yield",
    "10Y_minus_3Mo":      "TenY_minus_ThreeM_Treasury",
    "Mortgage_30Y":       "Mortgage_Rates",
    "HY_Credit_Spread":   "HY_Credit_Spreads",
    "IG_Credit_Spread":   "IG_Credit_Spreads",
    "SLOOS_Lending_Stds": "SLOOS_C_and_I_Lending_Standards",
}


def apply_aliases_to_dict(data_dict):
    """Add aliased keys to a dict so both FRED-style and engine-style names exist."""
    for src, dst in COLUMN_ALIASES.items():
        if src in data_dict and dst not in data_dict:
            data_dict[dst] = data_dict[src]
        elif dst in data_dict and src not in data_dict:
            data_dict[src] = data_dict[dst]
    return data_dict


def apply_aliases_to_df(df):
    """Add aliased columns to a DataFrame so both name styles exist."""
    for src, dst in COLUMN_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
        elif dst in df.columns and src not in df.columns:
            df[src] = df[dst]
    return df


class LiveDataFeeder:
    def __init__(self, fred_api_key=None):
        # Initialize FRED. Pass your API key when instantiating if pulling live data.
        if not fred_api_key:
            # Try Streamlit Secrets first (for deployment)
            try:
                import streamlit as st
                if "FRED_API_KEY" in st.secrets:
                    fred_api_key = st.secrets["FRED_API_KEY"]
            except Exception:
                pass
                
        fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None

    def _load_local_macro_data(self):
        """Load the bundled macro CSV snapshots when live FRED data is unavailable."""
        base_dir = os.path.dirname(os.path.abspath(__file__))

        current_df = pd.read_csv(os.path.join(base_dir, "macro_current_data.csv"))
        current_macro = {row["metric"]: float(row["value"]) for _, row in current_df.iterrows()}

        historical_df = pd.read_csv(os.path.join(base_dir, "macro_historical_data.csv"), index_col=0)
        historical_df = historical_df.apply(pd.to_numeric, errors="coerce")
        # Drop any leftover duplicate date columns
        date_cols = [c for c in historical_df.columns if c.startswith("date")]
        if date_cols:
            historical_df = historical_df.drop(columns=date_cols)

        # Ensure aliases exist in both the dict and DataFrame
        current_macro = apply_aliases_to_dict(current_macro)
        historical_df = apply_aliases_to_df(historical_df)

        return current_macro, historical_df
        
    def scale_signal(self, raw_pct):
        """Scales raw 3-month percentages into the -3.0 to +3.0 model bounds."""
        if pd.isna(raw_pct): 
            return 0.0
        return max(-3.0, min(3.0, raw_pct / 10.0))

    def _download_ticker_change(self, ticker, target_date=None, period="4mo"):
        """Download a single ticker and compute 3-month % change safely."""
        try:
            if target_date:
                end_date = target_date + datetime.timedelta(days=1)
                start_date = target_date - datetime.timedelta(days=120)
                df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            else:
                df = yf.download(ticker, period=period, interval="1d", progress=False)
                
            if df.empty or len(df) < 2:
                return None, None

            # Flatten MultiIndex columns if present (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df["Close"].dropna()
            if close.empty:
                return None, None

            latest = float(close.iloc[-1])
            # Use the earliest available row (up to ~63 trading days back)
            lookback_idx = max(0, len(close) - 63)
            past = float(close.iloc[lookback_idx])

            if past == 0:
                return latest, None
            change_pct = ((latest / past) - 1) * 100
            return latest, change_pct
        except Exception:
            return None, None

    def fetch_market_data(self, target_date=None):
        """Pulls Commodities, FX, and Proxies via Yahoo Finance and returns scaled signals."""
        print("Fetching Yahoo Finance Market Data...")

        # Download each ticker individually for robustness
        wti_price, wti_chg = self._download_ticker_change("CL=F", target_date)
        copper_price, copper_chg = self._download_ticker_change("HG=F", target_date)
        gold_price, gold_chg = self._download_ticker_change("GC=F", target_date)
        bdry_price, bdry_chg = self._download_ticker_change("BDRY", target_date)
        dxy_price, dxy_chg = self._download_ticker_change("DX-Y.NYB", target_date)
        cny_price, cny_chg = self._download_ticker_change("CNY=X", target_date)
        vix_price, _ = self._download_ticker_change("^VIX", target_date)

        # Copper/Gold ratio change
        cugold_raw_pct = None
        if copper_price and gold_price and gold_price != 0:
            try:
                # Get historical ratio as well
                if target_date:
                    end_date = target_date + datetime.timedelta(days=1)
                    start_date = target_date - datetime.timedelta(days=120)
                    df_cu = yf.download("HG=F", start=start_date, end=end_date, interval="1d", progress=False)
                    df_au = yf.download("GC=F", start=start_date, end=end_date, interval="1d", progress=False)
                else:
                    df_cu = yf.download("HG=F", period="4mo", interval="1d", progress=False)
                    df_au = yf.download("GC=F", period="4mo", interval="1d", progress=False)
                if isinstance(df_cu.columns, pd.MultiIndex):
                    df_cu.columns = df_cu.columns.get_level_values(0)
                if isinstance(df_au.columns, pd.MultiIndex):
                    df_au.columns = df_au.columns.get_level_values(0)
                cu_close = df_cu["Close"].dropna()
                au_close = df_au["Close"].dropna()
                if len(cu_close) >= 2 and len(au_close) >= 2:
                    current_ratio = float(cu_close.iloc[-1]) / float(au_close.iloc[-1])
                    lb = max(0, min(len(cu_close), len(au_close)) - 63)
                    past_ratio = float(cu_close.iloc[lb]) / float(au_close.iloc[lb])
                    if past_ratio != 0:
                        cugold_raw_pct = ((current_ratio / past_ratio) - 1) * 100
            except Exception:
                pass

        # Broad commodity proxy: average of available changes
        commodity_changes = [c for c in [wti_chg, copper_chg, gold_chg] if c is not None and not pd.isna(c)]
        broad_commodity = float(pd.Series(commodity_changes).mean()) if commodity_changes else None

        def safe_float(v):
            return float(v) if v is not None and not pd.isna(v) else None

        market_data = {
            "WTI_Crude_3Mo_%": safe_float(wti_chg),
            "WTI_Crude_3M_Change_Pct": safe_float(wti_chg),
            "Copper_3Mo_%": safe_float(copper_chg),
            "Copper_3M_Change_Pct": safe_float(copper_chg),
            "Gold_3Mo_%": safe_float(gold_chg),
            "Gold_3M_Change_Pct": safe_float(gold_chg),
            "Broad_Commodity_Index_3M_Change_Pct": safe_float(broad_commodity),
            "Baltic_Dry_Proxy_3Mo_%": safe_float(bdry_chg),
            "Baltic_Dry_Index_3M_Change_Pct": safe_float(bdry_chg),
            "Copper_Gold_Ratio_3M_Relative_Change_Pct": safe_float(cugold_raw_pct),
            "DXY_3Mo_%": safe_float(dxy_chg),
            "DXY_3M_Change_Pct": safe_float(dxy_chg),
            "USD_CNY_Level": safe_float(cny_price),
            "USD_CNY": safe_float(cny_price),
            "USD_CNY_3Mo_%": safe_float(cny_chg),
            "USD_CNY_3M_Change_Pct": safe_float(cny_chg),
            "EM_FX_Index_3M_Change_Pct": safe_float(-cny_chg) if cny_chg is not None else None,
            "VIX_Level": safe_float(vix_price),
            "Commodity_Signals": {
                "Oil_Signal": self.scale_signal(wti_chg),
                "BDI_Signal": self.scale_signal(bdry_chg),
                "CuGold_Signal": self.scale_signal(cugold_raw_pct)
            }
        }
        
        return market_data

    def fetch_macro_inputs(self):
        """
        Fetches live FRED data and historical FRED data, 
        and strictly transforms historical levels into YoY% and 3Mo Averages 
        so they perfectly match the current macro inputs.
        """
        print("Fetching FRED Macroeconomic Data...")
        if not self.fred:
            print("FRED API key not provided; using bundled macro CSV fallback data.")
            return self._load_local_macro_data()

        # 1. Define the raw series we need to pull
        series_map = {
            "Real_GDP_Growth": "A191RL1Q225SBEA", 
            "Ind_Prod_YoY": "INDPRO",
            "Retail_Sales_YoY": "RSAFS",
            "Mfg_PMI_Proxy": "GACDFSA066MSFRBPHI", 
            "Unemployment_Rate": "UNRATE",
            "NFP_3Mo_Avg": "PAYEMS",
            "Initial_Claims_4Wk": "IC4WSA",
            "Prime_Age_LFPR": "LNS11300060",
            "Core_CPI_YoY": "CPILFESL",
            "Core_PCE_YoY": "PCEPILFE",
            "Sticky_CPI": "CORESTICKM159SFRBATL",
            "Trimmed_Mean_PCE": "PCETRIM12M159SFRBDAL",
            "5Y_Breakeven": "T5YIE",
            "Fed_Funds_Rate": "FEDFUNDS",
            "2s10s_Curve": "T10Y2Y",
            "10Y_Real_Yield": "DFII10",
            "10Y_minus_3Mo": "T10Y3M",
            "Mortgage_30Y": "MORTGAGE30US",
            "HY_Credit_Spread": "BAMLH0A0HYM2",
            "IG_Credit_Spread": "BAMLC0A0CM",
            "Chicago_Fed_NFCI": "NFCI",
            "SLOOS_Lending_Stds": "DRTSCILM"
        }

        try:
            raw_historical_data = {}
            for name, series_id in series_map.items():
                try:
                    # Pull 15 years of history to ensure we have enough data after pct_change drops
                    raw_historical_data[name] = self.fred.get_series(series_id, observation_start='2010-01-01')
                except Exception as e:
                    print(f"Warning: Could not fetch {series_id} ({name}): {e}")

            # Assemble into a DataFrame and align by month
            historical_df = pd.DataFrame(raw_historical_data).resample('ME').last()
            historical_df = historical_df.ffill() # Forward fill missing daily/weekly gaps

            # =========================================================
            # 2. MATHEMATICAL UNIT TRANSFORMATIONS
            # All series are converted to YoY changes before entering
            # the z-score engine, so scores reflect momentum/direction
            # rather than absolute levels.
            # =========================================================

            # --- A. Index levels → YoY % change: pct_change(12) * 100 ---
            for col in ['Ind_Prod_YoY', 'Retail_Sales_YoY', 'Core_CPI_YoY',
                        'Core_PCE_YoY', 'Initial_Claims_4Wk']:
                if col in historical_df.columns:
                    historical_df[col] = historical_df[col].pct_change(periods=12) * 100

            # --- B. Payrolls → 3-month avg of monthly job gains (diff, not %) ---
            if 'NFP_3Mo_Avg' in historical_df.columns:
                historical_df['NFP_3Mo_Avg'] = (
                    historical_df['NFP_3Mo_Avg'].diff().rolling(window=3).mean()
                )

            # --- C. Rates, spreads, diffusion indices → YoY change (diff in pp) ---
            # These are already in percentage/index units; diff(12) gives the
            # year-over-year change in percentage points or index points.
            yoy_diff_cols = [
                'Real_GDP_Growth',      # QoQ annualized rate → YoY change in rate
                'Unemployment_Rate',    # Rate (%) → YoY pp change
                'Prime_Age_LFPR',       # Rate (%) → YoY pp change
                'Mfg_PMI_Proxy',        # Diffusion index → YoY change
                '5Y_Breakeven',         # Rate (%) → YoY pp change
                'Fed_Funds_Rate',       # Rate (%) → YoY pp change
                '2s10s_Curve',          # Spread (pp) → YoY pp change
                '10Y_Real_Yield',       # Rate (%) → YoY pp change
                '10Y_minus_3Mo',        # Spread (pp) → YoY pp change
                'Mortgage_30Y',         # Rate (%) → YoY pp change
                'HY_Credit_Spread',     # Spread (%) → YoY pp change
                'IG_Credit_Spread',     # Spread (%) → YoY pp change
                'Chicago_Fed_NFCI',     # Index → YoY change
                'SLOOS_Lending_Stds',   # Net % of banks → YoY pp change
            ]
            for col in yoy_diff_cols:
                if col in historical_df.columns:
                    historical_df[col] = historical_df[col].diff(periods=12)

            # --- D. Already YoY % from FRED — no transform needed ---
            # Sticky_CPI (CORESTICKM159SFRBATL) — published as YoY%
            # Trimmed_Mean_PCE (PCETRIM12M159SFRBDAL) — published as YoY%

            # Drop the NaN rows created by the 12-month lookback windows
            historical_df = historical_df.dropna(subset=['Core_CPI_YoY', 'NFP_3Mo_Avg'])

            # 3. Apply canonical aliases so engine indicator names are present
            historical_df = apply_aliases_to_df(historical_df)

            # 4. EXTRACT CURRENT MACRO INPUTS (The absolute latest valid row)
            current_macro = historical_df.iloc[-1].to_dict()
            current_macro = apply_aliases_to_dict(current_macro)
            
            return current_macro, historical_df
        except Exception as exc:
            print(f"Live FRED feed unavailable: {exc}. Falling back to bundled macro CSV data.")
            return self._load_local_macro_data()

    def fetch_technical_inputs(self, tickers, target_date=None):
        """
        Calculates Price vs 200D MA and RSI using yfinance data.
        """
        tech_data = {}
        print("Fetching Technical Data (200D MA & RSI)...")

        for t in tickers:
            try:
                if target_date:
                    end_date = target_date + datetime.timedelta(days=1)
                    start_date = target_date - datetime.timedelta(days=365)
                    df = yf.download(t, start=start_date, end=end_date, interval="1d", progress=False)
                else:
                    df = yf.download(t, period="1y", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                series = df['Close'].dropna()
                if series.empty:
                    raise ValueError(f"No data for {t}")

                current_price = float(series.iloc[-1])

                # 200-Day Moving Average
                ma_200 = float(series.rolling(window=200).mean().iloc[-1])
                price_vs_200d = ((current_price / ma_200) - 1) * 100 if ma_200 != 0 else 0.0
                
                # 14-Day RSI
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = float(gain.iloc[-1]) / (float(loss.iloc[-1]) + 1e-9)
                rsi = 100 - (100 / (1 + rs))
                
                tech_data[t] = {"Price_vs_200D": round(price_vs_200d, 2), "RSI": round(rsi, 2)}
            except Exception:
                tech_data[t] = {"Price_vs_200D": 0.0, "RSI": 50.0} # Fallback
                
        return tech_data