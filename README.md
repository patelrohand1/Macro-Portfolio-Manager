# Macro Portfolio Manager

A comprehensive, quantitative macroeconomic portfolio management system that builds data-driven portfolio allocations based on live macroeconomic indicators, market data, and technical health signals.

**Deployed Website**: https://dashboardpy-jgu35wnno6cv3ed5qm6uhe.streamlit.app/

## Project Overview

This project consists of a full-stack data pipeline, quantitative engine, and interactive dashboard to dynamically adjust portfolio weights across 13 core ETFs (Sector and Broad Market) based on the current economic regime. 

The portfolio adjusts around predefined neutral weights, applying calculated tilts driven by macroeconomic data (via FRED API) and market signals (via Yahoo Finance). 

### Core Components
1. **Live Data Feeder (`live_data_feeder.py`)**: Fetches, cleans, and transforms raw data from external APIs.
2. **Macro Engine (`MACRO ENGINE.py`)**: Processes data through a z-score pipeline, determines the macroeconomic regime, applies sensitivities to calculate tilts, incorporates technical multipliers, and enforces risk controls.
3. **Dashboard (`dashboard.py`)**: An interactive Streamlit web application to visualize the current regime, bucket scores, and final portfolio composition.

---

## 1. Data Ingestion & API Calls (`live_data_feeder.py`)

The data pipeline relies on two primary data sources:

*   **FRED API (`fredapi`)**: Fetches over 20 macroeconomic time series (e.g., Real GDP Growth, ISM Manufacturing, Nonfarm Payrolls, Core CPI, Fed Funds Rate, Yield Curve, Credit Spreads).
*   **Yahoo Finance API (`yfinance`)**: Fetches market data for commodities (WTI Crude, Copper, Gold, Baltic Dry Index), Foreign Exchange (DXY, USD/CNY), Volatility (VIX), and technical data (price history) for the target portfolio ETFs.

### Data Transformations
To ensure the mathematical models accurately capture economic *momentum* rather than raw absolute levels, the feeder performs strict transformations:
*   **Index Levels to YoY % Change**: Applied to indicators like Industrial Production, Retail Sales, CPI.
*   **Averages & Differencing**: Payrolls are converted to 3-month averages of monthly gains. Rates and spreads (e.g., Fed Funds, 10Y Yield, Credit Spreads) are transformed into YoY changes in percentage points.
*   **Local Fallback**: If APIs are unavailable (e.g., missing API key or network failure), the system gracefully falls back to local CSV snapshots (`macro_current_data.csv`, `macro_historical_data.csv`, etc.).

---

## 2. Quantitative Math Processes & Weighting (`MACRO ENGINE.py`)

The Engine calculates target portfolio weights using a multi-step mathematical pipeline.

### Step 1: Z-Score Normalization & Winsorization
Every macro indicator is compared against its historical distribution.
*   **Math**: `Z = (Current Value - Historical Mean) / Historical Standard Deviation`
*   **Winsorization**: The raw z-score is strictly bounded (capped) between `-3.0` and `+3.0`. This prevents "math explosions" during extreme economic events (like the COVID-19 shock).
*   **Directionality**: Scores are inverted if a "lower" number is economically positive for the bucket (e.g., lower Unemployment Rate).

### Step 2: Regime Classification
Z-scores are averaged into 6 core "buckets": Growth, Labor, Inflation, Fed Policy, Credit, and Rates.
These are further aggregated into 3 primary regime scores:
*   **Growth Score** = `Average(Growth Bucket, Labor Bucket)`
*   **Inflation Score** = `Inflation Bucket`
*   **Financial Conditions Score** = `Average(Fed Policy, Credit, Rates)`

Based on these 3 scores, the engine classifies the economy into one of four regimes:
1.  **Reflationary Boom**: Growth > 0, Inflation <= 0, Financial Cond > 0
2.  **Disinflationary Bust**: Growth <= 0, Inflation <= 0
3.  **Inflationary Boom**: Growth > 0, Inflation > 0
4.  **Stagflation**: Growth <= 0, Inflation > 0

### Step 3: Macro Themes & Sensitivity Matrix
The engine establishes 6 "Macro Themes": Real Economy, Monetary Pulse, Financial Conditions, Oil, Baltic Dry Index (BDI), and Copper/Gold Ratio.
*   A **Sensitivity Matrix** defines how each ETF reacts (multiplier values from -3.0 to +3.0) to these 6 themes.
*   **Raw Tilts**: Calculated via Matrix Multiplication (Dot Product) of the Sensitivity Matrix and the Macro Themes array. This is scaled down by a `conviction_flag` (0.75) and a `scalar` (0.0075).

### Step 4: Technical Multipliers (MRC - Mean Reversion / Momentum)
The raw tilts are adjusted based on the technical health of each ETF, calculated using:
*   **Price vs. 200-Day Moving Average**
*   **14-Day RSI (Relative Strength Index)**
*   **Logic**: If an ETF has a positive tilt but extremely poor technical health (falling knife), the tilt is shrunk (0.20x multiplier). If it has strong technical momentum matching a positive tilt, it is amplified (1.75x multiplier). Contrarian setups (negative tilt but strong technicals) are halved (0.50x multiplier).

### Step 5: Risk Controls & Final Weights
The engine adds the adjusted tilts to the **Neutral Weights** (the baseline portfolio allocation). Before finalizing, strict risk controls are applied:
1.  **Deviation Caps**: No asset can deviate by more than +/- 15% from its neutral weight.
2.  **Absolute Limits**: Emerging Markets (EEM) and Small Caps (IWM) are capped at 12% each. Energy (XLE) + Materials (XLB) combined cannot exceed 15%.
3.  **Defensive Limits**: Staples (XLP) + Health Care (XLV) + Utilities (XLU) combined cannot exceed 30%.
4.  **Turnover Constraints**: The system calculates the absolute turnover from the previous weights. If total turnover exceeds 25%, a shrink factor is applied to all trades to forcibly keep turnover at exactly 25%, minimizing transaction costs.

---

## 3. Interactive Dashboard (`dashboard.py`)

The Streamlit dashboard serves as the UI for the engine. 
*   **Live Controls**: A sidebar allows users to fetch fresh data and manually adjust the neutral weight anchors.
*   **Visualizations (Plotly)**:
    *   **Regime Banner**: Distinct color-coded banners indicating the current macroeconomic regime.
    *   **Radar & Bar Charts**: Visualize the z-scores for the 6 macroeconomic buckets.
    *   **Portfolio Allocations**: Grouped bar charts comparing the Neutral Weights vs. Final Recommended Weights, alongside a donut chart of the final composition.
    *   **Data Explorer**: Drill down into the underlying raw data, individual indicator z-scores, and commodity signals.

---

## Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   Create a `.env` file in the root directory and add your FRED API Key:
   ```env
   FRED_API_KEY=your_api_key_here
   ```
3. **Run the Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
