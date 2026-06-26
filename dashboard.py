import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import importlib
import os

# Import modules
from live_data_feeder import LiveDataFeeder, apply_aliases_to_dict, apply_aliases_to_df
macro_engine_module = importlib.import_module("MACRO ENGINE")
MacroEngine = macro_engine_module.MacroEngine

TICKER_NAMES = {
    "XLC": "Communication Services (XLC)",
    "XLY": "Consumer Discretionary (XLY)",
    "XLP": "Consumer Staples (XLP)",
    "XLE": "Energy (XLE)",
    "XLF": "Financials (XLF)",
    "XLV": "Health Care (XLV)",
    "XLI": "Industrials (XLI)",
    "XLK": "Technology (XLK)",
    "XLB": "Materials (XLB)",
    "XLRE": "Real Estate (XLRE)",
    "XLU": "Utilities (XLU)",
    "EEM": "Emerging Markets (EEM)",
    "IWM": "Russell 2000 (IWM)",
}

st.set_page_config(page_title="Macro Portfolio Manager", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZATION ---
@st.cache_resource
def get_engine():
    return MacroEngine()

@st.cache_resource
def get_feeder():
    return LiveDataFeeder()

engine = get_engine()
feeder = get_feeder()

# Initialize session state for data
if 'macro_data' not in st.session_state:
    st.session_state.macro_data = None
if 'engine_results' not in st.session_state:
    st.session_state.engine_results = None
if 'base_dir' not in st.session_state:
    st.session_state.base_dir = os.path.dirname(os.path.abspath(__file__))

def fetch_data():
    with st.spinner("Fetching Live Data (FRED & Yahoo Finance)..."):
        try:
            current_macro, historical_df = feeder.fetch_macro_inputs()
            market_data = feeder.fetch_market_data()
            technical_mrc = feeder.fetch_technical_inputs(engine.tickers)
            commodity_signals = market_data.get("Commodity_Signals", {})
            
            st.session_state.macro_data = {
                "current_macro": current_macro,
                "historical_df": historical_df,
                "market_data": market_data,
                "technical_mrc": technical_mrc,
                "commodity_signals": commodity_signals
            }
            st.success("Data fetched successfully!")
        except Exception as e:
            st.error(f"Live Feed Unavailable: {e}")
            st.warning("Falling back to local CSV data...")
            base_dir = st.session_state.base_dir
            try:
                current_macro = {row["metric"]: float(row["value"]) for _, row in pd.read_csv(os.path.join(base_dir, "macro_current_data.csv")).iterrows()}
                historical_df = pd.read_csv(os.path.join(base_dir, "macro_historical_data.csv"), index_col=0)
                historical_df = historical_df.apply(pd.to_numeric, errors="coerce")
                date_cols = [c for c in historical_df.columns if c.startswith("date")]
                if date_cols:
                    historical_df = historical_df.drop(columns=date_cols)
                current_macro = apply_aliases_to_dict(current_macro)
                historical_df = apply_aliases_to_df(historical_df)
                commodity_signals = {row["metric"]: float(row["value"]) for _, row in pd.read_csv(os.path.join(base_dir, "macro_commodity_data.csv")).iterrows()}
                technical_mrc = {row["ticker"]: {"Price_vs_200D": row["Price_vs_200D"], "RSI": row["RSI"]} for _, row in pd.read_csv(os.path.join(base_dir, "macro_technical_data.csv")).iterrows()}
                
                st.session_state.macro_data = {
                    "current_macro": current_macro,
                    "historical_df": historical_df,
                    "market_data": {"Commodity_Signals": commodity_signals},
                    "technical_mrc": technical_mrc,
                    "commodity_signals": commodity_signals
                }
                st.success("Loaded local fallback data.")
            except Exception as inner_e:
                st.error(f"Failed to load fallback data: {inner_e}")

def run_engine_pipeline(neutral_weights_input):
    if st.session_state.macro_data is None:
        st.warning("Please fetch data first!")
        return
    
    with st.spinner("Running Macro Engine..."):
        # Update engine's neutral weights based on UI input
        engine.neutral_weights = neutral_weights_input
        
        data = st.session_state.macro_data
        # Prev weights just using current neutral for now, or could persist
        prev_weights = engine.neutral_weights 
        
        results = engine.run_pipeline(
            data["current_macro"], 
            data["historical_df"], 
            data["commodity_signals"], 
            data["technical_mrc"], 
            prev_weights
        )
        st.session_state.engine_results = results
        
        # Save output using the module's write function
        macro_engine_module.write_live_outputs(
            st.session_state.base_dir,
            data["current_macro"],
            data["historical_df"],
            data["commodity_signals"],
            data["technical_mrc"]
        )

# --- SIDEBAR: NEUTRAL WEIGHTS & CONTROLS ---
st.sidebar.title("Macro Portfolio Manager")
st.sidebar.subheader("Controls")

if st.sidebar.button("🔄 Refresh Data (FRED / Yahoo)", use_container_width=True):
    fetch_data()

# Initialize data on first run if empty
if st.session_state.macro_data is None:
    fetch_data()

st.sidebar.markdown("---")
st.sidebar.subheader("Neutral Weights (%)")

# Create inputs for neutral weights
edited_neutral_weights = {}
total_weight = 0.0

# Two columns for compact layout in sidebar
col1, col2 = st.sidebar.columns(2)
for i, ticker in enumerate(engine.tickers):
    current_val = engine.neutral_weights[ticker] * 100
    with (col1 if i % 2 == 0 else col2):
        label = TICKER_NAMES.get(ticker, ticker)
        val = st.number_input(f"{label}", min_value=0.0, max_value=100.0, value=float(current_val), step=1.0, format="%.2f")
        edited_neutral_weights[ticker] = val / 100.0
        total_weight += val

# Display total weight
st.sidebar.markdown(f"**Total Weight: {total_weight:.2f}%**")
if not np.isclose(total_weight, 100.0, atol=0.01):
    st.sidebar.error("⚠️ Weights must sum to 100%")
    
if st.sidebar.button("▶ Run Engine", type="primary", use_container_width=True):
    if np.isclose(total_weight, 100.0, atol=0.01):
        # Re-normalize just to be safe
        normalized = {k: v / sum(edited_neutral_weights.values()) for k, v in edited_neutral_weights.items()}
        run_engine_pipeline(normalized)
    else:
        st.sidebar.error("Cannot run: Weights do not sum to 100%")

# Run once automatically if results are empty and weights are valid
if st.session_state.engine_results is None and np.isclose(total_weight, 100.0, atol=0.01):
    run_engine_pipeline(edited_neutral_weights)

# --- MAIN DASHBOARD AREA ---
if st.session_state.engine_results is not None:
    results = st.session_state.engine_results
    regime = results["regime"]
    classification = results["classification"]
    df_results = results["results_df"].copy()
    df_results.index = df_results.index.map(lambda x: TICKER_NAMES.get(x, x))
    
    # 1. Regime Banner
    regime_colors = {
        "Reflationary Boom": "#2e7d32", # Green
        "Disinflationary Bust": "#1565c0", # Blue
        "Inflationary Boom": "#ef6c00", # Orange
        "Stagflation": "#c62828" # Red
    }
    color = regime_colors.get(classification, "#333333")
    
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;">
            <h1 style="margin: 0; padding: 0; text-align: center;">{classification.upper()}</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Growth Score", f"{regime['Growth']:.2f}")
    col2.metric("Inflation Score", f"{regime['Inflation']:.2f}")
    col3.metric("Financial Conditions", f"{regime['Financial_Conditions']:.2f}")
    col4.metric("Turnover", f"{results['turnover']*100:.2f}%")
    
    st.markdown("---")
    
    # 2. Bucket Scores & Radar
    st.subheader("Macro Bucket Scores")
    c1, c2 = st.columns([1, 1])
    
    bucket_scores = regime["Bucket_Scores"]
    buckets = list(bucket_scores.keys())
    scores = list(bucket_scores.values())
    
    with c1:
        # Radar Chart
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=scores + [scores[0]], # Close the polygon
          theta=[b.title() for b in buckets] + [buckets[0].title()],
          fill='toself',
          line_color=color
        ))
        fig_radar.update_layout(
          polar=dict(
            radialaxis=dict(visible=True, range=[-3, 3])
          ),
          showlegend=False,
          margin=dict(l=40, r=40, t=20, b=20),
          height=350
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with c2:
        # Horizontal Bar Chart
        df_scores = pd.DataFrame({"Bucket": [b.title() for b in buckets], "Score": scores})
        df_scores = df_scores.sort_values("Score")
        df_scores["Color"] = df_scores["Score"].apply(lambda x: "positive" if x >= 0 else "negative")
        
        fig_bar = px.bar(
            df_scores, 
            x="Score", 
            y="Bucket", 
            orientation="h",
            color="Color",
            color_discrete_map={"positive": "#2e7d32", "negative": "#c62828"}
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="Z-Score", yaxis_title="", margin=dict(l=0, r=0, t=20, b=0), height=350)
        # Add a vertical line at 0
        fig_bar.add_vline(x=0, line_width=2, line_color="black")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # 3. Allocations
    st.subheader("Portfolio Allocations")
    
    # Plotly grouped bar chart
    df_chart = df_results.copy().reset_index()
    # Melt for grouped bar chart
    df_melt = df_chart.melt(id_vars=["Ticker"], value_vars=["Neutral", "Final_Wt"], var_name="Type", value_name="Weight")
    
    fig_alloc = px.bar(
        df_melt, 
        x="Ticker", 
        y="Weight", 
        color="Type", 
        barmode="group",
        color_discrete_map={"Neutral": "#9e9e9e", "Final_Wt": color},
        text_auto=".1%"
    )
    fig_alloc.update_layout(yaxis_tickformat='.1%', yaxis_title="Weight", xaxis_title="Sector / Asset Class")
    st.plotly_chart(fig_alloc, use_container_width=True)
    
    c_tab1, c_tab2 = st.columns([2, 1])
    
    with c_tab1:
        # Formatted table
        df_display = df_results.copy()
        df_display["Neutral"] = (df_display["Neutral"] * 100).map("{:.2f}%".format)
        df_display["Raw_Tilt"] = (df_display["Raw_Tilt"] * 100).map("{:+.2f}%".format)
        df_display["MRC_x"] = df_display["MRC_x"].map("{:.2f}x".format)
        df_display["Final_Wt"] = (df_display["Final_Wt"] * 100).map("{:.2f}%".format)
        st.dataframe(df_display, use_container_width=True)
        
    with c_tab2:
        # Donut Chart
        fig_donut = px.pie(
            df_chart, 
            values='Final_Wt', 
            names='Ticker', 
            hole=0.4,
            title="Final Composition"
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    
    # 4. Data Explorer & Commodities
    st.subheader("Underlying Data Explorer")
    
    tab1, tab2 = st.tabs(["Macro Bucket Metrics", "Commodity & FX Signals"])
    
    with tab1:
        # Build a dataframe of all metrics, their current values, and their buckets
        metrics_data = []
        historical_df = st.session_state.macro_data["historical_df"]
        current_data = st.session_state.macro_data["current_macro"]
        
        for spec in engine.indicator_specs:
            metric = spec["metric"]
            bucket = spec["bucket"].title()
            if metric in current_data and metric in historical_df.columns:
                val = current_data[metric]
                if not pd.isna(val):
                    # Get z-score
                    z = engine.transform_indicator_score(spec, val, historical_df[metric])
                    metrics_data.append({
                        "Bucket": bucket,
                        "Metric": metric,
                        "Value": val,
                        "Direction": spec["direction"],
                        "Z-Score Contribution": z
                    })
                    
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create sub-tabs for each bucket
            bucket_names = df_metrics["Bucket"].unique()
            bucket_tabs = st.tabs(list(bucket_names))
            
            for i, b_name in enumerate(bucket_names):
                with bucket_tabs[i]:
                    b_df = df_metrics[df_metrics["Bucket"] == b_name].drop(columns=["Bucket"])
                    # Format float columns
                    st.dataframe(
                        b_df.style.format({
                            "Value": "{:.4f}",
                            "Z-Score Contribution": "{:+.2f}"
                        }),
                        use_container_width=True
                    )
    
    with tab2:
        comm_data = st.session_state.macro_data["commodity_signals"]
        
        # Gauges
        c1, c2, c3 = st.columns(3)
        
        def make_indicator(title, val):
            if val < -1:
                color = "#ff9999" # light coral
            elif val > 1:
                color = "#99ff99" # light green
            else:
                color = "#f0f0f0" # light gray
                
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: black; border: 1px solid #ddd; margin-bottom: 20px;">
                <h4 style="margin: 0; padding-bottom: 10px;">{title}</h4>
                <h2 style="margin: 0;">{val:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with c1:
            make_indicator("Oil Signal", comm_data.get("Oil_Signal", 0))
        with c2:
            make_indicator("BDI Signal", comm_data.get("BDI_Signal", 0))
        with c3:
            make_indicator("CuGold Signal", comm_data.get("CuGold_Signal", 0))
            
        st.markdown("#### Raw Market Data")
        raw_market = st.session_state.macro_data["market_data"]
        raw_market_flat = [{"Metric": k, "Value": v} for k, v in raw_market.items() if k != "Commodity_Signals" and v is not None]
        if raw_market_flat:
            st.dataframe(pd.DataFrame(raw_market_flat), use_container_width=True)

else:
    st.info("👈 Please Fetch Data and Run the Engine from the sidebar.")
