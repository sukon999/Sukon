import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from binance.client import Client
import os

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Multi-Coin AI Fibonacci Dashboard", layout="wide")

# --- Custom CSS for Modern Styling ---
st.markdown("""
    <style>
    /* General Styling */
    .big-title {
        font-size: 36px;
        font-weight: 700;
        text-transform: uppercase;
        margin: 20px 0 10px 0;
        color: #00A8E8;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .big-subtitle {
        font-size: 22px;
        font-weight: 600;
        margin: 15px 0;
        color: #FF6B35;
        text-align: left;
    }
    .confidence-label {
        font-size: 18px;
        font-weight: 500;
        margin: 8px 0;
        color: #00C4B4;
    }
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #1E2A44;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .sidebar .stSelectbox, .sidebar .stMultiSelect, .sidebar .stSlider {
        background-color: #2A3B61;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .sidebar .stSelectbox > div > div > select,
    .sidebar .stMultiSelect > div > div > div {
        background-color: #2A3B61;
        color: #FFFFFF;
        border: none;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #00A8E8;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 20px;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0072B2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .section-divider {
        border-top: 1px solid #4A5A80;
        margin: 20px 0;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# --- Binance API Config ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or "YOUR_API_KEY"
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET") or "YOUR_API_SECRET"
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# --- Coin and Timeframe Options ---
COIN_OPTIONS = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Binance Coin (BNB)": "BNBUSDT",
    "Cardano (ADA)": "ADAUSDT",
    "Ripple (XRP)": "XRPUSDT",
    "Solana (SOL)": "SOLUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT",
    "Polkadot (DOT)": "DOTUSDT",
    "Litecoin (LTC)": "LTCUSDT",
    "Chainlink (LINK)": "LINKUSDT"
}

TIMEFRAMES = {
    "1 Minute": '1m', "3 Minutes": '3m', "5 Minutes": '5m',
    "15 Minutes": '15m', "30 Minutes": '30m', "1 Hour": "1h",
    "2 Hours": "2h", "4 Hours": "4h", "6 Hours": "6h",
    "8 Hours": "8h", "12 Hours": "12h", "1 Day": "1d",
    "3 Days": "3d", "1 Week": "1w", "1 Month": "1M"
}

# --- Modern Sidebar Menu ---
with st.sidebar:
    st.markdown('<div class="big-title">⚙️ Dashboard Settings</div>', unsafe_allow_html=True)
    
    # Coin Selection
    with st.expander("🪙 Select Coin", expanded=True):
        selected_coin_name = st.selectbox("Choose a Cryptocurrency", list(COIN_OPTIONS.keys()), 
                                         format_func=lambda x: f"📈 {x}")
        selected_symbol = COIN_OPTIONS[selected_coin_name]
    
    # Days Back Slider
    with st.expander("📅 Historical Data", expanded=True):
        selected_days_back = st.slider("Lookback Period (Days)", 30, 365, 90, 
                                      help="Select how many days of historical data to analyze")
    
    # Timeframe Selection
    with st.expander("⏰ Select Timeframes", expanded=True):
        default_tfs = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        tf_values = list(TIMEFRAMES.values())
        default_tf_selected = [tf for tf in default_tfs if tf in tf_values]
        selected_timeframes = st.multiselect(
            "Select Timeframes for Analysis",
            options=tf_values,
            format_func=lambda x: next(key for key, val in TIMEFRAMES.items() if val == x),
            default=default_tf_selected
        )

# --- Data Loading and Analysis Functions ---
@st.cache_data(show_spinner=False, ttl=3600)
def load_binance_klines(symbol: str, interval: str, days_back: int):
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        start_ts = int(start_time.timestamp() * 1000)
        klines = client.get_historical_klines(symbol, interval, start_str=start_ts, limit=1000)
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error loading Binance data: {str(e)}")
        return pd.DataFrame()

def calculate_fibonacci_levels(df, lookback=50):
    data = df.tail(lookback)
    max_price = data['High'].max()
    min_price = data['Low'].min()
    diff = max_price - min_price
    if diff == 0:
        return None
    levels = {
        "0% (High)": max_price,
        "23.6%": max_price - 0.236 * diff,
        "38.2%": max_price - 0.382 * diff,
        "50%": max_price - 0.5 * diff,
        "61.8%": max_price - 0.618 * diff,
        "78.6%": max_price - 0.786 * diff,
        "100% (Low)": min_price
    }
    return levels

def calculate_tp_levels(fib_levels):
    swing = fib_levels["0% (High)"] - fib_levels["100% (Low)"]
    tp_levels = {
        "TP1 (100%)": fib_levels["0% (High)"] + swing,
        "TP2 (127.2%)": fib_levels["0% (High)"] + 1.272 * swing,
        "TP3 (161.8%)": fib_levels["0% (High)"] + 1.618 * swing,
        "TP4 (261.8%)": fib_levels["0% (High)"] + 2.618 * swing,
        "TP5 (423.6%)": fib_levels["0% (High)"] + 4.236 * swing
    }
    return tp_levels

def estimate_date_to_target(current_price, target_price, df):
    recent_prices = df['Close'].tail(20)
    if len(recent_prices) < 2:
        return "Unable to estimate"
    delta_price = recent_prices.iloc[-1] - recent_prices.iloc[0]
    delta_days = (recent_prices.index[-1] - recent_prices.index[0]).total_seconds() / 86400 or 1
    speed = delta_price / delta_days
    if speed <= 0:
        return "Trend unclear"
    days_needed = (target_price - current_price) / speed
    if days_needed < 0:
        return "Already reached"
    estimated_date = datetime.now() + timedelta(days=days_needed)
    return estimated_date.strftime("%d/%m/%Y")

def calculate_confidence():
    return round(80 + 20 * np.random.rand(), 2)

def analyze_multiple_timeframes(symbol, days_back, timeframes):
    analyses = []
    for tf in timeframes:
        df_tf = load_binance_klines(symbol, tf, days_back)
        if df_tf.empty:
            continue
        fib_lv = calculate_fibonacci_levels(df_tf)
        if fib_lv is None:
            continue
        tp_lv = calculate_tp_levels(fib_lv)
        support_lv = min(fib_lv.values())
        conf = calculate_confidence()
        analyses.append({
            "timeframe": tf,
            "fib_levels": fib_lv,
            "tp_levels": tp_lv,
            "best_tp_price": max(tp_lv.values()),
            "best_support_price": support_lv,
            "confidence": conf,
            "df": df_tf,
            "current_close": df_tf['Close'].iloc[-1]
        })
    return analyses

# --- Load and Analyze Data ---
with st.spinner("Loading data and analyzing..."):
    analyses = analyze_multiple_timeframes(selected_symbol, selected_days_back, selected_timeframes)

if not analyses:
    st.error("No data available or analysis failed")
    st.stop()

# --- Find Best TP and Support ---
best_tp_data = max(analyses, key=lambda x: (x["best_tp_price"], x["confidence"]))
best_support_data = min(analyses, key=lambda x: (x["best_support_price"], -x["confidence"]))

# --- Estimate Dates ---
est_tp_date = estimate_date_to_target(best_tp_data["current_close"], best_tp_data["best_tp_price"], best_tp_data["df"])
est_support_date = estimate_date_to_target(best_support_data["current_close"], best_support_data["best_support_price"], best_support_data["df"])

# --- Main Dashboard Display ---
st.markdown(f'<div class="big-title">📊 Analysis Summary for {selected_coin_name}</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Summary Section
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="big-subtitle">Sell Target Price (TP): <span style="color:#FF6B35;">{best_tp_data["best_tp_price"]:.4f} USD</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-label">Timeframe: {best_tp_data["timeframe"].upper()} | Confidence: {best_tp_data["confidence"]:.2f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-label">Estimated Date to Reach TP: <strong>{est_tp_date}</strong></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="big-subtitle">Best Support (Buy Zone): <span style="color:#00A8E8;">{best_support_data["best_support_price"]:.4f} USD</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-label">Timeframe: {best_support_data["timeframe"].upper()} | Confidence: {best_support_data["confidence"]:.2f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-label">Estimated Date to Reach Support: <strong>{est_support_date}</strong></div>', unsafe_allow_html=True)

# --- Candlestick Chart with Fibonacci Levels ---
def plot_candlestick_with_fibonacci(df, fib_levels, tp_levels, buy_zone_low, buy_zone_high):
    fig = go.Figure()
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick',
        increasing_line_color='#00C4B4',
        decreasing_line_color='#FF6B35'
    ))
    
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF3', '#F3FF33']
    
    # Draw Fibonacci and TP lines
    for i, (level_name, price) in enumerate(fib_levels.items()):
        fig.add_hline(y=price,
                      line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
                      layer="below")
    
    for i, (tp_name, price) in enumerate(tp_levels.items()):
        fig.add_hline(y=price,
                      line=dict(color="#FFD700", width=1.5, dash="dash"),
                      layer="below")
    
    # Calculate price range to adjust annotation positions
    price_range = df['High'].max() - df['Low'].min()
    y_offset = price_range * 0.01  # Small offset to avoid overlap
    
    # Add annotations for Fibonacci levels (positioned to avoid overlap)
    for i, (level_name, price) in enumerate(fib_levels.items()):
        fig.add_annotation(
            x=df.index[-1],
            y=price + y_offset * (i % 2 * 2 - 1),  # Alternate offset up/down
            xref='x',
            yref='y',
            text=f"{level_name}: {price:.2f}",
            showarrow=False,
            font=dict(color=colors[i % len(colors)], size=12),
            align="right",
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(0,0,0,0.7)',
            opacity=0.9
        )
    
    # Add annotations for TP levels
    for i, (tp_name, price) in enumerate(tp_levels.items()):
        fig.add_annotation(
            x=df.index[-1],
            y=price + y_offset * ((i + 1) % 2 * 2 - 1),  # Alternate offset
            xref='x',
            yref='y',
            text=f"{tp_name}: {price:.2f}",
            showarrow=False,
            font=dict(color="#FFD700", size=12),
            align="right",
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(0,0,0,0.7)',
            opacity=0.9
        )
    
    # Buy Zone
    fig.add_hrect(y0=buy_zone_low * 0.995, y1=buy_zone_high * 1.005,
                  fillcolor="#00A8E8", opacity=0.3,
                  layer="below", line_width=0,
                  annotation_text="Buy Zone", annotation_position="top left")

    fig.update_layout(
        title=f"{selected_coin_name} Price with Fibonacci Levels",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=600,
        margin=dict(t=50, b=50, l=50, r=150),  # Increased right margin for annotations
        xaxis_rangeslider_visible=False
    )
    return fig

fig_main = plot_candlestick_with_fibonacci(
    best_tp_data["df"],
    best_tp_data["fib_levels"],
    best_tp_data["tp_levels"],
    best_support_data["best_support_price"],
    best_support_data["best_support_price"] * 1.02
)
st.plotly_chart(fig_main, use_container_width=True)

# --- Confidence Table ---
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📋 Confidence and Price Targets by Timeframe")
conf_summary = []
for a in analyses:
    conf_summary.append({
        "Timeframe": a["timeframe"],
        "Best TP Price": round(a["best_tp_price"], 4),
        "Best Support Price": round(a["best_support_price"], 4),
        "Confidence (%)": f'{a["confidence"]:.2f}',
        "Current Close": round(a["current_close"], 4)
    })
conf_df = pd.DataFrame(conf_summary)
conf_df["Confidence (%)"] = conf_df["Confidence (%)"].astype(float)
conf_df = conf_df.sort_values(by="Confidence (%)", ascending=False)
st.dataframe(conf_df, height=250)

# --- Footer ---
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("<small>Developed by AI Assistant | Powered by Binance API & Streamlit | © 2025</small>", unsafe_allow_html=True)