import streamlit as st
import yfinance as yf
import pandas as pd
import joblib

st.set_page_config(
    page_title="Smart Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: #f4f7fc;
}

.block-container {
    padding-top: 1.2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Header */
.header-box{
    background: linear-gradient(135deg,#0000FF,#1f2937);
    padding: 28px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 22px;
}

.header-title{
    font-size: 42px;
    font-weight: 700;
}

.header-sub{
    font-size: 16px;
    color: #d1d5db;
    margin-top: 5px;
}

/* Search */
.stTextInput input{
    background: white !important;
    color: black !important;
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

/* Button */
.stButton button{
    background: linear-gradient(90deg,#2563eb,#1d4ed8);
    color: white;
    border: none;
    padding: 12px 26px;
    border-radius: 12px;
    font-weight: 600;
}

.stButton button:hover{
    background: #1d4ed8;
}

/* Cards */
.metric-box{
    background:white;
    padding:22px;
    border-radius:18px;
    box-shadow:0 8px 18px rgba(0,0,0,0.06);
    text-align:center;
}

.metric-label{
    color:#6b7280;
    font-size:15px;
}

.metric-value{
    font-size:34px;
    font-weight:700;
    color:#111827;
}

/* Prediction */
.up-box{
    background:#ecfdf5;
    color:#065f46;
    border-left:6px solid #10b981;
    padding:18px;
    border-radius:14px;
    font-size:26px;
    font-weight:700;
}

.down-box{
    background:#fef2f2;
    color:#991b1b;
    border-left:6px solid #ef4444;
    padding:18px;
    border-radius:14px;
    font-size:26px;
    font-weight:700;
}

/* Footer */
.footer{
    text-align:center;
    color:#6b7280;
    margin-top:30px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header-box">
<div class="header-title">📈 Smart Stock Predictor</div>
<div class="header-sub">Live Market Data • AI Prediction • Professional Analytics</div>
</div>
""", unsafe_allow_html=True)

model = joblib.load("final_stock_model.pkl")

manual_map = {
    "tata motors": "TATAMOTORS.NS",
    "reliance": "RELIANCE.NS",
    "infosys": "INFY.NS",
    "tcs": "TCS.NS",
    "icici bank": "ICICIBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "axis bank": "AXISBANK.NS",
    "maruti": "MARUTI.NS",
    "wipro": "WIPRO.NS",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "google": "GOOGL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META"
}


def get_ticker(name):
    try:
        search = yf.Search(name)
        results = search.quotes

        for item in results:
            symbol = item.get("symbol", "")
            quote_type = item.get("quoteType", "")

            if quote_type == "EQUITY":
                return symbol

        return None
    except:
        return None

query = st.text_input("🔍 Enter Company Name or Symbol", "Nvidia")

if st.button("🚀 Search & Predict"):

    query_lower = query.lower().strip()

    if query_lower in manual_map:
        ticker = manual_map[query_lower]
    else:
        ticker = get_ticker(query)

    if ticker is None:
        st.error("❌ Company not found")
    else:
        st.success(f"✅ Ticker Found: {ticker}")

        data = yf.download(ticker, period="6mo", interval="1d", progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            st.error("❌ No stock data found")
        else:

            latest = data.iloc[-1]

            close_price = float(latest["Close"])
            open_price = float(latest["Open"])
            high_price = float(latest["High"])
            low_price = float(latest["Low"])

            # Features
            data["Returns"] = data["Close"].pct_change()
            data["SMA_5"] = data["Close"].rolling(5).mean()
            data["SMA_10"] = data["Close"].rolling(10).mean()

            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))

            data = data.dropna()

            latest_features = data.iloc[-1][["Returns","SMA_5","SMA_10","RSI"]]

            pred = model.predict([latest_features])[0]

            st.subheader(" Model Prediction")

            if pred == 1:
                st.markdown("<div class='up-box'>📈 Stock may go UP tomorrow</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='down-box'>📉 Stock may go DOWN tomorrow</div>", unsafe_allow_html=True)

            st.write("")

           
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f"""
                <div class='metric-box'>
                <div class='metric-label'>Open</div>
                <div class='metric-value'>{open_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class='metric-box'>
                <div class='metric-label'>High</div>
                <div class='metric-value'>{high_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class='metric-box'>
                <div class='metric-label'>Low</div>
                <div class='metric-value'>{low_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class='metric-box'>
                <div class='metric-label'>Close</div>
                <div class='metric-value'>{close_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.subheader("📊 Stock Price Trend")

            chart_data = data[["Close"]].dropna()

            if chart_data["Close"].nunique() > 1:
                st.line_chart(chart_data, height=420)
            else:
                st.warning("⚠️ Chart data not sufficient")


st.markdown("<div class='footer'> @2026 Copyright Made By Manvender Singh & Pranav Sharma using Streamlit + Machine Learning concepts</div>", unsafe_allow_html=True)