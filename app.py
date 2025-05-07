import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Add src directory to Python path
from pathlib import Path
import sys
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from src.data_fetcher import NSEDataFetcher
from src.indicators import TechnicalIndicators
from src.model import TradingModel
from src.backtest import Backtester

st.title("📊 AI-Driven Trading and Sentiment analysis system")

# Load stock list from CSV
symbols_df = pd.read_csv("nse_symbols.csv")
stock_dict = dict(zip(symbols_df["Company name"], symbols_df["Symbol"]))

# Autocomplete with selectbox
selected_company = st.sidebar.selectbox("🔍 Search Company", sorted(stock_dict.keys()))
symbol = stock_dict[selected_company]

# Initial capital input
initial_capital = st.sidebar.number_input("💰 Initial Capital (₹)", value=100000)

if st.button("Analyze"):
    st.subheader(f"Analyzing {selected_company} ({symbol})...")
    
    fetcher = NSEDataFetcher()
    data = fetcher.fetch_data(symbol, period='1y')

    if data is not None:
        indicators = TechnicalIndicators(data)
        data_with_indicators = indicators.add_all_indicators()

        model = TradingModel()
        features, target = model.prepare_features(data_with_indicators)
        model.train(features, target)

        latest_features = features.iloc[[-1]]
        prediction = model.predict(latest_features)
        probabilities = model.predict_proba(latest_features)

        st.markdown("### 📈 Prediction")
        signal = "BUY 🟢" if prediction[0] == 1 else "SELL 🔴"
        st.write(f"**Signal:** {signal}")
        st.write(f"**Confidence:** Up = {probabilities[0][1]:.2f}, Down = {probabilities[0][0]:.2f}")

        # ✅ Candlestick Chart
        st.markdown("### 📊 Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=data_with_indicators.index,
            open=data_with_indicators['Open'],
            high=data_with_indicators['High'],
            low=data_with_indicators['Low'],
            close=data_with_indicators['Close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig)

        # ✅ Moving Averages
        st.markdown("### 📉 Moving Averages")
        plt.figure(figsize=(10, 4))
        plt.plot(data_with_indicators['Close'], label='Close Price', linewidth=1.5)
        if 'SMA_20' in data_with_indicators.columns:
            plt.plot(data_with_indicators['SMA_20'], label='SMA 20')
        if 'EMA_50' in data_with_indicators.columns:
            plt.plot(data_with_indicators['EMA_50'], label='EMA 50')
        plt.legend()
        plt.title(f'{selected_company} Price with SMA & EMA')
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(plt)

        # ✅ Backtest
        st.markdown("### 🧪 Backtest Results")
        backtester = Backtester(initial_capital=initial_capital)
        stats = backtester.backtest(data_with_indicators, model.predict(features))
        for key, value in stats.items():
            if "Rate" in key or "Returns" in key:
                st.write(f"{key}: {value:.2%}")
            else:
                st.write(f"{key}: {value:.2f}")

        # ✅ PDF Export
        st.markdown("### 📄 Export Report as PDF")
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, f"Stock Analysis Report - {selected_company} ({symbol})")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, height - 110, f"Signal: {signal}")
        c.drawString(50, height - 130, f"Confidence: Up = {probabilities[0][1]:.2f}, Down = {probabilities[0][0]:.2f}")
        c.drawString(50, height - 160, "Backtest Results:")

        y = height - 180
        for key, value in stats.items():
            val = f"{value:.2%}" if "Rate" in key or "Returns" in key else f"{value:.2f}"
            c.drawString(60, y, f"- {key}: {val}")
            y -= 20

        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{symbol}_report.pdf",
            mime="application/pdf"
        )

    else:
        st.warning("⚠️ Failed to fetch data. Try again or check your internet.")
