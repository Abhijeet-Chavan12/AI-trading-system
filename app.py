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
import requests
import yfinance as yf

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from src.data_fetcher import NSEDataFetcher
from src.indicators import TechnicalIndicators
from src.model import TradingModel
from src.backtest import Backtester
from src.sentiment import fetch_company_news_sentiment  # ✅ Sentiment module

# Telegram alert function
def send_telegram_alert(message):
    bot_token = "7260593253:AAHc14JfN4mlasfW2v4r5ZljWgcGXwC83QU"
    chat_id = "7445572516"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Telegram alert error: {e}")
        return False

# Live price fetcher
def get_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period='1d', interval='1m')
        return todays_data['Close'].iloc[-1] if not todays_data.empty else None
    except Exception:
        return None

# Streamlit setup
st.set_page_config(page_title="AI Trading App", layout="wide")
st.title("📊 AI-Driven Trading and Sentiment Analysis System")

# Load symbols
symbols_df = pd.read_csv("nse_symbols.csv")
stock_dict = dict(zip(symbols_df["Company name"], symbols_df["Symbol"]))
selected_company = st.sidebar.selectbox("🔍 Search Company", sorted(stock_dict.keys()))
symbol = stock_dict[selected_company]

# Live price
live_price = get_live_price(symbol)
if live_price:
    st.sidebar.markdown(f"### 💹 Live Price of {symbol}: ₹{live_price:.2f}")
else:
    st.sidebar.markdown("⚠️ Unable to fetch live price.")

# Capital
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

        # ✅ Sentiment Analysis
        sentiment, sentiment_score, sentiment_articles = fetch_company_news_sentiment(selected_company)
        sentiment_icon = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}
        st.markdown("### 🧠 Market Sentiment")
        st.write(f"**Sentiment:** {sentiment_icon[sentiment]} {sentiment} (Score: {sentiment_score:.2f})")

        # ✅ Prediction
        st.markdown("### 📈 Prediction")
        signal = "BUY 🟢" if prediction[0] == 1 else "SELL 🔴"
        st.write(f"**Signal:** {signal}")
        st.write(f"**Confidence:** Up = {probabilities[0][1]:.2f}, Down = {probabilities[0][0]:.2f}")

        # ✅ Blended Signal with No Trade Suggestion
        st.markdown("### 🧠📈 Blended Signal")
        if prediction[0] == 1 and sentiment == "Positive":
            st.success(f"✅ Strong Buy Signal — AI and Sentiment agree for {selected_company} ({symbol})")
            blended_signal = "BUY"
        elif prediction[0] == 0 and sentiment == "Negative":
            st.error(f"🔻 Strong Sell Signal — AI and Sentiment agree for {selected_company} ({symbol})")
            blended_signal = "SELL"
        else:
            st.warning(f"⚖️ No Trade Recommended — AI and Sentiment disagree for {selected_company} ({symbol})")
            blended_signal = "HOLD"

        # ✅ Top News
        st.markdown("### 📰 News Impacting Sentiment")
        with st.expander("Show articles"):
            for article in sentiment_articles:
                icon = "🟢" if article["score"] > 0.05 else "🔴" if article["score"] < -0.05 else "🟡"
                st.markdown(f"- {icon} [{article['title']}]({article['url']}) — *Score:* `{article['score']:.2f}`")

        # ✅ Telegram Alert
        alert_msg = (
            f"📢 *Trade Alert for {selected_company} ({symbol}):*\n"
            f"🔀 *Blended Signal:* {blended_signal}\n"
            f"📊 *AI Confidence:* Up = {probabilities[0][1]:.2%}, Down = {probabilities[0][0]:.2%}\n"
            f"🧠 *Sentiment:* {sentiment} (Score: {sentiment_score:.2f})"
        )
        send_telegram_alert(alert_msg)

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
            st.write(f"{key}: {value:.2%}" if "Rate" in key or "Returns" in key else f"{key}: {value:.2f}")

        # ✅ PDF Report
        st.markdown("### 📄 Export Report as PDF")
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, f"Stock Analysis Report - {selected_company} ({symbol})")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, height - 110, f"Blended Signal: {blended_signal}")
        c.drawString(50, height - 130, f"Symbol: {symbol}")
        c.drawString(50, height - 150, f"Confidence: Up = {probabilities[0][1]:.2f}, Down = {probabilities[0][0]:.2f}")
        c.drawString(50, height - 170, f"Sentiment: {sentiment} (Score: {sentiment_score:.2f})")
        c.drawString(50, height - 190, "Backtest Results:")

        y = height - 210
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

        # ✅ Email Share Button
        email_subject = f"Trade Alert: {selected_company} ({symbol})"
        email_body = f"""
📢 *Trade Alert:*
🟩 *Signal:* {blended_signal}
🧠 *Sentiment:* {sentiment} (Score: {sentiment_score:.2f})
📊 *Confidence:* Up = {probabilities[0][1]:.2%}, Down = {probabilities[0][0]:.2%}
"""
        mailto_link = f"mailto:?subject={email_subject}&body={email_body}"

        st.markdown(f'<a href="{mailto_link}" target="_blank">📧 Click here to share via email</a>', unsafe_allow_html=True)

    else:
        st.warning("⚠️ Failed to fetch data. Try again or check your internet.")
