import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class NSEDataFetcher:
    def __init__(self):
        # Nifty 50 stocks - you can modify this list
        self.nifty50_tickers = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS'
        ]

    def fetch_data(self, ticker, period='1y', interval='1d'):
        """
        Fetch historical data for a given ticker
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df is None or df.empty:
                raise ValueError(f"No data found for {ticker}. It may be an invalid symbol or there could be network issues.")

            return df

        except Exception as e:
            import streamlit as st
            error_message = f"❌ Error fetching data for {ticker}: {e}"
            print(error_message)
            st.error(error_message)  # This shows up in the Streamlit UI
            return None

    def fetch_multiple_stocks(self, period='1y', interval='1d'):
        """Fetch data for multiple stocks"""
        data = {}
        for ticker in self.nifty50_tickers:
            data[ticker] = self.fetch_data(ticker, period, interval)
        return data

    def get_latest_data(self, ticker):
        """Get the most recent data for a ticker"""
        df = self.fetch_data(ticker, period='5d', interval='1d')
        if df is not None and not df.empty:
            return df.iloc[-1]
        return None

if __name__ == "__main__":
    # Example usage
    fetcher = NSEDataFetcher()
    reliance_data = fetcher.fetch_data('RELIANCE.NS')
    if reliance_data is not None:
        print("Reliance Data Sample:")
        print(reliance_data.tail())
