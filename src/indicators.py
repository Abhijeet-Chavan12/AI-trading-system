import pandas as pd
import numpy as np
import ta

class TechnicalIndicators:
    def __init__(self, data):
        """
        Initialize with a pandas DataFrame containing OHLCV data
        """
        self.data = data.copy()
        # Fill any missing values in the input data
        self.data = self.data.ffill().bfill()
        self.indicators = {}

    def add_moving_averages(self, short_window=20, long_window=50):
        """Add SMA and EMA indicators"""
        # Short-term SMA
        sma_short = ta.trend.sma_indicator(close=self.data['Close'], window=short_window)
        self.data[f'SMA_{short_window}'] = sma_short
        
        # Long-term SMA
        sma_long = ta.trend.sma_indicator(close=self.data['Close'], window=long_window)
        self.data[f'SMA_{long_window}'] = sma_long
        
        # EMA
        ema_short = ta.trend.ema_indicator(close=self.data['Close'], window=short_window)
        self.data[f'EMA_{short_window}'] = ema_short

    def add_rsi(self, window=14):
        """Add RSI indicator"""
        rsi = ta.momentum.rsi(close=self.data['Close'], window=window)
        self.data['RSI'] = rsi

    def add_macd(self, slow=26, fast=12, signal=9):
        """Add MACD indicator"""
        macd = ta.trend.MACD(close=self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Hist'] = macd.macd_diff()

    def add_bollinger_bands(self, window=20, window_dev=2):
        """Add Bollinger Bands"""
        bollinger = ta.volatility.BollingerBands(close=self.data['Close'])
        self.data['BB_Upper'] = bollinger.bollinger_hband()
        self.data['BB_Middle'] = bollinger.bollinger_mavg()
        self.data['BB_Lower'] = bollinger.bollinger_lband()

    def add_all_indicators(self):
        """Add all technical indicators to the dataset"""
        df = self.data.copy()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # Exponential Moving Averages
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()
        
        # Average True Range
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Additional Momentum Indicators
        df['ROC'] = ta.momentum.roc(df['Close'])  # Rate of Change
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        df['Stoch_RSI'] = ta.momentum.stochrsi(df['Close'])
        
        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['ADI'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Trend Indicators
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # Clean the data: handle NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        return df

    def generate_signals(self):
        """Generate trading signals based on indicators"""
        signals = pd.DataFrame(index=self.data.index)
        
        # RSI signals
        signals['RSI_Signal'] = 0
        signals.loc[self.data['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold
        signals.loc[self.data['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought
        
        # MACD signals
        signals['MACD_Signal'] = 0
        signals.loc[self.data['MACD'] > self.data['MACD_Signal'], 'MACD_Signal'] = 1
        signals.loc[self.data['MACD'] < self.data['MACD_Signal'], 'MACD_Signal'] = -1
        
        # Bollinger Bands signals
        signals['BB_Signal'] = 0
        signals.loc[self.data['Close'] < self.data['BB_Lower'], 'BB_Signal'] = 1
        signals.loc[self.data['Close'] > self.data['BB_Upper'], 'BB_Signal'] = -1
        
        return signals

if __name__ == "__main__":
    # Example usage with data from data_fetcher
    from data_fetcher import NSEDataFetcher
    
    fetcher = NSEDataFetcher()
    data = fetcher.fetch_data('RELIANCE.NS')
    
    if data is not None:
        indicators = TechnicalIndicators(data)
        indicators.add_all_indicators()
        signals = indicators.generate_signals()
        print("Generated Signals Sample:")
        print(signals.tail())