import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from data_fetcher import NSEDataFetcher
from indicators import TechnicalIndicators
from model import TradingModel
from backtest import Backtester

def analyze_stock(symbol, initial_capital=100000):
    print(f"\nAnalyzing {symbol}...")
    fetcher = NSEDataFetcher()
    data = fetcher.fetch_data(symbol, period='1y')
    
    if data is not None:
        # Calculate indicators
        indicators = TechnicalIndicators(data)
        data_with_indicators = indicators.add_all_indicators()
        
        # Train model and get predictions
        model = TradingModel()
        features, target = model.prepare_features(data_with_indicators)
        
        if len(features) > 0:
            # Train the model
            model.train(features, target)
            
            # Get latest prediction
            latest_features = features.iloc[[-1]]
            prediction = model.predict(latest_features)
            probabilities = model.predict_proba(latest_features)
            
            print(f"\nPrediction for {symbol}:")
            print(f"Signal: {'BUY' if prediction[0] == 1 else 'SELL'}")
            print(f"Confidence: Up={probabilities[0][1]:.2f}, Down={probabilities[0][0]:.2f}")
            
            # Run backtest
            backtester = Backtester(initial_capital=initial_capital)
            stats = backtester.backtest(data_with_indicators, model.predict(features))
            
            print("\nBacktest Results:")
            for key, value in stats.items():
                print(f"{key}: {value:.2%}" if "Rate" in key or "Returns" in key else f"{key}: {value:.2f}")

def main():
    # Test with major NSE stocks
    stocks = [
        'RELIANCE.NS',  # Reliance Industries
        'TCS.NS',       # Tata Consultancy Services
        'HDFCBANK.NS',  # HDFC Bank
        'INFY.NS',      # Infosys
        'SBIN.NS'
    ]
    
    for stock in stocks:
        analyze_stock(stock)

if __name__ == "__main__":
    main()