import os
import sys
from pathlib import Path

# Get the absolute path to the project directory
project_dir = str(Path(__file__).resolve().parent)
src_dir = os.path.join(project_dir, 'src')

# Add both directories to Python path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_fetcher import NSEDataFetcher
from indicators import TechnicalIndicators
from model import TradingModel
from backtest import Backtester

def analyze_stock(symbol, initial_capital=100000):
    print(f"\nAnalyzing {symbol}...")
    
    # Initialize components
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
            model.train(features, target)
            latest_features = features.iloc[[-1]]
            prediction = model.predict(latest_features)
            probabilities = model.predict_proba(latest_features)
            
            print(f"\nPrediction for {symbol}:")
            print(f"Signal: {'BUY' if prediction[0] == 1 else 'SELL'}")
            print(f"Confidence: Up={probabilities[0][1]:.2f}, Down={probabilities[0][0]:.2f}")
            
            backtester = Backtester(initial_capital=initial_capital)
            stats = backtester.backtest(data_with_indicators, model.predict(features))
            
            print("\nBacktest Results:")
            for key, value in stats.items():
                print(f"{key}: {value:.2%}" if "Rate" in key or "Returns" in key else f"{key}: {value:.2f}")
            
            return stats
    return None

if __name__ == "__main__":
    # Test with some major NSE stocks
    stocks = [
        'RELIANCE.NS',
        'TCS.NS',
        'HDFCBANK.NS',
        'INFY.NS'
    ]
    
    for stock in stocks:
        analyze_stock(stock)