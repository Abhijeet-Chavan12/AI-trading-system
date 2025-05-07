import pandas as pd
import numpy as np
from data_fetcher import NSEDataFetcher
from indicators import TechnicalIndicators
from model import TradingModel

class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = 0
        self.trades = []
        self.portfolio_values = []

    def calculate_position_size(self, capital, price):
        """Calculate number of shares to buy/sell"""
        return int(capital * 0.95 / price)  # Using 95% of capital, adjust as needed

    def backtest(self, data, model_predictions, stop_loss_pct=0.02, take_profit_pct=0.04):
        """
        Run backtest on historical data
        params:
            data: DataFrame with OHLCV data
            model_predictions: Array of 1 (buy) or 0 (sell) signals
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.capital = self.initial_capital
        self.positions = 0
        self.trades = []
        self.portfolio_values = []
        
        # Convert predictions to numpy array if it's not already
        predictions = np.array(model_predictions)
        
        # Make sure we only use as many predictions as we have data points
        n_periods = min(len(data)-1, len(predictions))
        
        for i in range(n_periods):
            current_price = data['Close'].iloc[i]
            next_price = data['Close'].iloc[i+1]
            date = data.index[i]
            
            # Record daily portfolio value
            portfolio_value = self.capital + (self.positions * current_price)
            self.portfolio_values.append({
                'Date': date,
                'Portfolio_Value': portfolio_value
            })

            # Check stop loss and take profit for existing position
            if self.positions > 0:
                entry_price = self.trades[-1]['Entry_Price']
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                    # Close position
                    self.capital += self.positions * current_price
                    self.trades[-1].update({
                        'Exit_Date': date,
                        'Exit_Price': current_price,
                        'PnL': self.positions * (current_price - entry_price)
                    })
                    self.positions = 0
                    continue

            # Generate new positions based on model predictions
            if self.positions == 0:  # Only enter new positions if not already in one
                if predictions[i] == 1:  # Buy signal
                    self.positions = self.calculate_position_size(self.capital, current_price)
                    self.capital -= self.positions * current_price
                    self.trades.append({
                        'Entry_Date': date,
                        'Entry_Price': current_price,
                        'Positions': self.positions,
                        'Type': 'Long'
                    })

        # Close any remaining position at the end
        if self.positions > 0:
            final_price = data['Close'].iloc[-1]
            self.capital += self.positions * final_price
            self.trades[-1].update({
                'Exit_Date': data.index[-1],
                'Exit_Price': final_price,
                'PnL': self.positions * (final_price - self.trades[-1]['Entry_Price'])
            })

        return self.generate_statistics()

    def generate_statistics(self):
        """Generate backtest statistics"""
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) == 0:
            return {
                'Total_Returns': 0,
                'Total_Trades': 0,
                'Win_Rate': 0,
                'Avg_Profit_Per_Trade': 0,
                'Max_Drawdown': 0
            }

        # Calculate statistics
        total_pnl = sum(trade.get('PnL', 0) for trade in self.trades)
        total_returns = (self.capital - self.initial_capital) / self.initial_capital
        winning_trades = len([trade for trade in self.trades if trade.get('PnL', 0) > 0])
        
        # Calculate drawdown
        if len(portfolio_df) > 0:
            portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'].expanding().max() - 
                                      portfolio_df['Portfolio_Value']) / portfolio_df['Portfolio_Value'].expanding().max()
            max_drawdown = portfolio_df['Drawdown'].max()
        else:
            max_drawdown = 0
        
        stats = {
            'Total_Returns': total_returns,
            'Total_Trades': len(self.trades),
            'Win_Rate': winning_trades / len(self.trades) if len(self.trades) > 0 else 0,
            'Avg_Profit_Per_Trade': total_pnl / len(self.trades) if len(self.trades) > 0 else 0,
            'Max_Drawdown': max_drawdown
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    fetcher = NSEDataFetcher()
    data = fetcher.fetch_data('RELIANCE.NS', period='1y')
    
    if data is not None:
        # Calculate indicators
        indicators = TechnicalIndicators(data)
        data_with_indicators = indicators.add_all_indicators()
        
        # Train model and get predictions
        trading_model = TradingModel()
        features, target = trading_model.prepare_features(data_with_indicators)
        
        if len(features) > 0:
            trading_model.train(features, target)
            predictions = trading_model.predict(features)
            
            # Run backtest
            backtester = Backtester(initial_capital=100000)
            stats = backtester.backtest(data_with_indicators, predictions)
            
            print("\nBacktest Results:")
            for key, value in stats.items():
                print(f"{key}: {value:.2%}" if "Rate" in key or "Returns" in key else f"{key}: {value:.2f}")