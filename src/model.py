import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class TradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """Prepare features for the model"""
        # Create features DataFrame
        features = pd.DataFrame(index=data.index)
        
        try:
            # Price and volume based features
            features['Returns'] = data['Close'].pct_change()
            features['Volume_Change'] = data['Volume'].pct_change()
            features['Price_Range'] = (data['High'] - data['Low']) / data['Close']
            
            # Trend features
            if all(col in data.columns for col in ['SMA_20', 'SMA_50', 'SMA_200']):
                features['SMA_Cross'] = data['SMA_20'] - data['SMA_50']
                features['Above_SMA200'] = (data['Close'] > data['SMA_200']).astype(int)
            
            if all(col in data.columns for col in ['EMA_20', 'EMA_50']):
                features['EMA_Cross'] = data['EMA_20'] - data['EMA_50']
            
            # Momentum features
            if 'RSI' in data.columns:
                features['RSI'] = data['RSI']
                features['RSI_Change'] = data['RSI'].diff()
                features['RSI_MA'] = data['RSI'].rolling(window=10).mean()
            
            # Volatility features
            if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                features['BB_Position'] = (data['Close'] - data['BB_Middle']) / (data['BB_Upper'] - data['BB_Lower'])
            
            features['Volatility'] = features['Returns'].rolling(window=20).std()
            
            if 'ATR' in data.columns:
                features['ATR'] = data['ATR']
            
            # MACD features
            if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
                features['MACD'] = data['MACD']
                features['MACD_Signal'] = data['MACD_Signal']
                features['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
            
            # Additional technical features
            for period in [5, 10, 20]:
                features[f'Return_{period}d'] = data['Close'].pct_change(period)
                features[f'Volume_{period}d'] = data['Volume'].pct_change(period)
                features[f'Volatility_{period}d'] = features['Returns'].rolling(window=period).std()
            
            # Additional indicators if available
            for indicator in ['ROC', 'Williams_R', 'Stoch_RSI', 'ADX', 'CCI']:
                if indicator in data.columns:
                    features[indicator] = data[indicator]
            
            # Target variable (1 if price goes up by more than 0.5%, 0 otherwise)
            future_returns = data['Close'].shift(-1) / data['Close'] - 1
            target = (future_returns > 0.005).astype(int)
            
            # Clean the data: handle NaN and infinite values
            features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            
            # Remove any remaining NaN values by dropping those rows
            features = features.dropna()
            target = target[features.index]
            
            return features, target
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            print("Columns available in data:", data.columns.tolist())
            raise

    def train(self, features, target):
        """Train the model using time series cross-validation"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        
        for train_idx, val_idx in tscv.split(scaled_features):
            # Split data
            X_train = scaled_features[train_idx]
            y_train = target.iloc[train_idx]
            X_val = scaled_features[val_idx]
            y_val = target.iloc[val_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            val_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            accuracies.append(accuracy)
        
        # Final training on full dataset
        self.model.fit(scaled_features, target)
        
        # Print model performance
        predictions = self.model.predict(scaled_features)
        print(f"\nModel Accuracy: {accuracy_score(target, predictions):.2f}\n")
        print("Classification Report:")
        print(classification_report(target, predictions))
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        return np.mean(accuracies)

    def predict(self, features):
        """Make predictions on new data"""
        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)
    
    def predict_proba(self, features):
        """Get prediction probabilities"""
        scaled_features = self.scaler.transform(features)
        return self.model.predict_proba(scaled_features)

if __name__ == "__main__":
    # Example usage
    from data_fetcher import NSEDataFetcher
    from indicators import TechnicalIndicators
    
    # Fetch data
    fetcher = NSEDataFetcher()
    data = fetcher.fetch_data('RELIANCE.NS', period='1y')
    
    if data is not None:
        # Calculate indicators
        indicators = TechnicalIndicators(data)
        data_with_indicators = indicators.add_all_indicators()
        
        # Create and train model
        trading_model = TradingModel()
        features, target = trading_model.prepare_features(data_with_indicators)
        
        if len(features) > 0:
            accuracy = trading_model.train(features, target)
            
            # Make prediction for the latest data point
            latest_features = features.iloc[[-1]]
            prediction = trading_model.predict(latest_features)
            probabilities = trading_model.predict_proba(latest_features)
            
            print(f"\nPrediction for next day: {'Up' if prediction[0] == 1 else 'Down'}")
            print(f"Probability: Up={probabilities[0][1]:.2f}, Down={probabilities[0][0]:.2f}")