"""
Feature Engineering Pipeline
Extracts comprehensive features from market data for RL training
Now includes news sentiment features
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import ta
import os
import json

class FeatureEngineer:
    """
    Creates features for RL agent including:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Momentum indicators
    - Volatility measures
    - Price patterns
    - Market regime indicators
    - News sentiment features
    """
    
    def __init__(self, data_dir: str = "data"):
        self.feature_names = []
        self.data_dir = data_dir
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        if df.empty or len(df) < 20:
            return df
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # Moving Averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Price relative to MAs
        df['price_sma5_ratio'] = df['Close'] / df['sma_5']
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['price_sma50_ratio'] = df['Close'] / df['sma_50']
        
        # ATR (Average True Range) for volatility
        df['atr'] = ta.volatility.AverageTrueRange(
            df['High'], df['Low'], df['Close']
        ).average_true_range()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.ADXIndicator(
            df['High'], df['Low'], df['Close']
        ).adx()
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        if df.empty:
            return df
        
        # Returns
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # Momentum
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Rate of change
        df['roc_5'] = ta.momentum.ROCIndicator(df['Close'], window=5).roc()
        df['roc_10'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        if df.empty:
            return df
        
        # Rolling volatility
        df['volatility_5'] = df['return_1d'].rolling(window=5).std()
        df['volatility_10'] = df['return_1d'].rolling(window=10).std()
        df['volatility_20'] = df['return_1d'].rolling(window=20).std()
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # High-Low range
        df['hl_range'] = (df['High'] - df['Low']) / df['Close']
        df['hl_range_5'] = df['hl_range'].rolling(window=5).mean()
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if df.empty or 'Volume' not in df.columns:
            return df
        
        # Volume moving averages
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # On Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            df['Close'], df['Volume']
        ).on_balance_volume()
        
        # Volume Price Trend
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(
            df['Close'], df['Volume']
        ).volume_price_trend()
        
        return df
    
    def add_news_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add news sentiment features"""
        if df.empty or 'symbol' not in df.columns:
            return df
        
        # Load sentiment scores
        sentiment_file = os.path.join(self.data_dir, 'news', 'latest_news.csv')
        
        if not os.path.exists(sentiment_file):
            # No sentiment data, add zeros
            df['news_sentiment'] = 0.0
            df['news_polarity'] = 0.0
            df['news_count'] = 0.0
            return df
        
        try:
            news_df = pd.read_csv(sentiment_file)
            
            # Create sentiment mapping by symbol
            sentiment_map = {}
            for symbol in df['symbol'].unique():
                symbol_news = news_df[news_df['mentioned_symbols'].str.contains(symbol, na=False)]
                
                if len(symbol_news) > 0:
                    sentiment_map[symbol] = {
                        'sentiment': symbol_news['sentiment_polarity'].mean(),
                        'polarity': symbol_news['sentiment_polarity'].mean(),
                        'count': len(symbol_news)
                    }
                else:
                    sentiment_map[symbol] = {
                        'sentiment': 0.0,
                        'polarity': 0.0,
                        'count': 0
                    }
            
            # Add sentiment features
            df['news_sentiment'] = df['symbol'].map(lambda x: sentiment_map.get(x, {}).get('sentiment', 0.0))
            df['news_polarity'] = df['symbol'].map(lambda x: sentiment_map.get(x, {}).get('polarity', 0.0))
            df['news_count'] = df['symbol'].map(lambda x: sentiment_map.get(x, {}).get('count', 0))
            
        except Exception as e:
            print(f"Error loading sentiment data: {e}")
            df['news_sentiment'] = 0.0
            df['news_polarity'] = 0.0
            df['news_count'] = 0.0
        
        return df
    
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators"""
        if df.empty:
            return df
        
        # Trend strength
        df['trend_strength'] = abs(df['adx']) if 'adx' in df.columns else 0
        
        # Market regime (trending vs ranging)
        if 'volatility_20' in df.columns and 'return_20d' in df.columns:
            df['regime'] = np.where(
                (df['volatility_20'] > df['volatility_20'].quantile(0.7)) & 
                (abs(df['return_20d']) > df['return_20d'].abs().quantile(0.7)),
                'high_volatility_trending',
                np.where(
                    df['volatility_20'] > df['volatility_20'].quantile(0.7),
                    'high_volatility_ranging',
                    'low_volatility'
                )
            )
        
        # Price position in range
        if 'High' in df.columns and 'Low' in df.columns:
            df['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                                  (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering"""
        if df.empty:
            return df
        
        df = self.add_technical_indicators(df)
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)
        df = self.add_volume_features(df)
        df = self.add_news_sentiment_features(df)  # Add news sentiment
        df = self.add_market_regime_features(df)
        
        # Drop rows with NaN values (from rolling calculations)
        df = df.dropna()
        
        # Store feature names
        self.feature_names = [col for col in df.columns 
                             if col not in ['symbol', 'date', 'Open', 'High', 'Low', 'Close', 'Volume', 'regime']]
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame, symbols: List[str]) -> np.ndarray:
        """Create feature matrix for RL agent"""
        features = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            if symbol_data.empty:
                continue
            
            symbol_data = self.engineer_features(symbol_data)
            
            # Use latest features
            if len(symbol_data) > 0:
                latest_features = symbol_data[self.feature_names].iloc[-1].values
                features.append(latest_features)
            else:
                # Pad with zeros if no data
                features.append(np.zeros(len(self.feature_names)))
        
        return np.array(features)


if __name__ == "__main__":
    import os
    
    # Load sample data
    data_dir = "data/prices"
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        if files:
            sample_file = os.path.join(data_dir, files[0])
            df = pd.read_csv(sample_file)
            df['date'] = pd.to_datetime(df['date'])
            
            engineer = FeatureEngineer()
            df_features = engineer.engineer_features(df)
            
            print(f"Original columns: {len(df.columns)}")
            print(f"Features added: {len(engineer.feature_names)}")
            print(f"\nFeature names: {engineer.feature_names[:10]}...")
            print(f"\nSample features:\n{df_features[engineer.feature_names].head()}")
