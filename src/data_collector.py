"""
Comprehensive Market Data Collector
Collects all possible market data for portfolio optimization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict
import time

class MarketDataCollector:
    """
    Collects comprehensive market data including:
    - Price data (OHLCV)
    - Technical indicators
    - Market metrics
    - Sector/industry data
    - Market sentiment indicators
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Comprehensive list of stocks across ALL major sectors for maximum diversity
        self.stocks = {
            # Technology (14 stocks)
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'CSCO'],
            
            # Financial Services (14 stocks)
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'COF', 'USB', 'PNC', 'TFC', 'BK'],
            
            # Healthcare (14 stocks)
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'ABBV', 'MRK', 'LLY', 'DHR', 'BMY', 'AMGN', 'GILD', 'CVS', 'CI'],
            
            # Consumer Discretionary (14 stocks)
            'consumer_disc': ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'ROST', 'DG', 'COST', 'AMZN', 'TSLA', 'NKE'],
            
            # Consumer Staples (13 stocks)
            'consumer_staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'CLX', 'CHD', 'GIS', 'CPB', 'SJM', 'HRL'],
            
            # Industrial (14 stocks)
            'industrial': ['BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'RTX', 'LMT', 'NOC', 'GD', 'EMR', 'ETN', 'ITW', 'DE'],
            
            # Energy (13 stocks)
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'HAL', 'OXY', 'DVN', 'FANG', 'MRO'],
            
            # Materials (13 stocks)
            'materials': ['LIN', 'APD', 'ECL', 'SHW', 'DD', 'PPG', 'NEM', 'FCX', 'VALE', 'RIO', 'BHP', 'NUE', 'STLD'],
            
            # Real Estate/REITs (13 stocks)
            'reit': ['AMT', 'PLD', 'EQIX', 'PSA', 'WELL', 'SPG', 'O', 'VICI', 'EXPI', 'CBRE', 'AVB', 'EQR', 'MAA'],
            
            # Utilities (13 stocks)
            'utilities': ['NEE', 'DUK', 'SO', 'AEP', 'SRE', 'D', 'EXC', 'XEL', 'ED', 'PEG', 'ES', 'EIX', 'WEC'],
            
            # Communication Services (12 stocks)
            'communication': ['VZ', 'T', 'CMCSA', 'DIS', 'NFLX', 'GOOGL', 'META', 'TMUS', 'CHTR', 'EA', 'TTWO', 'ATVI'],
            
            # ETFs for Diversification (13 ETFs)
            'etfs': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'GLD', 'SLV', 'USO']
        }
        
        # Flatten to single list
        self.all_symbols = [symbol for symbols in self.stocks.values() for symbol in symbols]
        
    def collect_price_data(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """Collect OHLCV price data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"Warning: No data for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['symbol'] = symbol
            data['date'] = data.index
            
            return data
        except Exception as e:
            print(f"Error collecting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def collect_fundamental_data(self, symbol: str) -> Dict:
        """Collect fundamental data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamental_data = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'sector': info.get('sector', None),
                'industry': info.get('industry', None),
            }
            
            return fundamental_data
        except Exception as e:
            print(f"Error collecting fundamentals for {symbol}: {e}")
            return {'symbol': symbol}
    
    def collect_all_data(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        """Collect all market data"""
        print(f"Collecting data for {len(self.all_symbols)} symbols...")
        
        all_price_data = {}
        all_fundamental_data = {}
        
        for i, symbol in enumerate(self.all_symbols):
            print(f"[{i+1}/{len(self.all_symbols)}] Collecting {symbol}...")
            
            # Price data
            price_data = self.collect_price_data(symbol)
            if not price_data.empty:
                all_price_data[symbol] = price_data
            
            # Fundamental data
            fundamental_data = self.collect_fundamental_data(symbol)
            all_fundamental_data[symbol] = fundamental_data
            
            # Rate limiting to avoid API limits
            time.sleep(0.5)
        
        if save:
            self.save_data(all_price_data, all_fundamental_data)
        
        return {
            'price_data': all_price_data,
            'fundamental_data': all_fundamental_data
        }
    
    def save_data(self, price_data: Dict[str, pd.DataFrame], fundamental_data: Dict[str, Dict]):
        """Save collected data"""
        # Save price data
        price_dir = os.path.join(self.data_dir, 'prices')
        os.makedirs(price_dir, exist_ok=True)
        
        for symbol, data in price_data.items():
            filepath = os.path.join(price_dir, f"{symbol}.csv")
            data.to_csv(filepath)
        
        # Save fundamental data
        fundamental_file = os.path.join(self.data_dir, 'fundamentals.json')
        with open(fundamental_file, 'w') as f:
            json.dump(fundamental_data, f, indent=2)
        
        # Create combined dataset
        combined_file = os.path.join(self.data_dir, 'combined_data.csv')
        all_data = []
        for symbol, data in price_data.items():
            all_data.append(data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(combined_file, index=False)
        
        print(f"Data saved to {self.data_dir}")
    
    def update_data(self):
        """Update existing data with latest market data"""
        print("Updating market data...")
        self.collect_all_data(save=True)


if __name__ == "__main__":
    collector = MarketDataCollector()
    
    # Collect all data
    data = collector.collect_all_data(save=True)
    
    print("\nData collection complete!")
    print(f"Collected price data for {len(data['price_data'])} symbols")
    print(f"Collected fundamental data for {len(data['fundamental_data'])} symbols")

