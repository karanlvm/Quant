"""
Generate Optimal Portfolio using Trained RL Model
"""

import os
import pandas as pd
import numpy as np
from portfolio_env import PortfolioEnv
from rl_agent import PortfolioRLAgent
from data_collector import MarketDataCollector
import json

class PortfolioGenerator:
    """
    Uses trained RL model to generate optimal portfolio allocations
    """
    
    def __init__(self, model_path: str, data_dir: str = "data"):
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Load agent
        self.agent = None
        self.env = None
    
    def load_model(self, symbols: list):
        """Load trained model and prepare environment"""
        print("Loading model and preparing environment...")
        
        # Load price data
        price_dir = os.path.join(self.data_dir, 'prices')
        all_data = []
        
        for symbol in symbols:
            filepath = os.path.join(price_dir, f"{symbol}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['symbol'] = symbol
                df['date'] = pd.to_datetime(df['date'])
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data files found!")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Create environment
        self.env = PortfolioEnv(
            data=combined_data,
            symbols=symbols,
            initial_balance=100000.0
        )
        
        # Load agent
        self.agent = PortfolioRLAgent(env=self.env, model_dir=os.path.dirname(self.model_path))
        self.agent.load_model(self.model_path)
        
        print("Model loaded successfully!")
    
    def generate_portfolio(self, symbols: list) -> dict:
        """Generate optimal portfolio allocation"""
        if self.agent is None:
            self.load_model(symbols)
        
        # Reset environment
        state, _ = self.env.reset()
        
        # Get optimal action (portfolio weights)
        action = self.agent.predict(state, deterministic=True)
        
        # Normalize to ensure weights sum to 1
        action = action / (action.sum() + 1e-8)
        
        # Create portfolio allocation
        portfolio = {}
        for i, symbol in enumerate(symbols):
            portfolio[symbol] = {
                'allocation': float(action[i]),
                'allocation_percent': float(action[i] * 100)
            }
        
        # Sort by allocation
        portfolio = dict(sorted(portfolio.items(), key=lambda x: x[1]['allocation'], reverse=True))
        
        return portfolio
    
    def simulate_portfolio(self, symbols: list, n_steps: int = 100) -> dict:
        """Simulate portfolio performance"""
        if self.agent is None:
            self.load_model(symbols)
        
        # Reset environment
        state, _ = self.env.reset()
        
        portfolio_values = []
        actions_history = []
        
        for step in range(n_steps):
            # Get action
            action = self.agent.predict(state, deterministic=True)
            actions_history.append(action.copy())
            
            # Step environment
            state, reward, done, truncated, info = self.env.step(action)
            
            portfolio_values.append(info['portfolio_value'])
            
            if done:
                break
        
        results = {
            'initial_value': self.env.initial_balance,
            'final_value': portfolio_values[-1] if portfolio_values else self.env.initial_balance,
            'return': ((portfolio_values[-1] / self.env.initial_balance) - 1) * 100 if portfolio_values else 0,
            'portfolio_values': portfolio_values,
            'final_allocation': actions_history[-1] if actions_history else None
        }
        
        return results


if __name__ == "__main__":
    import sys
    
    # Check if model exists
    model_path = "models/final_model.zip"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using trainer.py")
        sys.exit(1)
    
    # Symbols to use
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Generate portfolio
    generator = PortfolioGenerator(model_path=model_path)
    portfolio = generator.generate_portfolio(symbols)
    
    print("\n" + "=" * 50)
    print("OPTIMAL PORTFOLIO ALLOCATION")
    print("=" * 50)
    
    for symbol, allocation in portfolio.items():
        print(f"{symbol}: {allocation['allocation_percent']:.2f}%")
    
    # Simulate performance
    print("\n" + "=" * 50)
    print("PORTFOLIO SIMULATION")
    print("=" * 50)
    
    results = generator.simulate_portfolio(symbols, n_steps=100)
    print(f"Initial Value: ${results['initial_value']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Return: {results['return']:.2f}%")
    
    # Save portfolio
    portfolio_file = "portfolio_allocation.json"
    with open(portfolio_file, 'w') as f:
        json.dump(portfolio, f, indent=2)
    
    print(f"\nPortfolio saved to {portfolio_file}")

