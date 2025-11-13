"""
Portfolio Optimization Environment for Reinforcement Learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineer import FeatureEngineer

class PortfolioEnv(gym.Env):
    """
    RL Environment for Portfolio Optimization
    
    State: Market features for all assets
    Action: Portfolio weights (allocation percentages)
    Reward: Risk-adjusted returns (Sharpe ratio)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 60,
        reward_scaling: float = 100.0
    ):
        super().__init__()
        
        self.data = data
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Prepare data
        self._prepare_data()
        
        # Action space: portfolio weights (must sum to 1)
        # Using Box space with constraints
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # State space: features for each asset
        self.feature_dim = len(self.feature_engineer.feature_names)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets, self.feature_dim),
            dtype=np.float32
        )
        
        # Initialize
        self.reset()
    
    def _prepare_data(self):
        """Prepare and engineer features"""
        print("Preparing data and engineering features...")
        
        # Engineer features for all symbols
        self.data = self.feature_engineer.engineer_features(self.data)
        
        # Get dates
        self.dates = sorted(self.data['date'].unique())
        self.n_steps = len(self.dates) - self.lookback_window
        
        print(f"Data prepared: {len(self.dates)} dates, {self.n_steps} steps")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets  # Equal weights initially
        self.returns_history = []
        self.portfolio_values = [self.portfolio_value]
        
        # Get initial state
        state = self._get_state()
        
        return state, {}
    
    def _get_state(self) -> np.ndarray:
        """Get current state (features for all assets)"""
        current_date = self.dates[self.current_step]
        
        # Get data up to current date
        historical_data = self.data[self.data['date'] <= current_date]
        
        # Get features for each asset
        state_features = []
        for symbol in self.symbols:
            symbol_data = historical_data[historical_data['symbol'] == symbol]
            
            if len(symbol_data) > 0:
                # Use latest features
                latest_features = symbol_data[self.feature_engineer.feature_names].iloc[-1].values
                state_features.append(latest_features)
            else:
                # Pad with zeros if no data
                state_features.append(np.zeros(self.feature_dim))
        
        return np.array(state_features, dtype=np.float32)
    
    def _get_prices(self) -> np.ndarray:
        """Get current prices for all assets"""
        current_date = self.dates[self.current_step]
        prices = []
        
        for symbol in self.symbols:
            symbol_data = self.data[
                (self.data['symbol'] == symbol) & 
                (self.data['date'] == current_date)
            ]
            
            if len(symbol_data) > 0:
                prices.append(symbol_data['Close'].iloc[0])
            else:
                prices.append(0.0)
        
        return np.array(prices)
    
    def _get_returns(self) -> np.ndarray:
        """Get returns for all assets"""
        current_date = self.dates[self.current_step]
        prev_date = self.dates[self.current_step - 1] if self.current_step > 0 else current_date
        
        returns = []
        for symbol in self.symbols:
            current_data = self.data[
                (self.data['symbol'] == symbol) & 
                (self.data['date'] == current_date)
            ]
            prev_data = self.data[
                (self.data['symbol'] == symbol) & 
                (self.data['date'] == prev_date)
            ]
            
            if len(current_data) > 0 and len(prev_data) > 0:
                ret = (current_data['Close'].iloc[0] - prev_data['Close'].iloc[0]) / prev_data['Close'].iloc[0]
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return np.array(returns)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Normalize action to ensure weights sum to 1
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # Get current prices and returns
        prices = self._get_prices()
        returns = self._get_returns()
        
        # Calculate portfolio return
        portfolio_return = np.dot(action, returns)
        
        # Apply transaction costs (proportional to weight changes)
        weight_change = np.abs(action - self.portfolio_weights).sum()
        transaction_cost_penalty = weight_change * self.transaction_cost
        
        # Net return after transaction costs
        net_return = portfolio_return - transaction_cost_penalty
        
        # Update portfolio
        self.portfolio_weights = action
        self.portfolio_value *= (1 + net_return)
        self.returns_history.append(net_return)
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate reward (Sharpe ratio)
        if len(self.returns_history) > 1:
            mean_return = np.mean(self.returns_history)
            std_return = np.std(self.returns_history)
            sharpe_ratio = mean_return / (std_return + 1e-8) if std_return > 0 else 0
            reward = sharpe_ratio * self.reward_scaling
        else:
            reward = net_return * self.reward_scaling
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.dates) - 1
        truncated = False
        
        # Get next state
        next_state = self._get_state() if not done else self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'sharpe_ratio': reward / self.reward_scaling if len(self.returns_history) > 1 else 0,
            'current_step': self.current_step
        }
        
        return next_state, reward, done, truncated, info
    
    def render(self):
        """Render environment"""
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Portfolio Weights: {self.portfolio_weights}")
        if len(self.returns_history) > 0:
            print(f"Recent Return: {self.returns_history[-1]:.4f}")


if __name__ == "__main__":
    # Test environment
    import os
    
    data_dir = "data/prices"
    if os.path.exists(data_dir):
        # Load sample data
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')][:5]  # Use 5 symbols
        
        all_data = []
        symbols = []
        for file in files:
            symbol = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, file))
            df['symbol'] = symbol
            all_data.append(df)
            symbols.append(symbol)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data['date'] = pd.to_datetime(combined_data['date'])
            
            # Create environment
            env = PortfolioEnv(combined_data, symbols)
            
            # Test reset
            state, info = env.reset()
            print(f"State shape: {state.shape}")
            print(f"Action space: {env.action_space}")
            
            # Test step
            action = np.ones(len(symbols)) / len(symbols)  # Equal weights
            next_state, reward, done, truncated, info = env.step(action)
            
            print(f"\nReward: {reward:.4f}")
            print(f"Info: {info}")

