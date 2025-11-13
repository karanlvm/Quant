"""
Training Pipeline for RL Portfolio Optimization Agent
"""

import os
import pandas as pd
import numpy as np
from data_collector import MarketDataCollector
from portfolio_env import PortfolioEnv
from rl_agent import PortfolioRLAgent
from feature_engineer import FeatureEngineer
from datetime import datetime
import json
import argparse

class PortfolioTrainer:
    """
    Main training pipeline that:
    1. Collects/updates market data
    2. Collects news and analyzes sentiment
    3. Prepares environment
    4. Trains RL agent
    5. Evaluates and saves model
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        symbols: list = None,
        initial_balance: float = 100000.0
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_balance = initial_balance
        
        # Default symbols if not provided - use all available
        if symbols is None:
            # Will load from collected data
            self.symbols = None
        else:
            self.symbols = symbols
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
    
    def collect_data(self, update: bool = False):
        """Collect or update market data"""
        print("=" * 50)
        print("STEP 1: Collecting Market Data")
        print("=" * 50)
        
        collector = MarketDataCollector(data_dir=self.data_dir)
        
        if update:
            collector.update_data()
        else:
            collector.collect_all_data(save=True)
        
        print("Data collection complete!\n")
    
    def prepare_environment(self) -> PortfolioEnv:
        """Prepare RL environment with data"""
        print("=" * 50)
        print("STEP 2: Preparing Environment")
        print("=" * 50)
        
        # Load price data
        price_dir = os.path.join(self.data_dir, 'prices')
        
        if not os.path.exists(price_dir):
            raise ValueError("No data directory found! Please collect data first.")
        
        files = [f for f in os.listdir(price_dir) if f.endswith('.csv')]
        
        if not files:
            raise ValueError("No data files found! Please collect data first.")
        
        # Use all available symbols or limit for performance
        if self.symbols is None:
            self.symbols = [f.replace('.csv', '') for f in files][:30]  # Limit to 30 for performance
        
        all_data = []
        
        for symbol in self.symbols:
            filepath = os.path.join(price_dir, f"{symbol}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['symbol'] = symbol
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                else:
                    # Try to use index if it's datetime
                    df.reset_index(inplace=True)
                    if 'index' in df.columns:
                        df['date'] = pd.to_datetime(df['index'])
                all_data.append(df)
            else:
                print(f"Warning: No data file for {symbol}")
        
        if not all_data:
            raise ValueError("No data files found! Please collect data first.")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"Loaded data for {len(self.symbols)} symbols")
        print(f"Total data points: {len(combined_data)}")
        if 'date' in combined_data.columns:
            print(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}\n")
        
        # Create environment
        env = PortfolioEnv(
            data=combined_data,
            symbols=self.symbols,
            initial_balance=self.initial_balance
        )
        
        return env
    
    def train_agent(
        self,
        env: PortfolioEnv,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        save_freq: int = 10000,
        load_existing: bool = False
    ):
        """Train the RL agent"""
        print("=" * 50)
        print("STEP 3: Training RL Agent")
        print("=" * 50)
        
        # Create agent
        agent = PortfolioRLAgent(
            env=env,
            model_dir=self.model_dir,
            learning_rate=learning_rate
        )
        
        # Load existing model if requested
        if load_existing:
            model_path = os.path.join(self.model_dir, "final_model.zip")
            if os.path.exists(model_path):
                print(f"Loading existing model from {model_path}...")
                agent.load_model(model_path)
        
        # Train
        callback = agent.train(
            total_timesteps=total_timesteps,
            save_freq=save_freq
        )
        
        print("\nTraining complete!\n")
        
        return agent, callback
    
    def evaluate_agent(self, agent: PortfolioRLAgent, n_episodes: int = 10):
        """Evaluate trained agent"""
        print("=" * 50)
        print("STEP 4: Evaluating Agent")
        print("=" * 50)
        
        results = agent.evaluate(n_episodes=n_episodes)
        
        print(f"Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        print(f"Mean Episode Length: {results['mean_length']:.0f}")
        if results['mean_final_value']:
            print(f"Mean Final Portfolio Value: ${results['mean_final_value']:.2f}")
            print(f"Return: {((results['mean_final_value'] / self.initial_balance) - 1) * 100:.2f}%")
        
        # Save evaluation results
        results_file = os.path.join(self.model_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}\n")
        
        return results
    
    def run_full_training(
        self,
        collect_data: bool = True,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        update: bool = False
    ):
        """Run complete training pipeline"""
        print("\n" + "=" * 50)
        print("PORTFOLIO OPTIMIZATION RL TRAINING")
        print("=" * 50 + "\n")
        
        # Step 1: Collect data
        if collect_data:
            self.collect_data(update=update)
        
        # Step 1.5: Collect news and analyze sentiment
        try:
            from news_collector import NewsCollector
            print("=" * 50)
            print("STEP 1.5: Collecting News and Analyzing Sentiment")
            print("=" * 50)
            news_collector = NewsCollector()
            news_collector.collect_market_news()
            print("News collection complete!\n")
        except Exception as e:
            print(f"Warning: News collection failed: {e}\n")
        
        # Step 2: Prepare environment
        env = self.prepare_environment()
        
        # Step 3: Train agent
        agent, callback = self.train_agent(
            env=env,
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            load_existing=update
        )
        
        # Step 4: Evaluate
        results = self.evaluate_agent(agent)
        
        print("=" * 50)
        print("TRAINING PIPELINE COMPLETE")
        print("=" * 50)
        
        return agent, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL Portfolio Optimization Agent')
    parser.add_argument('--timesteps', type=int, default=50000, help='Number of training timesteps')
    parser.add_argument('--update', action='store_true', help='Update existing model')
    parser.add_argument('--no-data', action='store_true', help='Skip data collection')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = PortfolioTrainer(
        data_dir="data",
        model_dir="models"
    )
    
    # Run training
    agent, results = trainer.run_full_training(
        collect_data=not args.no_data,
        total_timesteps=args.timesteps,
        learning_rate=3e-4,
        update=args.update
    )
    
    print("\nTraining completed successfully!")
