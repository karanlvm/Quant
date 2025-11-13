"""
Reinforcement Learning Agent for Portfolio Optimization
Uses PPO (Proximal Policy Optimization) algorithm
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import os
from typing import Optional, Dict

class PortfolioCallback(BaseCallback):
    """Custom callback to track portfolio performance during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.rewards = []
        self.sharpe_ratios = []
    
    def _on_step(self) -> bool:
        # Track metrics from info dict
        if 'portfolio_value' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]
            self.portfolio_values.append(info.get('portfolio_value', 0))
            self.sharpe_ratios.append(info.get('sharpe_ratio', 0))
        
        self.rewards.append(self.locals.get('rewards', [0])[0])
        return True
    
    def _on_training_end(self) -> None:
        print(f"\nTraining completed!")
        print(f"Final portfolio value: ${self.portfolio_values[-1]:.2f}" if self.portfolio_values else "")
        print(f"Average reward: {np.mean(self.rewards):.4f}")
        print(f"Final Sharpe ratio: {self.sharpe_ratios[-1]:.4f}" if self.sharpe_ratios else "")


class PortfolioRLAgent:
    """
    RL Agent for Portfolio Optimization using PPO
    
    PPO is chosen because:
    - Stable training
    - Good sample efficiency
    - Handles continuous action spaces well
    - Works well for portfolio optimization problems
    """
    
    def __init__(
        self,
        env,
        model_dir: str = "models",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5
    ):
        self.env = env
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=1,
            tensorboard_log=os.path.join(model_dir, "tensorboard")
        )
    
    def train(self, total_timesteps: int = 100000, save_freq: int = 10000):
        """Train the RL agent"""
        print(f"Training agent for {total_timesteps} timesteps...")
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self.model_dir,
            name_prefix="portfolio_model"
        )
        
        portfolio_callback = PortfolioCallback()
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, portfolio_callback],
            progress_bar=True
        )
        
        # Save final model
        self.save_model("final_model")
        
        return portfolio_callback
    
    def predict(self, state: Optional[np.ndarray] = None, deterministic: bool = True) -> np.ndarray:
        """Predict action (portfolio weights)"""
        if state is None:
            action, _ = self.model.predict(
                self.vec_env.reset()[0],
                deterministic=deterministic
            )
        else:
            action, _ = self.model.predict(state, deterministic=deterministic)
        
        return action
    
    def save_model(self, name: str = "portfolio_model"):
        """Save the trained model"""
        path = os.path.join(self.model_dir, f"{name}.zip")
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """Evaluate the agent"""
        episode_rewards = []
        episode_lengths = []
        final_portfolio_values = []
        
        for episode in range(n_episodes):
            obs, _ = self.vec_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.vec_env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if info and 'portfolio_value' in info[0]:
                final_portfolio_values.append(info[0]['portfolio_value'])
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_final_value': np.mean(final_portfolio_values) if final_portfolio_values else None
        }
        
        return results


if __name__ == "__main__":
    from portfolio_env import PortfolioEnv
    import pandas as pd
    
    # This would be used with actual data
    print("RL Agent module loaded. Use with PortfolioEnv for training.")

