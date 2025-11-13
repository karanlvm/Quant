# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Step 1: Collect Market Data

Collect comprehensive market data for training:

```bash
python src/data_collector.py
```

This will:
- Collect price data (OHLCV) for 50+ stocks across sectors
- Gather fundamental data (P/E, market cap, etc.)
- Save data to `data/` directory

## Step 2: Train the RL Agent

Train the reinforcement learning agent:

```bash
python src/trainer.py
```

This will:
- Load and prepare market data
- Engineer features (technical indicators, momentum, volatility)
- Train PPO agent to learn optimal portfolio allocations
- Save trained model to `models/` directory

**Training Parameters:**
- `total_timesteps`: Number of training steps (default: 100000)
- `learning_rate`: Learning rate (default: 3e-4)
- Adjust in `trainer.py` if needed

## Step 3: Generate Optimal Portfolio

Use the trained model to generate portfolio allocations:

```bash
python src/generate_portfolio.py
```

This will:
- Load the trained model
- Generate optimal portfolio weights
- Simulate portfolio performance
- Save allocation to `portfolio_allocation.json`

## Continuous Learning

To continuously improve the model:

1. **Update Data Regularly:**
```python
from src.data_collector import MarketDataCollector
collector = MarketDataCollector()
collector.update_data()  # Updates existing data
```

2. **Continue Training:**
```python
from src.trainer import PortfolioTrainer
trainer = PortfolioTrainer()
env = trainer.prepare_environment()
agent = PortfolioRLAgent(env=env)
agent.load_model("models/final_model.zip")  # Load existing model
agent.train(total_timesteps=50000)  # Continue training
```

## Customization

### Add More Stocks

Edit `src/data_collector.py`:
```python
self.stocks = {
    'tech': ['AAPL', 'MSFT', ...],  # Add more symbols
    # ... other sectors
}
```

### Adjust RL Parameters

Edit `src/trainer.py`:
```python
agent, callback = trainer.train_agent(
    env=env,
    total_timesteps=200000,  # More training
    learning_rate=1e-4  # Different learning rate
)
```

### Modify Reward Function

Edit `src/portfolio_env.py`:
- Change reward calculation in `step()` method
- Adjust `reward_scaling` parameter

## Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir models/tensorboard
```

View training metrics:
- Portfolio value over time
- Reward progression
- Sharpe ratio improvements

## Project Structure

```
Quant/
├── data/              # Market data (CSV files)
├── models/            # Trained RL models
├── src/
│   ├── data_collector.py      # Data collection
│   ├── feature_engineer.py    # Feature engineering
│   ├── portfolio_env.py       # RL environment
│   ├── rl_agent.py           # PPO agent
│   ├── trainer.py             # Training pipeline
│   └── generate_portfolio.py # Portfolio generation
└── requirements.txt
```

## Next Steps

1. **Experiment with different RL algorithms:**
   - Try DQN for discrete actions
   - Test SAC (Soft Actor-Critic) for continuous control
   - Compare with A2C (Advantage Actor-Critic)

2. **Add more features:**
   - Market sentiment data
   - Economic indicators
   - Sector/industry trends

3. **Improve reward function:**
   - Add risk constraints
   - Include drawdown penalties
   - Factor in transaction costs more heavily

4. **Backtesting:**
   - Test on historical data
   - Compare with benchmark (S&P 500)
   - Analyze performance metrics

## Troubleshooting

**Issue: No data files found**
- Run `python src/data_collector.py` first

**Issue: Out of memory during training**
- Reduce number of symbols
- Decrease `lookback_window` in PortfolioEnv
- Reduce `n_steps` in RL agent

**Issue: Training is slow**
- Reduce `total_timesteps` for testing
- Use fewer symbols initially
- Consider GPU acceleration (CUDA)

## Questions?

The system is designed to continuously learn and improve. The RL agent will:
- Learn optimal portfolio allocations
- Adapt to changing market conditions
- Improve performance over time
- Handle risk and transaction costs

Happy training! 🚀

