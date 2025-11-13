# Quant Portfolio Optimizer - ML/RL System

An AI-powered portfolio optimization system using Reinforcement Learning that continuously learns and adapts to market conditions, incorporating news sentiment analysis for better decision-making.

## 🌟 Features

- 🤖 **Reinforcement Learning**: Uses PPO (Proximal Policy Optimization) to learn optimal portfolio allocations
- 📊 **Comprehensive Data Collection**: Analyzes 150+ stocks across 11 major sectors
- 📰 **News Sentiment Analysis**: Integrates market news and sentiment to inform decisions
- 🔄 **Continuous Learning**: GitHub Actions workflow updates model every 6 hours
- 🌐 **Web Interface**: Simple, beautiful web dashboard showing real-time portfolio recommendations
- 📈 **Diverse Portfolio**: Optimizes across tech, finance, healthcare, energy, materials, REITs, utilities, and more

## 🚀 Quick Start

### Local Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd Quant
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up News API (Free):**
   - Get free API key from [NewsAPI.org](https://newsapi.org/)
   - Add to environment or GitHub Secrets:
   ```bash
   export NEWS_API_KEY="your_api_key_here"
   ```

4. **Collect initial data:**
```bash
python src/data_collector.py
```

5. **Train the model:**
```bash
python src/trainer.py
```

6. **Start web server:**
```bash
python web/app.py
```

Visit `http://localhost:8000` to see the dashboard!

## 📁 Project Structure

```
Quant/
├── .github/
│   └── workflows/
│       └── continuous_learning.yml  # Auto-training workflow
├── data/                             # Market data storage
│   ├── prices/                       # Price data (CSV)
│   └── news/                         # News and sentiment data
├── models/                           # Trained RL models
├── src/
│   ├── data_collector.py            # Market data collection (150+ stocks)
│   ├── news_collector.py            # News API + sentiment analysis
│   ├── feature_engineer.py          # Feature engineering (includes sentiment)
│   ├── portfolio_env.py             # RL environment
│   ├── rl_agent.py                  # PPO agent
│   ├── trainer.py                   # Training pipeline
│   └── generate_portfolio.py        # Portfolio generation
├── web/
│   ├── app.py                       # FastAPI web server
│   ├── templates/
│   │   └── index.html              # Dashboard UI
│   └── static/                      # CSS/JS files
└── requirements.txt
```

## 🔄 Continuous Learning

The system includes a GitHub Actions workflow that:

1. **Runs every 6 hours** (configurable)
2. Collects latest market data
3. Fetches and analyzes news sentiment
4. Trains/updates the RL model
5. Generates new portfolio recommendations
6. Commits results to repository

### Setup GitHub Actions

1. Add `NEWS_API_KEY` to GitHub Secrets:
   - Go to Repository → Settings → Secrets → Actions
   - Add secret: `NEWS_API_KEY` with your API key

2. Push to GitHub - workflow will run automatically!

## 📊 Sectors Analyzed

The system analyzes stocks across **11 major sectors**:

- **Technology** (14 stocks): AAPL, MSFT, GOOGL, NVDA, TSLA, etc.
- **Financial Services** (14 stocks): JPM, BAC, GS, MS, etc.
- **Healthcare** (14 stocks): JNJ, PFE, UNH, ABT, etc.
- **Consumer Discretionary** (14 stocks): WMT, HD, MCD, NKE, etc.
- **Consumer Staples** (13 stocks): PG, KO, PEP, etc.
- **Industrial** (14 stocks): BA, CAT, GE, etc.
- **Energy** (13 stocks): XOM, CVX, COP, etc.
- **Materials** (13 stocks): LIN, APD, ECL, etc.
- **Real Estate/REITs** (13 stocks): AMT, PLD, EQIX, etc.
- **Utilities** (13 stocks): NEE, DUK, SO, etc.
- **Communication Services** (12 stocks): VZ, T, DIS, etc.
- **ETFs** (13 ETFs): SPY, QQQ, VTI, etc.

**Total: 150+ assets** for maximum diversification!

## 🧠 How It Works

### 1. Data Collection
- Collects OHLCV price data for 150+ stocks
- Gathers fundamental data (P/E, market cap, etc.)
- Fetches market news from NewsAPI
- Analyzes sentiment using TextBlob

### 2. Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ADX
- **Momentum**: Returns, momentum, rate of change
- **Volatility**: Rolling volatility, ATR, volatility ratios
- **Volume**: OBV, VPT, volume ratios
- **News Sentiment**: Polarity, subjectivity, sentiment scores
- **Market Regime**: Trending vs ranging detection

### 3. Reinforcement Learning
- **Environment**: Portfolio optimization with Sharpe ratio rewards
- **Agent**: PPO (Proximal Policy Optimization)
- **Action Space**: Portfolio weights (continuous allocations)
- **State Space**: Market features + news sentiment for all assets
- **Reward**: Risk-adjusted returns (Sharpe ratio)

### 4. Continuous Improvement
- Model trains on new data every 6 hours
- Incorporates latest news sentiment
- Adapts to changing market conditions
- Improves portfolio allocations over time

## 🌐 Web Interface

The web dashboard shows:
- **Optimal Portfolio**: Real-time allocation recommendations
- **News Sentiment**: Sentiment scores for tracked stocks
- **System Status**: Model training status, data availability

Access at: `http://localhost:8000`

## 📰 News Integration

The system uses **NewsAPI.org** (free tier available):
- Fetches financial news articles
- Analyzes sentiment (positive/negative/neutral)
- Extracts mentioned stock symbols
- Incorporates sentiment into RL features

### Getting News API Key

1. Sign up at [NewsAPI.org](https://newsapi.org/)
2. Get free API key (100 requests/day)
3. Add to environment: `export NEWS_API_KEY="your_key"`
4. Or add to GitHub Secrets for CI/CD

## 🔧 Configuration

### Training Parameters

Edit `src/trainer.py`:
```python
trainer.run_full_training(
    total_timesteps=100000,  # Training steps
    learning_rate=3e-4,      # Learning rate
    update=True              # Continue training existing model
)
```

### GitHub Actions Schedule

Edit `.github/workflows/continuous_learning.yml`:
```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours (change as needed)
```

## 📈 Performance

The RL agent learns to:
- Optimize portfolio allocations
- Balance risk and return
- Adapt to market volatility
- Incorporate news sentiment
- Diversify across sectors

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources
- More sophisticated sentiment analysis
- Different RL algorithms
- Enhanced web interface
- Backtesting framework

## 📝 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- **Stable-Baselines3** for RL algorithms
- **NewsAPI.org** for news data
- **yfinance** for market data
- **TextBlob** for sentiment analysis

---

**Note**: This system is for educational/research purposes. No actual trades are executed. Always do your own research before making investment decisions.
