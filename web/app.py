"""
Simple Web Interface for Portfolio Optimization
FastAPI web app showing real-time portfolio recommendations
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import pandas as pd
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from news_collector import NewsCollector
from generate_portfolio import PortfolioGenerator

app = FastAPI(title="Quant Portfolio Optimizer")

# Templates
templates = Jinja2Templates(directory="web/templates")

# Static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current optimal portfolio"""
    model_path = "models/final_model.zip"
    
    if not os.path.exists(model_path):
        return {
            "error": "Model not trained yet",
            "portfolio": {},
            "status": "training"
        }
    
    try:
        # Load symbols from data
        data_dir = "data/prices"
        symbols = []
        if os.path.exists(data_dir):
            files = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')]
            symbols = files[:20]  # Limit to 20 for display
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        generator = PortfolioGenerator(model_path=model_path)
        portfolio = generator.generate_portfolio(symbols)
        
        # Format for frontend
        portfolio_list = [
            {
                'symbol': symbol,
                'allocation': round(allocation['allocation_percent'], 2)
            }
            for symbol, allocation in portfolio.items()
        ]
        
        return {
            "portfolio": portfolio_list,
            "total_allocation": sum(p['allocation'] for p in portfolio_list),
            "status": "ready",
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "portfolio": {},
            "status": "error"
        }

@app.get("/api/sentiment")
async def get_sentiment():
    """Get news sentiment scores"""
    try:
        collector = NewsCollector()
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM']
        sentiment_scores = collector.get_sentiment_scores(symbols)
        
        # Format for frontend
        sentiment_list = [
            {
                'symbol': symbol,
                'score': round(scores['sentiment_score'], 3),
                'polarity': round(scores['avg_polarity'], 3),
                'news_count': scores['news_count']
            }
            for symbol, scores in sentiment_scores.items()
        ]
        
        return {
            "sentiment": sentiment_list,
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "sentiment": []
        }

@app.get("/api/stats")
async def get_stats():
    """Get portfolio statistics"""
    portfolio_file = "portfolio_allocation.json"
    
    stats = {
        "model_trained": os.path.exists("models/final_model.zip"),
        "data_collected": os.path.exists("data/prices"),
        "news_available": os.path.exists("data/news/latest_news.csv"),
        "last_update": None
    }
    
    if os.path.exists(portfolio_file):
        try:
            with open(portfolio_file, 'r') as f:
                portfolio = json.load(f)
                stats["last_update"] = datetime.now().isoformat()
                stats["num_assets"] = len(portfolio)
        except:
            pass
    
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

