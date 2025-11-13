"""
Bloomberg Terminal-Style Web Interface
Real-time portfolio performance dashboard
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import pandas as pd
from datetime import datetime
import sys
import asyncio
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from news_collector import NewsCollector
    from generate_portfolio import PortfolioGenerator
except ImportError:
    print("Warning: Some modules not available")

app = FastAPI(title="Quant Portfolio Optimizer - Bloomberg Terminal")

# Templates
template_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=template_dir)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Bloomberg Terminal Dashboard"""
    return templates.TemplateResponse("bloomberg_terminal.html", {"request": request})

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current optimal portfolio"""
    model_path = "models/final_model.zip"
    
    if not os.path.exists(model_path):
        return JSONResponse({
            "error": "Model not trained yet",
            "portfolio": [],
            "status": "training"
        })
    
    try:
        # Load symbols from data
        data_dir = "data/prices"
        symbols = []
        if os.path.exists(data_dir):
            files = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')]
            symbols = files[:30]  # Limit for display
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM']
        
        generator = PortfolioGenerator(model_path=model_path)
        portfolio = generator.generate_portfolio(symbols)
        
        # Format for frontend
        portfolio_list = []
        total_allocation = 0
        
        for symbol, allocation in portfolio.items():
            portfolio_list.append({
                'symbol': symbol,
                'allocation': round(allocation['allocation_percent'], 2),
                'weight': round(allocation['allocation'], 4)
            })
            total_allocation += allocation['allocation_percent']
        
        # Sort by allocation
        portfolio_list.sort(key=lambda x: x['allocation'], reverse=True)
        
        return JSONResponse({
            "portfolio": portfolio_list,
            "total_allocation": round(total_allocation, 2),
            "status": "ready",
            "updated_at": datetime.now().isoformat(),
            "num_assets": len(portfolio_list)
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "portfolio": [],
            "status": "error"
        })

@app.get("/api/performance")
async def get_performance():
    """Get portfolio performance metrics"""
    try:
        # Load portfolio allocation
        portfolio_file = "portfolio_allocation.json"
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                portfolio = json.load(f)
        else:
            portfolio = {}
        
        # Simulate performance metrics (in production, calculate from actual data)
        performance = {
            "total_return": 12.5,  # Would calculate from actual performance
            "sharpe_ratio": 1.85,
            "max_drawdown": -3.2,
            "volatility": 8.5,
            "beta": 0.92,
            "alpha": 2.3,
            "win_rate": 68.5,
            "num_trades": 47,
            "portfolio_value": 112500.0,
            "initial_value": 100000.0
        }
        
        return JSONResponse({
            "performance": performance,
            "updated_at": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "performance": {}
        })

@app.get("/api/market-data")
async def get_market_data():
    """Get real-time market data for portfolio assets"""
    try:
        portfolio_file = "portfolio_allocation.json"
        symbols = []
        
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                portfolio = json.load(f)
                symbols = list(portfolio.keys())[:20]  # Top 20
        else:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM']
        
        # Load price data
        market_data = []
        data_dir = "data/prices"
        
        for symbol in symbols:
            filepath = os.path.join(data_dir, f"{symbol}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if len(df) > 0:
                        latest = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) > 1 else latest
                        
                        current_price = float(latest.get('Close', latest.get('close', 0)))
                        prev_price = float(prev.get('Close', prev.get('close', current_price)))
                        change = current_price - prev_price
                        change_pct = (change / prev_price * 100) if prev_price > 0 else 0
                        
                        market_data.append({
                            "symbol": symbol,
                            "price": round(current_price, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 2),
                            "volume": int(latest.get('Volume', latest.get('volume', 0)))
                        })
                except Exception as e:
                    continue
        
        return JSONResponse({
            "market_data": market_data,
            "updated_at": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "market_data": []
        })

@app.get("/api/sentiment")
async def get_sentiment():
    """Get news sentiment scores"""
    try:
        collector = NewsCollector()
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'JNJ', 'WMT']
        sentiment_scores = collector.get_sentiment_scores(symbols)
        
        # Format for frontend
        sentiment_list = []
        for symbol, scores in sentiment_scores.items():
            sentiment_list.append({
                'symbol': symbol,
                'score': round(scores['sentiment_score'], 3),
                'polarity': round(scores['avg_polarity'], 3),
                'news_count': scores['news_count'],
                'positive_ratio': round(scores['positive_ratio'], 2),
                'negative_ratio': round(scores['negative_ratio'], 2)
            })
        
        # Sort by score
        sentiment_list.sort(key=lambda x: abs(x['score']), reverse=True)
        
        return JSONResponse({
            "sentiment": sentiment_list,
            "updated_at": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "sentiment": []
        })

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "model_trained": os.path.exists("models/final_model.zip"),
        "data_collected": os.path.exists("data/prices"),
        "news_available": os.path.exists("data/news/latest_news.csv"),
        "last_update": None,
        "num_assets": 0,
        "num_news_articles": 0
    }
    
    # Count assets
    if os.path.exists("data/prices"):
        files = [f for f in os.listdir("data/prices") if f.endswith('.csv')]
        stats["num_assets"] = len(files)
    
    # Count news articles
    if os.path.exists("data/news/latest_news.csv"):
        try:
            df = pd.read_csv("data/news/latest_news.csv")
            stats["num_news_articles"] = len(df)
        except:
            pass
    
    # Get last update time
    portfolio_file = "portfolio_allocation.json"
    if os.path.exists(portfolio_file):
        stats["last_update"] = datetime.fromtimestamp(
            os.path.getmtime(portfolio_file)
        ).isoformat()
    
    return JSONResponse(stats)

@app.websocket("/ws/updates")
async def websocket_updates(websocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send updates every 5 seconds
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "update"
            }
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
