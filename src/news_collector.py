"""
News Collector and Sentiment Analyzer
Collects market news and analyzes sentiment to inform portfolio decisions
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict
import time
from textblob import TextBlob
import re

class NewsCollector:
    """
    Collects financial news and performs sentiment analysis
    Uses NewsAPI.org (free tier available)
    """
    
    def __init__(self, api_key: str = None, data_dir: str = "data"):
        self.api_key = api_key or os.getenv('NEWS_API_KEY', '')
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, 'news'), exist_ok=True)
        
        self.base_url = "https://newsapi.org/v2/everything"
        
        # Financial keywords for better news filtering
        self.financial_keywords = [
            'stock', 'market', 'trading', 'investment', 'portfolio',
            'earnings', 'revenue', 'profit', 'loss', 'dividend',
            'IPO', 'merger', 'acquisition', 'economy', 'inflation',
            'interest rate', 'Fed', 'Federal Reserve', 'GDP'
        ]
        
        # Stock symbols to track
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'BAC', 'JNJ', 'WMT', 'XOM', 'SPY', 'QQQ'
        ]
    
    def get_news(self, query: str, days_back: int = 1, max_results: int = 100) -> List[Dict]:
        """Fetch news articles from NewsAPI"""
        if not self.api_key:
            print("Warning: No NEWS_API_KEY provided. Using mock data.")
            return self._get_mock_news(query)
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': min(max_results, 100),  # API limit
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return self._get_mock_news(query)
    
    def _get_mock_news(self, query: str) -> List[Dict]:
        """Generate mock news for testing when API key is not available"""
        return [
            {
                'title': f'{query} shows strong performance',
                'description': f'Market analysis indicates positive trends for {query}',
                'content': f'Recent market data suggests {query} is performing well.',
                'publishedAt': datetime.now().isoformat(),
                'url': f'https://example.com/news/{query}',
                'source': {'name': 'Mock News'}
            }
        ]
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of news text"""
        if not text:
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment': 'neutral'}
        
        # Clean text
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = text[:1000]  # Limit length for analysis
        
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': float(polarity),
            'subjectivity': float(subjectivity),
            'sentiment': sentiment
        }
    
    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text"""
        found_symbols = []
        text_upper = text.upper()
        
        for symbol in self.symbols:
            if symbol in text_upper:
                found_symbols.append(symbol)
        
        return found_symbols
    
    def collect_market_news(self) -> pd.DataFrame:
        """Collect and analyze market news"""
        print("Collecting market news...")
        
        all_news = []
        
        # Collect general market news
        queries = [
            'stock market',
            'financial markets',
            'trading',
            'investment',
            'economy'
        ]
        
        for query in queries:
            articles = self.get_news(query, days_back=1, max_results=20)
            
            for article in articles:
                # Analyze sentiment
                text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
                sentiment = self.analyze_sentiment(text)
                
                # Extract mentioned symbols
                symbols = self.extract_symbols(text)
                
                news_item = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'query': query,
                    'sentiment_polarity': sentiment['polarity'],
                    'sentiment_subjectivity': sentiment['subjectivity'],
                    'sentiment': sentiment['sentiment'],
                    'mentioned_symbols': ','.join(symbols),
                    'collected_at': datetime.now().isoformat()
                }
                
                all_news.append(news_item)
            
            time.sleep(1)  # Rate limiting
        
        # Collect news for specific symbols
        for symbol in self.symbols[:10]:  # Limit to avoid rate limits
            articles = self.get_news(symbol, days_back=1, max_results=10)
            
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_sentiment(text)
                
                news_item = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'query': symbol,
                    'sentiment_polarity': sentiment['polarity'],
                    'sentiment_subjectivity': sentiment['subjectivity'],
                    'sentiment': sentiment['sentiment'],
                    'mentioned_symbols': symbol,
                    'collected_at': datetime.now().isoformat()
                }
                
                all_news.append(news_item)
            
            time.sleep(1)
        
        # Create DataFrame
        df = pd.DataFrame(all_news)
        
        # Save
        news_file = os.path.join(self.data_dir, 'news', f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(news_file, index=False)
        
        # Also save latest news
        latest_file = os.path.join(self.data_dir, 'news', 'latest_news.csv')
        df.to_csv(latest_file, index=False)
        
        print(f"Collected {len(df)} news articles")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
    
    def get_sentiment_scores(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get aggregated sentiment scores for symbols"""
        if symbols is None:
            symbols = self.symbols
        
        # Load latest news
        latest_file = os.path.join(self.data_dir, 'news', 'latest_news.csv')
        
        if not os.path.exists(latest_file):
            return {}
        
        df = pd.read_csv(latest_file)
        
        sentiment_scores = {}
        
        for symbol in symbols:
            # Filter news mentioning this symbol
            symbol_news = df[df['mentioned_symbols'].str.contains(symbol, na=False)]
            
            if len(symbol_news) > 0:
                avg_polarity = symbol_news['sentiment_polarity'].mean()
                avg_subjectivity = symbol_news['sentiment_subjectivity'].mean()
                positive_count = (symbol_news['sentiment'] == 'positive').sum()
                negative_count = (symbol_news['sentiment'] == 'negative').sum()
                total_count = len(symbol_news)
                
                sentiment_scores[symbol] = {
                    'avg_polarity': float(avg_polarity),
                    'avg_subjectivity': float(avg_subjectivity),
                    'positive_ratio': float(positive_count / total_count) if total_count > 0 else 0,
                    'negative_ratio': float(negative_count / total_count) if total_count > 0 else 0,
                    'news_count': int(total_count),
                    'sentiment_score': float(avg_polarity * (1 - avg_subjectivity))  # Weighted by confidence
                }
            else:
                sentiment_scores[symbol] = {
                    'avg_polarity': 0.0,
                    'avg_subjectivity': 0.5,
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0,
                    'news_count': 0,
                    'sentiment_score': 0.0
                }
        
        return sentiment_scores


if __name__ == "__main__":
    collector = NewsCollector()
    
    # Collect news
    news_df = collector.collect_market_news()
    
    # Get sentiment scores
    sentiment_scores = collector.get_sentiment_scores()
    
    print("\nSentiment Scores:")
    for symbol, scores in sentiment_scores.items():
        print(f"{symbol}: {scores['sentiment_score']:.3f} (polarity: {scores['avg_polarity']:.3f}, news: {scores['news_count']})")

