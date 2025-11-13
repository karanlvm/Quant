# Deployment Guide

## GitHub Deployment

### 1. Initial Setup

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: ML Portfolio Optimizer"

# Add remote repository
git remote add origin https://github.com/yourusername/quant-portfolio.git
git push -u origin main
```

### 2. Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add:
   - **Name**: `NEWS_API_KEY`
   - **Value**: Your NewsAPI.org API key

### 3. Enable GitHub Actions

The workflow will automatically run:
- Every 6 hours (scheduled)
- On push to main branch
- Manually via workflow_dispatch

### 4. Monitor Workflow

- Go to **Actions** tab in GitHub
- View workflow runs
- Check logs for training progress

## Web Deployment

### Option 1: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start web server
python web/app.py
```

Visit: `http://localhost:8000`

### Option 2: Deploy to Heroku

1. Create `Procfile`:
```
web: uvicorn web.app:app --host 0.0.0.0 --port $PORT
```

2. Deploy:
```bash
heroku create your-app-name
heroku config:set NEWS_API_KEY=your_key
git push heroku main
```

### Option 3: Deploy to Railway/Render

1. Set environment variables:
   - `NEWS_API_KEY`
   - `PORT` (if needed)

2. Deploy from GitHub

## Environment Variables

Create `.env` file:
```
NEWS_API_KEY=your_news_api_key_here
```

Or export:
```bash
export NEWS_API_KEY="your_key"
```

## Continuous Learning Setup

The GitHub Actions workflow will:
1. ✅ Collect market data
2. ✅ Fetch news and analyze sentiment
3. ✅ Train/update RL model
4. ✅ Generate portfolio
5. ✅ Commit results

Results are saved to:
- `data/` - Market data
- `models/` - Trained models
- `portfolio_allocation.json` - Latest portfolio

## Monitoring

Check workflow status:
- GitHub Actions tab
- View logs for errors
- Check generated files

## Troubleshooting

**Workflow fails:**
- Check NEWS_API_KEY is set
- Verify API key is valid
- Check GitHub Actions logs

**Model not training:**
- Ensure data files exist
- Check Python dependencies
- Review training logs

**Web interface not loading:**
- Verify model is trained
- Check data files exist
- Review FastAPI logs

