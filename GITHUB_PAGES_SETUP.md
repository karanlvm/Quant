# GitHub Pages Deployment Setup

## Quick Setup

1. **Enable GitHub Pages:**
   - Go to your repository → **Settings** → **Pages**
   - Under **Source**, select **GitHub Actions**
   - Save

2. **Add News API Key (Optional):**
   - Go to **Settings** → **Secrets and variables** → **Actions**
   - Add secret: `NEWS_API_KEY` with your API key from [NewsAPI.org](https://newsapi.org/)

3. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add GitHub Pages deployment"
   git push origin main
   ```

4. **Wait for Deployment:**
   - Go to **Actions** tab
   - Wait for "Deploy to GitHub Pages" workflow to complete
   - Your site will be available at: `https://yourusername.github.io/Quant/`

## How It Works

1. **GitHub Actions Workflow** (`.github/workflows/deploy_pages.yml`):
   - Runs on every push to `main` branch
   - Runs hourly to update data
   - Generates static JSON files from your data
   - Deploys to GitHub Pages

2. **Static Files Generated:**
   - `portfolio.json` - Portfolio allocations
   - `market-data.json` - Market prices and changes
   - `sentiment.json` - News sentiment scores
   - `performance.json` - Performance metrics
   - `stats.json` - System statistics

3. **Frontend:**
   - Reads from static JSON files
   - Updates every 30 seconds
   - No backend server needed!

## Custom Domain (Optional)

To use a custom domain:

1. Add `CNAME` file to `docs/` folder:
   ```
   yourdomain.com
   ```

2. Configure DNS:
   - Add CNAME record pointing to `yourusername.github.io`

## Troubleshooting

**Pages not updating?**
- Check GitHub Actions workflow status
- Ensure workflow completed successfully
- Wait a few minutes for Pages to rebuild

**Data not showing?**
- Ensure data files exist in `data/` directory
- Check workflow logs for errors
- Verify JSON files are generated in `docs/` folder

**404 errors?**
- Ensure repository name matches URL path
- Check `basePath` in `index.html` matches your repo name
- Verify `docs/.nojekyll` file exists

## Manual Update

To manually trigger an update:

1. Go to **Actions** tab
2. Select "Deploy to GitHub Pages" workflow
3. Click "Run workflow"
4. Select branch: `main`
5. Click "Run workflow"

Your Bloomberg Terminal will be live on GitHub Pages! 🚀

