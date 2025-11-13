#!/bin/bash

# Start the web server
echo "Starting Quant Portfolio Optimizer Web Server..."
echo "Make sure you have:"
echo "1. Installed dependencies: pip install -r requirements.txt"
echo "2. Trained the model: python src/trainer.py"
echo "3. Set NEWS_API_KEY (optional): export NEWS_API_KEY='your_key'"
echo ""

cd "$(dirname "$0")"
python web/app.py

