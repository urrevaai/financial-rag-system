import pandas as pd
import numpy as np
import yfinance as yf
from transformers import pipeline
from ingest_data import fetch_news_headlines

# --- Configuration ---
RISK_FACTORS_PATH = "risk_factors.txt"
TICKER = "AAPL"

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_stock_trend():
    """
    Analyzes stock trends using fresh data and returns structured data for plotting and metrics.
    """
    try:
        stock_data = yf.download(TICKER, period="1y", progress=False, auto_adjust=True)
        if stock_data.empty:
            return None

        # --- Calculations ---
        stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()
        
        latest_data = stock_data.iloc[-1]
        previous_close = stock_data.iloc[-2]['Close']
        
        metrics = {
            "ticker": TICKER,
            "latest_price": latest_data['Close'],
            "price_change": latest_data['Close'] - previous_close,
            "price_change_pct": (latest_data['Close'] - previous_close) / previous_close,
            "volume": latest_data['Volume'],
            "50_day_ma": latest_data['50_MA'],
            "200_day_ma": latest_data['200_MA'],
        }
        
        # Return both the metrics and the full DataFrame for charting
        return {
            "metrics": metrics,
            "chart_data": stock_data
        }

    except Exception as e:
        print(f"Error in analyze_stock_trend: {e}")
        return None

# --- Other tools remain the same ---
def summarize_risk_factors():
    """Reads and summarizes the extracted 'Risk Factors' section from the 10-K."""
    try:
        with open(RISK_FACTORS_PATH, 'r', encoding='utf-8') as f:
            risk_text = f.read()
        summary = risk_text[:1500] + "..."
        return f"**Summary of Key Risk Factors (from latest 10-K filing):**\n\n{summary}"
    except Exception:
        return "Risk factors file not found. Please run `ingest_data.py` first."

def assess_news_sentiment():
    """Fetches latest news and performs sentiment analysis."""
    headlines = fetch_news_headlines()
    if not headlines:
        return "Could not fetch news headlines."
    
    sentiments = sentiment_pipeline(headlines[:10])
    positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
    negative_count = len(sentiments) - positive_count
    return f"**Recent News Sentiment:**\n- Positive: {positive_count}/10\n- Negative: {negative_count}/10"