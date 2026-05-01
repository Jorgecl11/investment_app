# Investment App 📈

Building an ML app that tells me whether to buy a stock or not.
The goal is simple — learn machine learning and make money at 
the same time.

## What it does
Takes any stock ticker, pulls real market data, engineers 13 technical 
indicators, and uses a Random Forest model to predict whether the stock 
will be higher 30 days from now. Outputs a BUY, DONT BUY, or 
HOLD / UNCERTAIN signal with confidence probabilities.

## Why I built this
I just graduated with a CS degree and wanted to build something 
real with ML instead of just following tutorials. Stock market 
felt like the most interesting problem to tackle.

## Tech Stack
- Python
- pandas
- yfinance
- scikit-learn
- pandas-ta

## How to run
1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python main.py
4. Enter any stock ticker when prompted (e.g. NVDA, AAPL, TSLA)

## What the model analyzes
- Price change over last 5, 20, and 50 days
- 20 and 50 day moving averages
- Price vs moving average ratio
- 20 day volatility
- RSI - overbought/oversold signal
- MACD - momentum shifts
- Volume signal - is the move backed by real volume?
- SPY market context - is the overall market up or down?
- Relative strength - is this stock beating the market?
- Trend direction - is the 20 day MA above the 50 day MA?
- Profit margin - is the company making money?
- Revenue growth - is the business growing?
- Free cashflow - is the company generating cash?
- Current ratio - can they pay short term debt?

## Architecture
The codebase is split into 4 modules:
- `data.py` — fetches stock and SPY data from Yahoo Finance
- `features.py` — engineers all 13 technical indicators
- `model.py` — trains Random Forest and generates prediction
- `main.py` — runs the full pipeline with user input

## Key ML decisions
- Chronological train/test split (last 252 trading days = 1 year for testing)
- No random split — avoids time-series data leakage
- Confidence threshold filter — only signals BUY or DONT BUY above 60% confidence, otherwise shows HOLD / UNCERTAIN
- class_weight="balanced" — handles imbalanced buy/sell labels

## ⚠️ Disclaimer
This app is built for educational purposes and personal experimentation.
It is not financial advice. Stock predictions are never guaranteed —
use this as a learning tool, not as a substitute for proper financial research.

## Status
- ✅ Stage 1 - Fetching real stock data
- ✅ Stage 2 - Feature engineering (13 indicators)
- ✅ Stage 3 - Random Forest model training
- ✅ Stage 4 - Model evaluation with classification report
- ✅ Stage 5 - Live buy/sell/uncertain signal with confidence filter
- ✅ Refactored into modular pipeline (data, features, model, main)

## Roadmap
- Add backtesting — simulate trades and measure returns
- Train on multiple stocks for more robust predictions
- Add walk-forward validation
- Build a web UI with Flask

## About me
Recent CS grad teaching myself ML by building real projects.
Currently looking for SWE or data roles.

📧 jlchavez.cs@gmail.com
🔗 https://www.linkedin.com/in/jorge-chavez-764028257/