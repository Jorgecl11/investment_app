# Investment App 📈

Building an ML app that tells me whether to buy a stock or not.
The goal is simple — learn machine learning and make money at 
the same time.

## What it does
Uses machine learning to predict the probability of a stock being 
higher 30 days from now based on historical price patterns. 
Built as a learning project — not a financial advisor.

## Why I built this
I just graduated with a CS degree and wanted to build something 
real with ML instead of just following tutorials. Stock market 
felt like the most interesting problem to tackle.

## Tech Stack
- Python
- pandas
- yfinance
- scikit-learn

## How to run
1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Change ticker to whatever stock you want:
   ticker = "NVDA"
4. Run:
   python main.py

## What the model analyzes
- Price change over last 5, 20, and 50 days
- 20 and 50 day moving averages
- Whether price is above or below its average trend
- How volatile the stock has been lately

## ⚠️ Disclaimer
This app is built for educational purposes and personal experimentation.
It is not financial advice. Stock predictions are never guaranteed —
use this as a learning tool, not as a substitute for proper financial research.

## Status
- ✅ Stage 1 - Fetching real stock data
- ✅ Stage 2 - Feature engineering and buy signals
- 🔧 Stage 3 - ML model training (in progress)
- ⏳ Stage 4 - Model evaluation
- ⏳ Stage 5 - UI

## Roadmap
- Add more technical indicators (RSI, MACD)
- Train on multiple stocks
- Add backtesting
- Add confidence threshold before giving buy signal

## Status
Active development — currently building the ML model

## About me
Recent CS grad teaching myself ML by building real projects.
Currently looking for SWE or data roles.

📧 jlchavez.cs@gmail.com
🔗 https://www.linkedin.com/in/jorge-chavez-764028257/
