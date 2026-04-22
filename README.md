# Investment App 📈

Building an ML app that tells me whether to buy a stock or not.
The goal is simple — learn machine learning and make money at 
the same time.

## What it does
Pulls real stock data from Yahoo Finance, calculates technical 
signals, and uses a Random Forest model to predict whether a 
stock will be higher 30 days from now. Buy or don't buy — 
that's it.

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

## Status
Active development — currently building the ML model

## About me
Recent CS grad teaching myself ML by building real projects.
Currently looking for SWE or data roles.

📧 jlchavez.cs@gmail.com
🔗 https://www.linkedin.com/in/jorge-chavez-764028257/
