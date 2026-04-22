#yfinance library that talks to yahoo finance api and downloads stock data
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

ticker = "NVDA"
today = date.today()
yesterday = today - timedelta(days=1)
stock = yf.download(ticker, start="2024-01-01", end=yesterday.strftime("%Y-%m-%d"))
stock.columns = stock.columns.get_level_values(0)

# returns how much price has changed over n days.
stock["return_5d"] = stock["Close"].pct_change(5)
stock["return_20d"] = stock["Close"].pct_change(20)
stock["return_50d"] = stock["Close"].pct_change(50)

# calculate the average closing price over the last 20 and 50 trading days
stock["ma_20"] = stock["Close"].rolling(20).mean()
stock["ma_50"] = stock["Close"].rolling(50).mean()

# closing price divided by 20 and 50 day average
stock["price_vs_ma20"] = stock["Close"] / stock["ma_20"]
stock["price_vs_ma50"] = stock["Close"] / stock["ma_50"]

# calculates how wildly the price has been swinging over the last 20 days.
stock["volatility_20d"] = stock["Close"].rolling(20).std()

# looks into the price 30 days in the future if its higher than 
# 1 = yes (good time to buy), 0 = no (dont buy!).
stock["target"] = (stock["Close"].shift(-30) > stock["Close"]).astype(int)

# drop the rows that we cant calculate features or target.
stock.dropna(inplace=True)

total =len(stock)
up = stock["target"].sum()
down = (stock["target"] == 0).sum()

# previews features.
print(f"Stock: {ticker.upper()}")
print(f"Clean dataset shape: {stock.shape}")
print(f"Rows where price went UP in 30 days: {up} ({up/total*100:.1f}%)")
print(f"Rows where price went DOWN in 30 days: {down} ({down/total*100:.1f}%)")
print(f"\nSample of target labels:")
print(stock[["Close", "target"]].head(30))

