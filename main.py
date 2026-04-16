#yfinance library that talks to yahoo finance api and downloads stock data
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

today = date.today()

five_days_ago = today - timedelta(days=5)
yesterday = today - timedelta(days=1)

stock = yf.download("NVDA", start="2024-01-01", end=yesterday.strftime("%Y-%m-%d"))

print(type(stock))
print(stock.shape)
print(stock.head())

stock.columns = stock.columns.get_level_values(0)

stock["return_5d"] = stock["Close"].pct_change(5)
stock["return_20d"] = stock["Close"].pct_change(20)
stock["return_50d"] = stock["Close"].pct_change(50)
print(stock[["return_5d", "return_20d", "return_50d"]].head(25))
