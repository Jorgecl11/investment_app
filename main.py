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