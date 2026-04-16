import yfinance as yf

stock = yf.download("AAPL", start="2024-01-01", end="2026-04-15")

print(stock.head())

print(stock.shape)