#yfinance library that talks to yahoo finance api and downloads stock data
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas_ta as ta


ticker = "NVDA"
today = date.today()
yesterday = today - timedelta(days=1)
stock = yf.download(ticker, start="2022-01-01", end=yesterday.strftime("%Y-%m-%d"))
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

# RSI - measures if stock is overbought (>70) or oversold (<30)
stock["rsi"]= ta.rsi(stock["Close"], length=14)

# MACD - measures momentum shifts
macd = ta.macd(stock["Close"])
stock["macd"] = macd["MACD_12_26_9"]
stock["macd_signal"] = macd["MACDs_12_26_9"]

# Volume signal - is the price move backed by real Volume
stock["volume_signal"] = stock["Volume"] / stock["Volume"].rolling(20).mean()

spy = yf.download("SPY", start="2022-01-01", end=yesterday.strftime("%Y-%m-%d"))
spy.columns = spy.columns.get_level_values(0)
spy_return = spy["Close"].pct_change(5)
spy_return.name = "spy_return_5d"
stock = stock.join(spy_return)

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

features = ["return_5d", "return_20d", "return_50d",
            "price_vs_ma20", "price_vs_ma50",
            "volatility_20d", "rsi", "macd_signal"]

# Feature matrix: selected stock indicators returns, moving averages, volatility
X = stock[features]

# Target variable:
# 1 = price increased after 30 days (buy signal)
# 0 = price did not increase after 30 days (no-buy signal)
y = stock["target"]

#Stage 3: Training the model.

# split by date - use old data to train and recent data to test.
split = int(len(X) * 0.9)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print(f"\nTraining on {len(X_train)} days of data")
print(f"Testing on {len(X_test)} days of data")

#training the model.
model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

print("\nModel trained Succesfully!")

#stage 4- Evaluate the model

#use the model to predict on data it has never seen.
predictions = model.predict(X_test)

# checks overall accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy*100:.1f}%")
print(f"\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=["Dont Buy", "Buy"]))