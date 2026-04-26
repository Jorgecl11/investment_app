import pandas_ta as ta

def add_features(stock, spy, drop_target_rows=True):
    """
    Calculates technical indicators and target label.
    drop_target_rows=True  → training mode
    drop_target_rows=False → prediction mode
    """
    stock = stock.copy()

    stock["return_5d"]  = stock["Close"].pct_change(5)
    stock["return_20d"] = stock["Close"].pct_change(20)
    stock["return_50d"] = stock["Close"].pct_change(50)

    stock["ma_20"] = stock["Close"].rolling(20).mean()
    stock["ma_50"] = stock["Close"].rolling(50).mean()

    stock["price_vs_ma20"] = stock["Close"] / stock["ma_20"]
    stock["price_vs_ma50"] = stock["Close"] / stock["ma_50"]

    stock["volatility_20d"] = stock["Close"].rolling(20).std()

    stock["rsi"] = ta.rsi(stock["Close"], length=14)

    macd = ta.macd(stock["Close"])
    stock["macd"]        = macd["MACD_12_26_9"]
    stock["macd_signal"] = macd["MACDs_12_26_9"]

    stock["volume_signal"] = stock["Volume"] / stock["Volume"].rolling(20).mean()

    spy_return = spy["Close"].pct_change(5)
    spy_return.name = "spy_return_5d"
    stock = stock.join(spy_return, how="left").sort_index()

    stock["alpha_5d"] = stock["return_5d"] - stock["spy_return_5d"]
    stock["trend_20_50"] = (stock["ma_20"] > stock["ma_50"]).astype(int)

    if drop_target_rows:
        future_close = stock["Close"].shift(-30)
        stock["target"] = float("nan")
        stock.loc[future_close > stock["Close"], "target"] = 1
        stock.loc[future_close <= stock["Close"], "target"] = 0
        stock.dropna(inplace=True)
        stock["target"] = stock["target"].astype(int)
    else:
        stock = stock.dropna(subset=features).copy()

    return stock


features = [
    "return_5d", "return_20d", "return_50d",
    "price_vs_ma20", "price_vs_ma50",
    "volatility_20d", "rsi", "macd", "macd_signal",
    "volume_signal", "spy_return_5d", "alpha_5d", "trend_20_50"
]