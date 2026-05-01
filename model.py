from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(stock, features):
    """
    Trains a Random Forest model using a time-based split.
    Returns the trained model.
    """
    if len(stock) <= 252:
        raise ValueError("Not enough data: need more than 252 rows.")

    train = stock.iloc[:-252]
    test  = stock.iloc[-252:]

    X_train = train[features]
    X_test  = test[features]
    y_train = train["target"]
    y_test  = test["target"]

    print(f"Training on {len(X_train)} days of data")
    print(f"Testing on {len(X_test)} days of data")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy*100:.1f}%")
    print(classification_report(
        y_test, predictions,
        target_names=["Dont Buy", "Buy"],
        zero_division=0
    ))

    return model


def predict(model, latest_features, stock_live, ticker, fund_score=0, fundamentals=None):
    """
    Makes a prediction using the most recent trading day's features.
    Combines model signal with fundamental score.
    """
    prediction = model.predict(latest_features)[0]
    confidence = model.predict_proba(latest_features)[0]

    current_price = stock_live["Close"].iloc[-1]

    # warn about penny stocks
    if current_price < 1.00:
        print(f"  WARNING: Price below $1.00 - data may be unreliable")

    rsi       = stock_live["rsi"].iloc[-1]
    return_5d = stock_live["return_5d"].iloc[-1]
    volatility = stock_live["volatility_20d"].iloc[-1]
    trend     = stock_live["trend_20_50"].iloc[-1]
    alpha     = stock_live["alpha_5d"].iloc[-1]

    # determine base signal from model confidence
    if confidence[1] >= 0.60:
        base_signal = "BUY"
    elif confidence[0] >= 0.60:
        base_signal = "DONT BUY"
    else:
        base_signal = "UNCERTAIN"

    # combine model signal with fundamental score
    if base_signal == "BUY":
        if fund_score >= 2:
            final_signal = "STRONG BUY"
        elif fund_score <= -1:
            final_signal = "CAUTION - fundamentals weak"
        else:
            final_signal = "BUY"
    elif base_signal == "DONT BUY":
        if fund_score >= 2:
            final_signal = "WATCH - good company, bad timing"
        elif fund_score <= -2:
            final_signal = "AVOID"
        else:
            final_signal = "DONT BUY"
    else:
        if fund_score >= 2:
            final_signal = "WATCH - good company, uncertain timing"
        elif fund_score <= -2:
            final_signal = "AVOID"
        else:
            final_signal = "HOLD / UNCERTAIN"

    # display results
    print("\n" + "="*40)
    print(f"  {ticker.upper()} Investment Signal")
    print("="*40)

    # technical indicators section
    print(f"  Current Price:  ${current_price:.2f}")
    print(f"  RSI:            {rsi:.1f}")
    print(f"  5 Day Return:   {return_5d*100:.1f}%")
    print(f"  Volatility:     {volatility:.1f}")
    print(f"  Trend:          {'Uptrend' if trend == 1 else 'Downtrend'}")
    print(f"  vs Market:      {alpha*100:.1f}%")

    # fundamentals section
    print()
    print(f"  --- Fundamentals ---")
    if fundamentals:
        margin   = fundamentals.get("profit_margin")
        growth   = fundamentals.get("revenue_growth")
        cashflow = fundamentals.get("free_cashflow")
        ratio    = fundamentals.get("current_ratio")

        print(f"  Profit Margin:  {f'{margin*100:.1f}%' if margin is not None else 'N/A'}")
        print(f"  Revenue Growth: {f'{growth*100:.1f}%' if growth is not None else 'N/A'}")
        print(f"  Free Cashflow:  {f'${cashflow/1e9:.1f}B' if cashflow is not None else 'N/A'}")
        print(f"  Current Ratio:  {f'{ratio:.2f}' if ratio is not None else 'N/A'}")
    else:
        print(f"  No fundamental data available")
    print(f"  Fundamental Score: {fund_score} / 4")

    # signal section
    print()
    print(f"  --- Signal ---")
    print(f"  Signal:             {final_signal}")
    print()
    print(f"  Technical Dont Buy: {confidence[0]*100:.1f}%")
    print(f"  Technical Buy:      {confidence[1]*100:.1f}%")
    print("="*40)
    print("  Not financial advice")
    print("="*40)

    return prediction, confidence


def backtest(model, stock, features):
    """
    Simulates trades on test data and measures real world performance.
    Only trades on the last 252 days (data model never trained on).
    """
    import pandas as pd

    # get test data - last 252 days
    test = stock.iloc[-252:]

    # make predictions on all test days
    predictions = model.predict(test[features])

    # store results of each trade
    trades = []

    # loop through each day in test data
    for i in range(len(test) - 30):
        if predictions[i] == 1:
            entry_price  = test["Close"].iloc[i]
            exit_price   = test["Close"].iloc[i + 30]
            trade_return = (exit_price - entry_price) / entry_price

            trades.append({
                "date":        test.index[i].date(),
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "return":      trade_return,
                "win":         trade_return > 0
            })

    # calculate overall stats
    if len(trades) == 0:
        print("No buy signals generated in test period.")
        return

    trades_df = pd.DataFrame(trades)

    win_rate        = trades_df["win"].mean() * 100
    avg_return      = trades_df["return"].mean() * 100
    median_return   = trades_df["return"].median() * 100
    total_trades    = len(trades_df)
    buy_hold_return = (test["Close"].iloc[-1] - test["Close"].iloc[0]) / test["Close"].iloc[0] * 100

    print(f"\n{'='*40}")
    print(f"  Backtest Results")
    print(f"{'='*40}")
    print(f"  Total trades:      {total_trades}")
    print(f"  Win rate:          {win_rate:.1f}%")
    print(f"  Avg return/trade:  {avg_return:.1f}%")
    print(f"  Median return:     {median_return:.1f}%")
    print(f"  Buy & Hold return: {buy_hold_return:.1f}%")
    print(f"{'='*40}")

    return trades_df