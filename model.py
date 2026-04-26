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


def predict(model, latest_features, stock_live, ticker):
    """
    Makes a prediction using the most recent trading day's features.
    """
    prediction = model.predict(latest_features)[0]
    confidence = model.predict_proba(latest_features)[0]

    current_price = stock_live["Close"].iloc[-1]
    rsi           = stock_live["rsi"].iloc[-1]
    return_5d     = stock_live["return_5d"].iloc[-1]
    volatility    = stock_live["volatility_20d"].iloc[-1]
    trend         = stock_live["trend_20_50"].iloc[-1]
    alpha         = stock_live["alpha_5d"].iloc[-1]

    print("\n" + "="*40)
    print(f"  {ticker.upper()} Investment Signal")
    print("="*40)
    print(f"  Current Price:  ${current_price:.2f}")
    print(f"  RSI:            {rsi:.1f}")
    print(f"  5 Day Return:   {return_5d*100:.1f}%")
    print(f"  Volatility:     {volatility:.1f}")
    print(f"  Trend:          {'Uptrend 📈' if trend == 1 else 'Downtrend 📉'}")
    print(f"  vs Market:      {alpha*100:.1f}%")
    print()

    if confidence[1] >= 0.60:
        print(f"  Signal:         BUY 📈")
    elif confidence[0] >= 0.60:
        print(f"  Signal:         DONT BUY 📉")
    else:
        print(f"  Signal:         HOLD / UNCERTAIN ⚠️")

    print(f"  Dont Buy Prob:  {confidence[0]*100:.1f}%")
    print(f"  Buy Prob:       {confidence[1]*100:.1f}%")
    print("="*40)
    print("  ⚠️  Not financial advice")
    print("="*40)

    return prediction, confidence