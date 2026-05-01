from data import get_stock_data
from features import add_features, features
from model import train_model, predict, backtest

while True:
    ticker = input("\nEnter stock ticker (e.g. NVDA, AAPL, TSLA) or 'quit' to exit: ").upper()

    if not ticker:
        print("Please enter a valid ticker.")
        continue

    if ticker == "QUIT":
        print("Goodbye!")
        break

    try:
        print(f"\nFetching data for {ticker}...")
        stock_raw, spy = get_stock_data(ticker)

        # build features for live prediction
        stock_live = add_features(stock_raw, spy, drop_target_rows=False)
        latest_features = stock_live[features].tail(1)

        # build features for training
        stock_clean = add_features(stock_raw, spy, drop_target_rows=True)

        # train the model
        print(f"Training model on {ticker} data...")
        model = train_model(stock_clean, features)

        # run backtest
        backtest(model, stock_clean, features)

        # make live prediction
        prediction, confidence = predict(
            model, latest_features, stock_live, ticker
        )

    except ValueError as e:
        print(f" {e}")
        continue
    except Exception as e:
        print(f" Unexpected error: {e}")
        continue