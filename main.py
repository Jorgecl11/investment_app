from data import get_stock_data
from features import add_features, features
from model import train_model, predict

while True:
    ticker = input("\nEnter stock ticker (e.g. NVDA, AAPL, TSLA) or 'quit' to exit: ").upper()

    if ticker == "QUIT":
        print("Goodbye!")
        break

    try:
        print(f"\nFetching data for {ticker}...")
        stock_raw, spy = get_stock_data(ticker)

        stock_live = add_features(stock_raw, spy, drop_target_rows=False)
        latest_features = stock_live[features].tail(1)

        stock_clean = add_features(stock_raw, spy, drop_target_rows=True)

        print(f"Training model on {ticker} data...")
        model = train_model(stock_clean, features)

        prediction, confidence = predict(
            model, latest_features, stock_live, ticker
        )

    except ValueError as e:
        print(f"❌ {e}")
        continue
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        continue