import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from data import get_stock_data
from features import add_features, features
from model import train_model, predict, backtest
from fundamentals import get_fundamentals, score_fundamentals

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

        # fetch and score fundamentals
        print(f"Fetching fundamentals for {ticker}...")
        fundamentals = get_fundamentals(ticker)
        fund_score = score_fundamentals(fundamentals)
        print(f"Fundamental score: {fund_score} / 4")

        # make live prediction with fundamental score
        prediction, confidence = predict(
            model, latest_features, stock_live, ticker, fund_score, fundamentals
        )

    except ValueError as e:
        print(f" {e}")
        continue
    except Exception as e:
        print(f" Unexpected error: {e}")
        continue