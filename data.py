import yfinance as yf

def get_stock_data(ticker, start="2022-01-01"):
    """
    Downloads historical stock data from Yahoo Finance.
    Also downloads SPY for market context.
    Returns stock DataFrame and SPY DataFrame.
    """
    stock = yf.download(ticker, start=start, progress=False)
    spy   = yf.download("SPY", start=start, progress=False)

    if stock.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    stock.columns = stock.columns.get_level_values(0)
    spy.columns   = spy.columns.get_level_values(0)

    return stock, spy