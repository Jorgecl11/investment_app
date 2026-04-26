import yfinance as yf

def get_stock_data(ticker, start="2021-01-01"):
    """
    Downloads historical stock data from Yahoo Finance.
    Returns a cleaned DataFrame with single level columns.
    """
    stock = yf.download(ticker, start=start, progress=False)

    if stock.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    stock.columns = stock. columns.get_level_values(0)
    
    return stock
    