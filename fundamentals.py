import yfinance as yf
import pandas as pd

def get_fundamentals(ticker):
    """
    Fetches key fundamental data for a stock.
    Returns a dictionary of financial health metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        fundamentals = {
            "profit_margin":  info.get("profitMargins", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "total_cash":     info.get("totalCash", None),
            "total_debt":     info.get("totalDebt", None),
            "free_cashflow":  info.get("freeCashflow", None),
            "current_ratio":  info.get("currentRatio", None),
        }

        return fundamentals

    except Exception as e:
        print(f"Could not fetch fundamentals: {e}")
        return None


def score_fundamentals(fundamentals):
    """
    Scores a company's financial health from -4 to +4.
    Positive = strong fundamentals
    Negative = weak fundamentals
    Zero = unknown or neutral
    """
    if fundamentals is None:
        return 0

    score = 0

    # profit margin - is the company making money?
    margin = fundamentals.get("profit_margin")
    if margin is not None:
        if margin > 0:
            score += 1
        elif margin < 0:
            score -= 1

    # revenue growth - is the business growing?
    growth = fundamentals.get("revenue_growth")
    if growth is not None:
        if growth > 0.10:
            score += 1
        elif growth < 0:
            score -= 1

    # free cashflow - is the company generating cash?
    cashflow = fundamentals.get("free_cashflow")
    if cashflow is not None:
        if cashflow > 0:
            score += 1
        elif cashflow < 0:
            score -= 1

    # current ratio - can they pay short term debt?
    ratio = fundamentals.get("current_ratio")
    if ratio is not None:
        if ratio > 1.5:
            score += 1
        elif ratio < 1.0:
            score -= 1

    return score