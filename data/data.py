import yfinance as yf

def get_stock_data(ticker: str, start_date: str, end_date: str, period: str = "1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval=period)
    return data

def get_multiple_stocks_data(tickers: list, start_date: str, end_date: str, period: str = "1d"):
    data = {}
    for ticker in tickers:
        data[ticker] = get_stock_data(ticker, start_date, end_date, period)
    return data