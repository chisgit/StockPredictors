import yfinance as yf

def get_recent_data(ticker):
    """Download recent stock data for the given ticker."""
    try:
        data = yf.download(ticker, period='5d', interval='1d')
        if data.empty or len(data) < 2:
            raise ValueError(f"Insufficient data available for {ticker}. Need at least 2 days of data.")
        return data
    except Exception as e:
        print(f"Error getting data for {ticker}: {str(e)}")
        return None

def group_predictions_by_ticker(todays_close_predictions, next_day_close_predictions):
    grouped_predictions = {}
    grouped_next_day_predictions = {}
    
    for ticker, prediction in todays_close_predictions:
        if ticker not in grouped_predictions:
            grouped_predictions[ticker] = []
        grouped_predictions[ticker].append(prediction)

    for ticker, prediction in next_day_close_predictions:
        if ticker not in grouped_next_day_predictions:
            grouped_next_day_predictions[ticker] = []
        grouped_next_day_predictions[ticker].append(prediction)

    return grouped_predictions, grouped_next_day_predictions