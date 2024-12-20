import yfinance as yf
from utils import get_nyse_date, get_nyse_datetime
from data_processor import preprocess_data

def fetch_data(ticker, end_date, start_date='2010-01-01'):
    """
    Fetch historical data for a given ticker
    """
    # Get current date in NYSE timezone
    end_date = get_nyse_date()
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def fetch_features(ticker):
    """
    Fetch recent stock data for prediction features.
    """
    # Get current time in NYSE timezone
    current_time = get_nyse_datetime()
    
    # Download data using period to ensure we get the right trading days
    stock_data = yf.download(ticker, period='5d')
    print(f"Fetch Features stock_data")
    print(stock_data)
    
    prediction_data = preprocess_data(stock_data).tail(1)
    prediction_features = prediction_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']]
    
    print(f"Fetch DF to get Features prediction_data")
    print(prediction_features)

    return prediction_features