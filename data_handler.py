import yfinance as yf
from utils import get_nyse_date, get_nyse_datetime
from data_processor import preprocess_data
from feature_engineering import get_feature_columns
import pandas as pd

def fetch_data(ticker, end_date, start_date='2008-01-01'):
    """
    Fetch historical data for a given ticker, using adjusted prices to handle stock splits
    """
    # Get current date in NYSE timezone
    end_date = get_nyse_date()
    
    # Download data with adjusted prices
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Use adjusted prices to handle stock splits correctly
    df['Open'] = df['Adj Close'] * df['Open'] / df['Close']
    df['High'] = df['Adj Close'] * df['High'] / df['Close']
    df['Low'] = df['Adj Close'] * df['Low'] / df['Close']
    df['Close'] = df['Adj Close']
    
    # Create MultiIndex columns
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']  # Each price type is its own level
    df = df[cols]  # Reorder columns
    df.columns = pd.MultiIndex.from_product([cols, [ticker]])  # Add ticker as second level
    
    return df

def fetch_features(ticker, model_type="linear_regression"):
    """
    Get the feature columns to use for prediction based on model type.
    """
    # Download data using period to ensure we get the right trading days
    stock_data = yf.download(ticker, period='5d')
    
    # Use adjusted prices
    stock_data['Open'] = stock_data['Adj Close'] * stock_data['Open'] / stock_data['Close']
    stock_data['High'] = stock_data['Adj Close'] * stock_data['High'] / stock_data['Close']
    stock_data['Low'] = stock_data['Adj Close'] * stock_data['Low'] / stock_data['Close']
    stock_data['Close'] = stock_data['Adj Close']
    
    # Create MultiIndex columns
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']  # Each price type is its own level
    stock_data = stock_data[cols]  # Reorder columns
    stock_data.columns = pd.MultiIndex.from_product([cols, [ticker]])  # Add ticker as second level
    
    if model_type == "linear_regression":
        # Process the data to add Prev Close with correct MultiIndex format
        processed_data = preprocess_data(stock_data)
        
        # Get the ticker value and use the last row with correct column structure
        ticker_value = processed_data.columns[0][1]  # Get ticker from second level
        prediction_features = processed_data[[('Open', ticker_value), 
                                           ('High', ticker_value), 
                                           ('Low', ticker_value), 
                                           ('Prev Close', ticker_value),
                                           ('Volume', ticker_value)]
                                           ].iloc[-1:]
        
        return prediction_features.values
    
    # Use get_feature_columns to get the right features for the model type
    if model_type!="linear_regression": 
        return fetch_non_linear_features(processed_data, model_type)