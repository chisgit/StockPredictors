import yfinance as yf
from utils import get_nyse_date, get_nyse_datetime
from data_processor import preprocess_data, preprocess_non_linear_data
from feature_engineering import get_feature_columns
import pandas as pd

def fetch_data(ticker, end_date, start_date='2008-01-01'):
    """
    Fetch historical data for a given ticker
    """
    # Get current date in NYSE timezone
    #end_date = get_nyse_date()
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Convert to multi-index DataFrame
    # df.columns = pd.MultiIndex.from_product([[c for c in df.columns], [ticker]])
    
    return df

def fetch_features(ticker, model_type="linear_regression"):
    """
    Get the feature columns to use for prediction based on model type.
    """
    # Download data using period to ensure we get the right trading days
    stock_data = yf.download(ticker, period='5d')
    
    if model_type == "linear_regression":
        # Process the data to add Prev Close with correct MultiIndex format
        processed_data = preprocess_data(stock_data)
        
        # Get the ticker value and use the last row with correct column structure
        ticker_value = processed_data.columns[1][1]
        prediction_features = processed_data[[('Open', ticker_value), 
                                           ('High', ticker_value), 
                                           ('Low', ticker_value), 
                                           ('Prev Close', ticker_value),
                                           ('Volume', ticker_value)]
                                           ].iloc[-1:]
        
        print(f"DEBUG - fetch_features() - Prediction features:\n", prediction_features.values)
        return prediction_features.values
    
    # Use get_feature_columns to get the right features for the model type
    if model_type!="linear_regression": 
        return fetch_non_linear_features(processed_data, model_type)
