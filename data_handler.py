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
    # end_date = get_nyse_date() removed this because we add 1 day already
    
    # Download data with adjusted prices #not 0.2.52 automatically adjusts the prices for stock splits
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Only need this code for older yfinance versions
    # df['Open'] = df['Adj Close'] * df['Open'] / df['Close']
    # df['High'] = df['Adj Close'] * df['High'] / df['Close']
    # df['Low'] = df['Adj Close'] * df['Low'] / df['Close']
    # df['Close'] = df['Adj Close']
    
    # Create MultiIndex columns
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']  # Each price type is its own level
    df = df[cols]  # Reorder columns
    df.columns = pd.MultiIndex.from_product([cols, [ticker]])  # Add ticker as second level
    
    return df

