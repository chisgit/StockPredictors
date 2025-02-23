import yfinance as yf
from utils import get_nyse_date, get_nyse_datetime
from data_processor import preprocess_data
from feature_engineering import get_feature_columns
import pandas as pd
import os


def fetch_data(ticker, end_date, start_date="2008-01-01", use_local_data=False):
    """
    Fetch historical data for a given ticker, using adjusted prices to handle stock splits.
    If use_local_data is True, it will check for local data before fetching from the cloud.
    """
    local_file = f"{ticker}_data.csv"  # Define a local file name

    # Check if local data exists and use it if the flag is set
    if use_local_data and os.path.exists(local_file):
        df = pd.read_csv(local_file)
        print("Loaded data from local file.")
    else:
        # Fetch data from yfinance
        df = yf.download(ticker, start=start_date, end=end_date)
        # Save to local file for future use
        df.to_csv(local_file, index=False)
        print("Fetched data from yfinance.")

    # Only need this code for older yfinance versions
    # df['Open'] = df['Adj Close'] * df['Open'] / df['Close']
    # df['High'] = df['Adj Close'] * df['High'] / df['Close']
    # df['Low'] = df['Adj Close'] * df['Low'] / df['Close']
    # df['Close'] = df['Adj Close']

    # Create MultiIndex columns
    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]  # Each price type is its own level
    df = df[cols]  # Reorder columns
    df.columns = pd.MultiIndex.from_product(
        [cols, [ticker]]
    )  # Add ticker as second level

    return df
