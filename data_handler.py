import yfinance as yf
from utils import get_nyse_date, get_nyse_datetime
from data_processor import preprocess_data
from feature_engineering import get_feature_columns
import pandas as pd
import os


def fetch_data(ticker, end_date, start_date="2008-01-01", use_local_data=True):
    """
    Fetch historical data for a given ticker, using adjusted prices to handle stock splits.
    If use_local_data is True, it will check for local data before fetching from the cloud.
    """
    local_file = f"{ticker}_data.csv"  # Define a local file name

    # Check if local data exists and use it if the flag is set
    if use_local_data and os.path.exists(local_file):
        df = custom_read_csv(local_file)
        print("Loaded data from local file.")

    else:
        # Fetch data from yfinance
        df = yf.download(ticker, start=start_date, end=end_date)

        # Debug prints
        print("DEBUG: Type of df:", type(df))
        print("DEBUG: Shape of df:", df.shape)
        print("DEBUG: Columns of df:", df.columns)
        print("DEBUG: Index of df:", df.index)
        print("DEBUG: Sample data:", df.head())

        # Reset index to include 'Date' as a column
        df.reset_index(inplace=True)

        # Check if columns are flat before creating MultiIndex
        if not isinstance(df.columns, pd.MultiIndex):
            # Set the ticker as the first level of the multi-index for columns
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])

        # Reorder columns using the first level of the MultiIndex
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], [ticker]]
        )

        # Debug print before reordering
        print("DEBUG: Reordering columns with cols:", cols)

        # Access the columns using the correct MultiIndex structure by column name and then ticker
        df = df.loc[:, cols]  # Reorder columns

        # Save the DataFrame to CSV without specifying columns
        df.to_csv(local_file, index=True)  # Save with index if needed

        print("Fetched data from yfinance.")

    # Only need this code for older yfinance versions
    # df['Open'] = df['Adj Close'] * df['Open'] / df['Close']
    # df['High'] = df['Adj Close'] * df['High'] / df['Close']
    # df['Low'] = df['Adj Close'] * df['Low'] / df['Close']
    # df['Close'] = df['Adj Close']

    return df


def custom_read_csv(file_path):
    # Read the first row to extract column names
    header = pd.read_csv(file_path, nrows=1, header=None)
    column_names = header.iloc[0].tolist()  # Get the column names

    # Read the second row to extract the ticker
    ticker_row = pd.read_csv(file_path, skiprows=1, nrows=1, header=None)
    ticker = ticker_row.iloc[0, 0]  # Assuming the ticker is in the first column

    # Read the data, skipping the first two rows
    data = pd.read_csv(file_path, skiprows=2)

    # Set 'Date' as the index and create a MultiIndex with 'Ticker'
    data.set_index("Date", inplace=True)
    data.columns = pd.MultiIndex.from_product(
        [[ticker], column_names[1:]]
    )  # Use the rest of the column names

    return data
