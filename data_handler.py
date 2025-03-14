import yfinance as yf
import pandas as pd
import os

def custom_read_csv(file_path):
    """
    Read and parse the custom CSV file format with Type/Ticker headers.
    """
    # Read the entire file
    df = pd.read_csv(file_path)
    
    # Extract the column types and ticker info
    types = df.iloc[0]  # First row contains Types
    tickers = df.iloc[1]  # Second row contains Tickers
    
    # Remove the Type and Ticker rows and reset the index
    df = df.iloc[2:].reset_index(drop=True)
    
    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create MultiIndex columns
    columns = pd.MultiIndex.from_arrays(
        [types[1:].values, tickers[1:].values],
        names=['Type', 'Ticker']
    )
    df.columns = columns
    
    return df

def fetch_data(ticker, end_date, start_date="2008-01-01", use_local_data=False):
    """
    Fetch historical data for a given ticker, either from local file or yfinance.
    """
    local_file = f"{ticker}_data.csv"

    if use_local_data and os.path.exists(local_file):
        df = custom_read_csv(local_file)
        print(f"Loaded data from local file: {local_file}")
    else:
        # Fetch data from yfinance
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Create MultiIndex columns if not already present
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product(
                [['Price'], df.columns],
                names=['Type', 'Ticker']
            )
        
        # Save to CSV with custom format
        if not df.empty:
            # Prepare the DataFrame for saving
            df_to_save = df.copy()
            df_to_save.index.name = 'Date'
            df_to_save.to_csv(local_file)
            print(f"Fetched data from yfinance and saved to: {local_file}")
        else:
            raise ValueError(f"No data retrieved for ticker {ticker}")

    return df
