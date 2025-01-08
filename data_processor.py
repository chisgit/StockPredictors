import pandas as pd
import numpy as np
import streamlit as st

def preprocess_data(df):
    """
    Preprocess stock data for machine learning
    
    Args:
        df (pd.DataFrame): Input stock data DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed stock data, to add Prev Close col
    """
    if df.empty:
        st.warning("No data returned for the ticker. Skipping.")
        return df
    
    # Get the ticker value from the DataFrame
    ticker_value = df.columns[0][1]
    
    # Add 'Prev Close' column by shifting 'Close'
    df[('Prev Close', ticker_value)] = df[('Close', ticker_value)].shift(1)
    df.dropna(inplace=True)

    return df

def preprocess_non_linear_data(df):
    """
    Preprocess stock data for machine learning with consistent features
    """
    if df.empty:
        st.warning("No data returned for the ticker. Skipping.")
        return df
    
    ticker_value = df.columns[1][1]

    # Create next day's close (our target variable)
    df[('Next_Day_Close', ticker_value)] = df[('Close', ticker_value)].shift(-1)
    
    # Add standard technical indicators
    df[('SMA_20', ticker_value)] = df[('Close', ticker_value)].rolling(window=20).mean()
    df[('RSI', ticker_value)] = calculate_rsi(df[('Close', ticker_value)])
    
    # Add MACD
    macd_data = calculate_macd(df[('Close', ticker_value)])
    df[('MACD', ticker_value)] = macd_data['MACD']
    df[('MACD_Signal', ticker_value)] = macd_data['Signal']
    df[('MACD_Hist', ticker_value)] = macd_data['Histogram']
    
    # Add Bollinger Bands
    bb_data = calculate_bollinger_bands(df[('Close', ticker_value)])
    df[('BB_Upper', ticker_value)] = bb_data['Upper']
    df[('BB_Lower', ticker_value)] = bb_data['Lower']

    # Add Lagged features
    base_features = ['Close', 'Volume']
    lags = [1, 2]
    lagged_features = []
    
    for feature in base_features:
        for lag in lags:
            lagged_col = (f'{feature}_Lag_{lag}')
            df[lagged_col, ticker_value] = df[(feature, ticker_value)].shift(lag)
            lagged_features.append(lagged_col)
    
    # Drop rows with NaN values (from technical indicators)
    df.dropna(inplace=True)

    # Add this right before training the model (after feature extraction)
    check_data_alignment(df, 'xgboost')
    
    return df

def check_data_alignment(df, model_type):
    # Print first few rows to check the alignment
    print(f"DEBUG-XBoost - First few rows of the dataframe:\n{df.head()}")
    # Print column names
    print(f"DEBUG-XBoost - DataFrame Columns:\n{df.columns.tolist()}")

    # Check for missing values in the feature columns

    missing_values = df.isnull().sum()
    print(f"DEBUG-XBoost - Missing Values in Columns:\n{missing_values}")
    
    # Check if any NaN values in features
    if df.isnull().any().any():
        print(f"DEBUG-XBoost - DataFrame contains NaN values!\n")
    
    # Print first few rows to check the alignment
    print(f"DEBUG-XBoost - First few rows of the dataframe:\n{df.head()}")

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD technical indicator."""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return {
        'MACD': macd,
        'Signal': signal,
        'Histogram': macd - signal
    }

def calculate_bollinger_bands(prices, window=20):
    """Calculate Bollinger Bands technical indicator."""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return {
        'Upper': sma + (std * 2),
        'Lower': sma - (std * 2)
    }


