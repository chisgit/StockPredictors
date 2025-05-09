import pandas as pd
import numpy as np
import streamlit as st


def preprocess_data(df, model_type):
    """
    Preprocess stock data for machine learning

    Args:
        df (pd.DataFrame): Input stock data DataFrame

    Returns:
        pd.DataFrame: Preprocessed stock data with MultiIndex columns
    """
    if df.empty:
        st.warning("No data returned for the ticker. Skipping.")
        return df

    ticker_value = df.columns[0][1]

    # Just add Prev Close without dropping NaNs
    df[("Prev Close", ticker_value)] = df[("Close", ticker_value)].shift(1)

    # Ensure proper MultiIndex columns
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], ticker_value) if col[1] == "" else col for col in df.columns.tolist()]
    )

    return df


def check_data_alignment(df, model_type):
    # Print last few rows to check the alignment
    print(f"DEBUG-XBoost - Last few rows of the dataframe:\n{df.tail()}")
    # Print column names
    print(f"DEBUG-XBoost - DataFrame Columns:\n{df.columns.tolist()}\n")

    # Check for missing values in the feature columns

    missing_values = df.isnull().sum()
    print(f"DEBUG-XBoost - Missing Values in Columns:\n{missing_values}")

    # Check if any NaN values in features
    if df.isnull().any().any():
        print(f"DEBUG-XBoost - DataFrame contains NaN values!\n")

    # Print last few rows to check the alignment
    print(f"DEBUG-XBoost - Last few rows of the dataframe:\n{df.tail()}")

    # Log the DataFrame structure before feature selection
    print(f"DEBUG - DataFrame columns before feature selection: {df.columns.tolist()}")

    # Define base features
    base_features = ["Prev Close", "Volume"]

    # Technical indicators - match on first level of column names
    technical_features = [
        col[0]
        for col in df.columns
        if col[0]
        in [
            "SMA_20",
            "EMA",
            "RSI",
            "ROC",
            "Momentum",
            "BB",
            "ATR",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "BB_Upper",
            "BB_Lower",
            "Price_Change",
            "Returns",
            "HL_Range",
            "HL_Range_Pct",
        ]
    ]

    # Lagged features
    lagged_features = [col[0] for col in df.columns if "Lag" in col[0]]

    # Log selected features
    print(f"DEBUG - Base Features: {base_features}")
    print(f"DEBUG - Technical Features: {technical_features}")
    print(f"DEBUG - Lagged Features: {lagged_features}")

    if model_type == "linear_regression":
        feature_columns = ["Open", "High", "Low"] + base_features
        print(f"DEBUG - Feature columns for linear regression: {feature_columns}")


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
    return {"MACD": macd, "Signal": signal, "Histogram": macd - signal}


def calculate_bollinger_bands(prices, window=20):
    """Calculate Bollinger Bands technical indicator."""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return {"Upper": sma + (std * 2), "Lower": sma - (std * 2)}
