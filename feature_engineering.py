import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def add_lagged_features(df, ticker_value, base_features = ['Close', 'Volume'], lags=[1, 2]):
    """
    Create lagged features for specified columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        base_features (list): Features to create lags for
        lags (list): Lag periods to create
    
    Returns:
        pd.DataFrame: DataFrame with lagged features
    """
    ticker_value = df.columns[0][1]
    lagged_features = []
    
    for feature in base_features:
        for lag in lags:
            lagged_col = f'{feature}_Lag_{lag}'
            df[(lagged_col, ticker_value)] = df[(feature, ticker_value)].shift(lag)
            lagged_features.append(lagged_col)
    
    df.dropna(inplace=True)
    return df

def add_moving_averages(df, ticker_value):
    """Calculate various moving averages."""
    windows = [5, 10] # 20, 50, 100, 200]
    close_prices = df[('Close', ticker_value)]
    
    for window in windows:
        df.loc[:, (f'SMA_{window}', ticker_value)] = close_prices.rolling(window=window).mean()
        df.loc[:, (f'EMA_{window}', ticker_value)] = close_prices.ewm(span=window, adjust=False).mean()
    
    return df

def add_momentum_indicators(df, ticker_value):
    """Calculate momentum indicators."""
    close_prices = df[('Close', ticker_value)]
    
    # RSI
    rsi = RSIIndicator(close=close_prices, window=14)
    df.loc[:, ('RSI', ticker_value)] = rsi.rsi()
    
    # Rate of Change
    df.loc[:, ('ROC_5', ticker_value)] = close_prices.pct_change(periods=5)
    df.loc[:, ('ROC_10', ticker_value)] = close_prices.pct_change(periods=10)
    df.loc[:, ('ROC_20', ticker_value)] = close_prices.pct_change(periods=20)
    
    # Momentum
    df.loc[:, ('Momentum_5', ticker_value)] = close_prices.diff(5)
    df.loc[:, ('Momentum_10', ticker_value)] = close_prices.diff(10)
    
    return df

def add_volatility_indicators(df, ticker_value):
    """Calculate volatility indicators."""
    close_prices = df[('Close', ticker_value)]
    
    # Bollinger Bands
    bb = BollingerBands(close=close_prices, window=20, window_dev=2)
    df.loc[:, ('BB_Upper', ticker_value)] = bb.bollinger_hband()
    df.loc[:, ('BB_Lower', ticker_value)] = bb.bollinger_lband()
    df.loc[:, ('BB_Middle', ticker_value)] = bb.bollinger_mavg()
    
    # Average True Range (ATR)
    high = df[('High', ticker_value)]
    low = df[('Low', ticker_value)]
    
    tr1 = high - low
    tr2 = (high - close_prices.shift()).abs()
    tr3 = (low - close_prices.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df.loc[:, ('ATR', ticker_value)] = true_range.rolling(window=14).mean()
    
    return df

def add_trend_indicators(df, ticker_value):
    """Calculate trend indicators."""
    close_prices = df[('Close', ticker_value)]
    
    # MACD
    macd = MACD(close=close_prices)
    df.loc[:, ('MACD', ticker_value)] = macd.macd()
    df.loc[:, ('MACD_Signal', ticker_value)] = macd.macd_signal()
    df.loc[:, ('MACD_Hist', ticker_value)] = macd.macd_diff()
    
    return df

def add_price_derivatives(df, ticker_value):
    """Calculate price-based features."""
    close_prices = df[('Close', ticker_value)]
    high = df[('High', ticker_value)]
    low = df[('Low', ticker_value)]
    
    # Price changes
    df.loc[:, ('Price_Change', ticker_value)] = close_prices.diff()
    df.loc[:, ('Returns', ticker_value)] = close_prices.pct_change()
    
    # High-Low range
    df.loc[:, ('HL_Range', ticker_value)] = high - low
    df.loc[:, ('HL_Range_Pct', ticker_value)] = (high - low) / close_prices
    
    return df

def add_technical_indicators(df, ticker_value):
    """
    Add technical indicators to the DataFrame
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data
        ticker_value (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
      
    df = add_moving_averages(df, ticker_value)
    df = add_momentum_indicators(df, ticker_value)
    df = add_volatility_indicators(df, ticker_value)

    
    df = add_trend_indicators(df, ticker_value)
    
    df = add_price_derivatives(df, ticker_value)
    
    # Add lagged features
    df = add_lagged_features(df, ticker_value)
    
    return df

def get_feature_columns(df, model_type="linear_regression"):
    """
    Get feature columns based on model type
    """
    if model_type == "linear_regression":
        # For linear regression, use base features
        return ['Open', 'High', 'Low', 'Prev Close', 'Volume']
        
    else:  # For XGBoost and other models
        # Base features
        base_features = ['Open', 'High', 'Low', 'Close', 'Prev Close', 'Volume']
        
        # Technical indicators
        tech_features = [
            'RSI', 'BB_Upper', 'BB_Lower', 'ATR',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'Price_Change', 'Returns', 'HL_Range', 'HL_Range_Pct'
        ]
        
        # Lagged features
        lagged_features = [
            'Close_Lag_1', 'Close_Lag_2',
            'Volume_Lag_1', 'Volume_Lag_2'
        ]
        
        return base_features + tech_features + lagged_features
