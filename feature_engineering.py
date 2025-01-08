import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def add_technical_indicators(df, ticker_value):
    """Add all technical indicators to the dataframe."""
    df = add_moving_averages(df, ticker_value)
    df = add_momentum_indicators(df, ticker_value)
    df = add_volatility_indicators(df, ticker_value)
    df = add_trend_indicators(df, ticker_value)
    df = add_price_derivatives(df, ticker_value)
    # Add lagged features
    df = add_lagged_features(df)
    return df

def add_lagged_features(df, base_features = ['Close', 'Volume'], lags=[1, 2]):
    """
    Create lagged features for specified columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        base_features (list): Features to create lags for
        lags (list): Lag periods to create
    
    Returns:
        list: Lagged feature column names
    """
    ticker_value = df.columns[1][1]
    lagged_features = []
    
    for feature in base_features:
        for lag in lags:
            lagged_col = (f'{feature}_Lag_{lag}', ticker_value)
            df[lagged_col] = df[(feature, ticker_value)].shift(lag)
            lagged_features.append(lagged_col)
    
    print(f"DEBUG - DataFrame Columns After: {df.columns.tolist()}")
    # Drop NaN rows created by lagging
    df.dropna(inplace=True)

def add_moving_averages(df, ticker_value):
    """Calculate various moving averages."""
    windows = [5, 10] # 20, 50, 100, 200]
    close_prices = df[('Close', ticker_value)]
    
    for window in windows:
        df[(f'SMA_{window}', ticker_value)] = close_prices.rolling(window=window).mean()
        df[(f'EMA_{window}', ticker_value)] = close_prices.ewm(span=window, adjust=False).mean()
    
    return df

def add_momentum_indicators(df, ticker_value):
    """Calculate momentum indicators."""
    close_prices = df[('Close', ticker_value)]
    
    # RSI
    rsi = RSIIndicator(close=close_prices, window=14)
    df[('RSI', ticker_value)] = rsi.rsi()
    
    # Rate of Change
    df[('ROC_5', ticker_value)] = close_prices.pct_change(periods=5)
    df[('ROC_10', ticker_value)] = close_prices.pct_change(periods=10)
    df[('ROC_20', ticker_value)] = close_prices.pct_change(periods=20)
    
    # Momentum
    df[('Momentum_5', ticker_value)] = close_prices.diff(5)
    df[('Momentum_10', ticker_value)] = close_prices.diff(10)
    
    return df

def add_volatility_indicators(df, ticker_value):
    """Calculate volatility indicators."""
    close_prices = df[('Close', ticker_value)]
    
    # Bollinger Bands
    bb = BollingerBands(close=close_prices, window=20, window_dev=2)
    df[('BB_Upper', ticker_value)] = bb.bollinger_hband()
    df[('BB_Lower', ticker_value)] = bb.bollinger_lband()
    df[('BB_Middle', ticker_value)] = bb.bollinger_mavg()
    
    # Average True Range (ATR)
    high = df[('High', ticker_value)]
    low = df[('Low', ticker_value)]
    close = df[('Close', ticker_value)]
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[('ATR', ticker_value)] = true_range.rolling(window=14).mean()
    
    return df

def add_trend_indicators(df, ticker_value):
    """Calculate trend indicators."""
    close_prices = df[('Close', ticker_value)]
    
    # MACD
    macd = MACD(close=close_prices)
    df[('MACD', ticker_value)] = macd.macd()
    df[('MACD_Signal', ticker_value)] = macd.macd_signal()
    df[('MACD_Hist', ticker_value)] = macd.macd_diff()
    
    return df

def add_price_derivatives(df, ticker_value):
    """Calculate price-based features."""
    # Price changes
    df[('Price_Change', ticker_value)] = df[('Close', ticker_value)].diff()
    df[('Returns', ticker_value)] = df[('Close', ticker_value)].pct_change()
    
    # High-Low range
    df[('HL_Range', ticker_value)] = df[('High', ticker_value)] - df[('Low', ticker_value)]
    df[('HL_Range_Pct', ticker_value)] = (df[('High', ticker_value)] - df[('Low', ticker_value)]) / df[('Close', ticker_value)]
    
    return df

def prepare_features(df):
    """
    Prepare all features for model training, including next day's close.
    """
    if df.empty:
        return df
    
    ticker_value = df.columns[1][1]
    
    # Create next day's close (target variable)
    df[('Next_Day_Close', ticker_value)] = df[('Close', ticker_value)].shift(-1)
    
    # Add all technical indicators
    df = add_technical_indicators(df, ticker_value)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def get_feature_columns(df, model_type):
    """
    Get list of feature columns to use for training.
    """
    ticker_value = df.columns[1][1]
    
    print(f"DEBUG - get_feature_columns() - All df.columns:\n{df.columns.tolist()}")
    # Base price and volume features
    #base_features = ['Open', 'High', 'Low', 'Close', 'Prev Close', 'Volume']
    base_features = ['Close', 'Prev Close', 'Volume']

    # Technical indicators
    technical_features = [col[0] for col in df.columns 
                         if col[0] in ['SMA_20', 'EMA', 'RSI', 'ROC', 'Momentum',
                                     'BB', 'ATR', 'MACD', 'MACD_Signal', 'MACD_Hist','BB_Upper', 'BB_Lower', 'Price_Change',
                                     'Returns', 'HL_Range', 'HL_Range_Pct']]

    # Lagged features
    lagged_features = [col[0] for col in df.columns if 'Lag' in col[0]]
    
     # DEBUG-XBoost: Print technical and lagged features
    print(f"DEBUG-XBoost - Base Features: {base_features}")
    print(f"DEBUG-XBoost - Technical Features: {technical_features}")
    print(f"DEBUG-XBoost - Lagged Features: {lagged_features}")
    
    if model_type == "linear_regression": 
        feature_columns = base_features
        print(f"Base Feature columns: {feature_columns}")
    # Combine all features
    elif model_type == "xgboost":
        feature_columns = base_features + technical_features + lagged_features
        # DEBUG-XBoost: Print final feature columns
        print(f"DEBUG-XBoost - Feature Columns for XGBoost: {feature_columns}")


    return feature_columns
