"""
This module contains project-wide rules, guidelines, and configurations that should be followed
throughout the Stock Predictor application.
"""

# Market Rules
MARKET_HOURS = {
    'open': '09:30',  # Market opening time (EST)
    'close': '16:00'  # Market closing time (EST)
}

# Data Processing Rules
DATA_RULES = {
    'min_historical_days': 30,  # Minimum number of days of historical data required
    'max_historical_days': 252,  # Maximum number of days of historical data to use (1 trading year)
    'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],  # Required columns for stock data
    'default_time_interval': '1d'  # Default time interval for data collection
}

# Model Rules
MODEL_RULES = {
    'train_test_split': 0.8,  # 80% training, 20% testing
    'validation_size': 0.1,   # 10% of training data for validation
    'prediction_horizon': 1,   # Number of days to predict into the future
    'minimum_samples': 30,    # Minimum number of samples required for training
    'acceptable_missing_data': 0.1  # Maximum acceptable proportion of missing data (10%)
}

# Feature Engineering Rules
FEATURE_RULES = {
    'technical_indicators': [
        'SMA',      # Simple Moving Average
        'EMA',      # Exponential Moving Average
        'RSI',      # Relative Strength Index
        'MACD',     # Moving Average Convergence Divergence
        'BB'        # Bollinger Bands
    ],
    'window_sizes': [5, 10, 20, 50],  # Window sizes for technical indicators
    'normalize_features': True,        # Whether to normalize features
    'handle_missing_values': 'ffill'   # Forward fill for missing values
}

# UI Display Rules
UI_RULES = {
    'max_tickers': 10,        # Maximum number of tickers that can be selected at once
    'refresh_interval': 60,   # Data refresh interval in seconds
    'price_decimals': 2,     # Number of decimal places for price display
    'volume_format': 'comma', # Format for volume numbers
    'chart_height': 400,     # Default chart height in pixels
    'default_tickers': ['AAPL', 'GOOGL', 'MSFT']  # Default tickers to show
}

# Error Handling Rules
ERROR_RULES = {
    'max_retries': 3,        # Maximum number of retries for failed API calls
    'retry_delay': 5,        # Delay between retries in seconds
    'timeout': 30,           # Timeout for API calls in seconds
    'log_errors': True       # Whether to log errors
}

# Data Quality Rules
QUALITY_RULES = {
    'min_price': 0.01,       # Minimum acceptable stock price
    'max_price_change': 50,  # Maximum acceptable price change percentage in a day
    'volume_threshold': 1000, # Minimum daily volume threshold
    'gap_threshold': 5       # Maximum acceptable gap in days for missing data
}

def get_all_rules():
    """
    Returns a dictionary containing all rules defined in this module.
    """
    return {
        'market_hours': MARKET_HOURS,
        'data_rules': DATA_RULES,
        'model_rules': MODEL_RULES,
        'feature_rules': FEATURE_RULES,
        'ui_rules': UI_RULES,
        'error_rules': ERROR_RULES,
        'quality_rules': QUALITY_RULES
    }

def validate_rules():
    """
    Validates that all rules are properly formatted and within acceptable ranges.
    Returns True if all rules are valid, raises ValueError otherwise.
    """
    try:
        # Validate market hours
        for time_str in MARKET_HOURS.values():
            hours, minutes = map(int, time_str.split(':'))
            if not (0 <= hours < 24 and 0 <= minutes < 60):
                raise ValueError(f"Invalid market hours: {time_str}")

        # Validate numeric ranges
        if not (0 < MODEL_RULES['train_test_split'] < 1):
            raise ValueError("Train-test split must be between 0 and 1")
        
        if not (0 < MODEL_RULES['validation_size'] < 1):
            raise ValueError("Validation size must be between 0 and 1")
        
        if DATA_RULES['min_historical_days'] >= DATA_RULES['max_historical_days']:
            raise ValueError("Minimum historical days must be less than maximum historical days")
        
        if UI_RULES['max_tickers'] <= 0:
            raise ValueError("Maximum number of tickers must be positive")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Rule validation failed: {str(e)}")