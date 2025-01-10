from datetime import datetime, time as dt_time, timedelta
import pytz
import pandas as pd

# Define NYSE market hours (Global constants)
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
NYSE_TIMEZONE = pytz.timezone('America/New_York')

def get_nyse_datetime():
    """Get current datetime in NYSE timezone"""
    return datetime.now(pytz.UTC).astimezone(NYSE_TIMEZONE) #- timedelta(hours=13) #for testing

def get_nyse_date():
    """Get current date in NYSE timezone"""
    return get_nyse_datetime().date()

def get_nyse_time():
    """Get current time in NYSE timezone"""
    return get_nyse_datetime().time()

def market_status():
    """Check the current market status based on NYSE time"""
    current_time = get_nyse_time()
    
    if current_time < MARKET_OPEN:
        return "BEFORE_MARKET_OPEN"
    elif MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return "MARKET_OPEN"
    else:
        return "AFTER_MARKET_CLOSE"

def get_last_row(data):
    if isinstance(data, pd.DataFrame) and not data.empty:
        return data.tail(1)  # Returns last row as DataFrame
    elif isinstance(data, pd.Series) and not data.empty:
        return data.iloc[-1]  # Returns last element as Series
    return None  # Returns None for empty or unsupported types