from datetime import datetime, time as dt_time, timedelta
import pytz

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