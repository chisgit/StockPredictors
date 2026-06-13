import os
from datetime import datetime, time as dt_time, timedelta
from functools import lru_cache
import pytz
import pandas as pd
import pandas_market_calendars as mcal

# Define NYSE market hours (Global constants)
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
NYSE_TIMEZONE = pytz.timezone('America/New_York')

# NYSE calendar (XNYS) — knows weekends AND holidays. Built once, reused.
_NYSE_CALENDAR = mcal.get_calendar("XNYS")


@lru_cache(maxsize=8)
def is_trading_day(date):
    """True if `date` is an NYSE trading day (weekends and holidays excluded).

    Cached per date so repeated calls within a run don't rebuild the schedule.
    """
    schedule = _NYSE_CALENDAR.schedule(start_date=date, end_date=date)
    return not schedule.empty

def get_nyse_datetime():
    """Current datetime in NYSE timezone.

    Manual/QA override: set env var MARKET_NOW to an ISO datetime (e.g.
    "2026-05-25T10:00") to force the app to believe it is that NYSE-local
    instant. Used to eyeball market-status states in the UI without waiting for
    a real weekend/holiday. Unset in production -> real wall clock.
    """
    override = os.environ.get("MARKET_NOW")
    if override:
        return NYSE_TIMEZONE.localize(datetime.fromisoformat(override))
    return datetime.now(pytz.UTC).astimezone(NYSE_TIMEZONE)

def get_nyse_date():
    """Get current date in NYSE timezone"""
    return get_nyse_datetime().date()

def get_nyse_time():
    """Get current time in NYSE timezone"""
    return get_nyse_datetime().time()

def market_status():
    """Current NYSE market status: BEFORE_MARKET_OPEN, MARKET_OPEN, or
    AFTER_MARKET_CLOSE.

    Holiday- and weekend-aware via the NYSE (XNYS) calendar. BEFORE_MARKET_OPEN
    only occurs on a trading day before the bell; any non-trading day (weekend
    or holiday) is AFTER_MARKET_CLOSE, since the last market event was a close.
    The user only ever sees "open" vs "closed"; when closed the UI shows the
    last completed session's date (from the data), so the before/after split is
    internal only.
    """
    now = get_nyse_datetime()

    # Non-trading day (weekend/holiday) -> market is closed (after last close).
    if not is_trading_day(now.date()):
        return "AFTER_MARKET_CLOSE"

    current_time = now.time()
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