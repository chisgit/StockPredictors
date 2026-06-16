import yfinance as yf
import yfinance.shared as yf_shared
import pandas as pd
import os
import threading
from pathlib import Path
import shutil

try:
    from yfinance import cache as yf_cache
except Exception:
    yf_cache = None

_YF_CACHE_LOCK = threading.Lock()
_YF_TEST_CACHE_REFRESH_MESSAGE = "placeholder"
_YF_RETRY_ERROR_MARKERS = (
    "try after a while",
    "failed download",
    "yfr",
    "runtimeerror",
)


def _default_yfinance_cache_dir():
    """
    Pick the yfinance cache location used by the remote service.
    """
    return Path.home() / ".cache" / "py-yfinance"


def configure_yfinance_cache(cache_dir=None):
    """
    Point yfinance at a writable cache location before any downloads happen.
    """
    cache_dir = Path(cache_dir) if cache_dir else _default_yfinance_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if yf_cache is not None and hasattr(yf_cache, "set_cache_location"):
        yf_cache.set_cache_location(str(cache_dir))
    return cache_dir


def debug_yfinance_cache_location():
    """
    Print the resolved cache path and current filesystem state for Render logs.
    """
    cache_dir = configure_yfinance_cache()
    print(f"yfinance cache path: {cache_dir}")
    print(f"yfinance cache exists: {cache_dir.exists()}")
    if cache_dir.exists():
        try:
            print(f"yfinance cache entries: {[p.name for p in cache_dir.iterdir()]}")
        except Exception as exc:
            print(f"yfinance cache listing failed: {exc}")
    return cache_dir


def clear_yfinance_cache():
    """
    Remove local yfinance cache files so the library can rebuild them cleanly.
    """
    cache_dir = configure_yfinance_cache()
    with _YF_CACHE_LOCK:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleared yfinance cache: {cache_dir}")


def _download_with_retry(ticker, start_date, end_date):
    """
    Download once, and retry after clearing cache if yfinance hits a cache schema error.
    """
    configure_yfinance_cache()
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if _should_refresh_after_download(ticker, df):
            print(f"yfinance returned a retryable failure for {ticker}; clearing cache and retrying once")
            clear_yfinance_cache()
            df = yf.download(ticker, start=start_date, end=end_date)
        return df
    except Exception as exc:
        message = str(exc).lower()
        if os.environ.get("YFINANCE_TEST_CACHE_REFRESH", "").lower() in ("1", "true", "yes"):
            if _YF_TEST_CACHE_REFRESH_MESSAGE in message:
                print(f"yfinance test cache refresh trigger for {ticker}: {exc}")
                clear_yfinance_cache()
                return yf.download(ticker, start=start_date, end=end_date)

        cache_error_signals = (
            "no such table",
            "_kv",
            "operationalerror",
            "sqlite",
            "database is locked",
            "disk i/o error",
            "malformed",
        )
        if any(signal in message for signal in cache_error_signals) or "operationalerror" in type(exc).__name__.lower():
            print(f"yfinance cache error for {ticker}: {exc}")
            clear_yfinance_cache()
            return yf.download(ticker, start=start_date, end=end_date)
        raise


def _should_refresh_after_download(ticker, df):
    """
    Detect the 'Failed download' / 'Try after a while.' path that yfinance logs
    via shared._ERRORS while still returning a frame.
    """
    if df is None:
        return True

    if not getattr(df, "empty", True):
        return False

    ticker_key = str(ticker).upper()
    error_text = ""
    if hasattr(yf_shared, "_ERRORS"):
        error_text = str(yf_shared._ERRORS.get(ticker_key, "")).lower()

    return any(marker in error_text for marker in _YF_RETRY_ERROR_MARKERS)


def invalidate_yfinance_cache():
    """
    Explicit cache reset hook you can call from a request handler or admin action.
    """
    clear_yfinance_cache()
    return True

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
        df = _download_with_retry(ticker, start_date, end_date)
        
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
            raise ValueError(f"No data retrieved for ticker {ticker} from yfinance")

    return df
