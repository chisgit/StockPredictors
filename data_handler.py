import yfinance as yf
import yfinance.shared as yf_shared
import pandas as pd
import os
import time
import json
import threading
from pathlib import Path
import shutil
from trace_utils import trace_event

KNOWN_TICKERS_FILE = "known_tickers.json"
_CACHE_FRESH_SECONDS = 30

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
    "yfratelimiterror",
    "runtimeerror",
    "rate limit",
    "rate-limit",
    "too many requests",
)
YFINANCE_PROVIDER_DOWN_MESSAGE = "Data Provider yfinance is currently down. Please try again later."


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
    trace_event("yfinance.cache_location", path=str(cache_dir), exists=cache_dir.exists())
    if cache_dir.exists():
        try:
            trace_event("yfinance.cache_entries", entries=[p.name for p in cache_dir.iterdir()])
        except Exception as exc:
            trace_event("yfinance.cache_listing_failed", error=str(exc))
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
        trace_event("yfinance.cache_cleared", path=str(cache_dir))


def _download_with_retry(ticker, start_date, end_date):
    """
    Download once, and retry after clearing cache if yfinance hits a cache schema error.
    """
    configure_yfinance_cache()
    try:
        trace_event(
            "yfinance.download.start",
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        trace_event(
            "yfinance.download.finish",
            ticker=ticker,
            empty=getattr(df, "empty", None),
            rows=len(df) if df is not None else None,
        )
        if _should_refresh_after_download(ticker, df):
            trace_event("yfinance.download.retryable_failure", ticker=ticker)
            clear_yfinance_cache()
            trace_event("yfinance.download.retry_start", ticker=ticker)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            trace_event(
                "yfinance.download.retry_finish",
                ticker=ticker,
                empty=getattr(df, "empty", None),
                rows=len(df) if df is not None else None,
            )
        return df
    except Exception as exc:
        message = str(exc).lower()
        trace_event(
            "yfinance.download.exception",
            ticker=ticker,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if os.environ.get("YFINANCE_TEST_CACHE_REFRESH", "").lower() in ("1", "true", "yes"):
            if _YF_TEST_CACHE_REFRESH_MESSAGE in message:
                trace_event("yfinance.test_cache_refresh", ticker=ticker, error=str(exc))
                clear_yfinance_cache()
                return yf.download(ticker, start=start_date, end=end_date, progress=False)

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
            trace_event("yfinance.cache_error", ticker=ticker, error=str(exc))
            clear_yfinance_cache()
            return yf.download(ticker, start=start_date, end=end_date, progress=False)
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

    should_refresh = any(marker in error_text for marker in _YF_RETRY_ERROR_MARKERS)
    if error_text:
        trace_event(
            "yfinance.shared_error",
            ticker=ticker_key,
            should_refresh=should_refresh,
            error=error_text,
        )
    return should_refresh


def yfinance_provider_error_text(ticker=None, exc=None):
    """
    Return yfinance provider/rate-limit error text when the latest failure looks
    external to the app. Used for user-facing messages without blocking search.
    """
    parts = []
    if exc is not None:
        parts.append(f"{type(exc).__name__}: {exc}")

    if ticker is not None and hasattr(yf_shared, "_ERRORS"):
        parts.append(str(yf_shared._ERRORS.get(str(ticker).upper(), "")))

    error_text = " ".join(part for part in parts if part).lower()
    if any(marker in error_text for marker in _YF_RETRY_ERROR_MARKERS):
        return error_text
    return ""


def is_yfinance_provider_down(ticker=None, exc=None):
    return bool(yfinance_provider_error_text(ticker, exc))


def invalidate_yfinance_cache():
    """
    Explicit cache reset hook you can call from a request handler or admin action.
    """
    clear_yfinance_cache()
    return True

def _load_known_tickers():
    if os.path.exists(KNOWN_TICKERS_FILE):
        try:
            with open(KNOWN_TICKERS_FILE) as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def _record_known_ticker(ticker):
    known = _load_known_tickers()
    if ticker not in known:
        known.add(ticker)
        with open(KNOWN_TICKERS_FILE, "w") as f:
            json.dump(sorted(known), f)


def is_known_ticker(ticker):
    return ticker in _load_known_tickers()


def custom_read_csv(file_path):
    """
    Read a cached CSV written by ``df.to_csv`` from a frame with MultiIndex
    (Type, Ticker) columns and a ``Date`` index.

    On disk that looks like::

        Price,Close,High,Low,Open,Volume   <- row 0: Type
        Ticker,AAPL,AAPL,AAPL,AAPL,AAPL     <- row 1: Ticker
        Date,,,,,                           <- row 2: leftover index-name row
        2008-01-02,5.83,...                 <- data

    So we read the two header rows as a MultiIndex, take column 0 as the
    index, drop the stray ``Date`` index-name row, and coerce the index to
    datetime.
    """
    df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    df.columns.names = ['Type', 'Ticker']

    # Row 2 ("Date,,,,,") survives as an index label of "Date"; drop it.
    df = df[df.index.astype(str) != 'Date']

    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # Numeric columns come back as object after the header gymnastics.
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def _normalize_columns(df):
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product(
            [["Price"], df.columns], names=["Type", "Ticker"]
        )
    return df


def fetch_data(ticker, end_date, start_date="2008-01-01", use_local_data=False):
    """Fetch stock data. Cache: CSV per ticker, delta-only after 30s staleness."""
    local_file = f"{ticker}_data.csv"

    if os.path.exists(local_file):
        file_age = time.time() - os.path.getmtime(local_file)
        if file_age < _CACHE_FRESH_SECONDS:
            trace_event("fetch_data.cache_hit", ticker=ticker, age_s=round(file_age, 1))
            return custom_read_csv(local_file)

        # Stale — delta fetch since last row date
        df = custom_read_csv(local_file)
        if df.empty:
            os.remove(local_file)
            trace_event("fetch_data.corrupt_cache_removed", ticker=ticker)
        else:
            last_date = df.index[-1]
            delta_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            trace_event("fetch_data.delta_start", ticker=ticker, delta_start=delta_start)
            delta_df = _download_with_retry(ticker, delta_start, end_date)
            if not delta_df.empty:
                delta_df = _normalize_columns(delta_df)
                df = pd.concat([df, delta_df])
                df = df[~df.index.duplicated(keep="last")]
                df.index.name = "Date"
                df.to_csv(local_file)
                trace_event("fetch_data.delta_saved", ticker=ticker, new_rows=len(delta_df))
            else:
                trace_event("fetch_data.delta_empty", ticker=ticker)
            _record_known_ticker(ticker)
            return df

    # No cache — full fetch
    trace_event("fetch_data.full_start", ticker=ticker, start_date=str(start_date), end_date=str(end_date))
    df = _download_with_retry(ticker, start_date, end_date)
    df = _normalize_columns(df)
    if not df.empty:
        df.index.name = "Date"
        df.to_csv(local_file)
        trace_event("fetch_data.full_saved", ticker=ticker, path=local_file, rows=len(df))
        _record_known_ticker(ticker)
    else:
        trace_event("fetch_data.full_empty", ticker=ticker)
        raise ValueError(f"No data retrieved for ticker {ticker} from yfinance")
    return df
