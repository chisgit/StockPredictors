"""Run the Streamlit app with yfinance patched to simulate provider outage.

This script is a local/browser verification helper. It keeps the production
app code unchanged while letting Playwright exercise the real Streamlit UI
against the same error shape yfinance emits for rate limiting.
"""

import sys

import yfinance

try:
    import yfinance.shared as yf_shared

    yf_shared._ERRORS = {}
except Exception:
    pass


def fail_download(*args, **kwargs):
    raise RuntimeError("YFRateLimitError: Try after a while.")


yfinance.download = fail_download

sys.argv = [
    "streamlit",
    "run",
    "stock_predictors.py",
    "--server.port=8510",
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
]

from streamlit.web.cli import main

main()
