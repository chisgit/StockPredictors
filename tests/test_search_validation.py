import pandas as pd
import streamlit as st

import render_helpers


def test_validate_ticker_accepts_non_empty_yfinance_data(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame({"Close": [100.0]})

    monkeypatch.setattr(render_helpers.yf, "download", fake_download)
    monkeypatch.setattr(render_helpers, "_yfinance_shared_error", lambda ticker: "")

    assert render_helpers.validate_ticker_with_yfinance("IBM") == (True, False)


def test_validate_ticker_rejects_empty_data_without_provider_error(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(render_helpers.yf, "download", fake_download)
    monkeypatch.setattr(render_helpers, "_yfinance_shared_error", lambda ticker: "")

    assert render_helpers.validate_ticker_with_yfinance("NOTREAL") == (False, False)


def test_validate_ticker_reports_provider_down_from_shared_error(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(render_helpers.yf, "download", fake_download)
    monkeypatch.setattr(
        render_helpers,
        "_yfinance_shared_error",
        lambda ticker: "Failed download: YFRateLimitError('Try after a while.')",
    )

    assert render_helpers.validate_ticker_with_yfinance("TSLA") == (False, True)


def test_validate_ticker_reports_provider_down_from_exception(monkeypatch):
    def fake_download(*args, **kwargs):
        raise RuntimeError("YFRateLimitError: Try after a while.")

    monkeypatch.setattr(render_helpers.yf, "download", fake_download)

    assert render_helpers.validate_ticker_with_yfinance("TSLA") == (False, True)


def test_validate_ticker_reports_provider_down_from_yfinance_rate_limit_exception(monkeypatch):
    YFRateLimitError = type("YFRateLimitError", (Exception,), {})

    def fake_download(*args, **kwargs):
        raise YFRateLimitError("Too Many Requests")

    monkeypatch.setattr(render_helpers.yf, "download", fake_download)

    assert render_helpers.validate_ticker_with_yfinance("TSLA") == (False, True)


def test_provider_down_message_text_is_stable():
    assert (
        render_helpers.YFINANCE_PROVIDER_DOWN_MESSAGE
        == "Market data is temporarily unavailable. Please try again later."
    )
    assert "yfinance" not in render_helpers.YFINANCE_PROVIDER_DOWN_MESSAGE.lower()


def test_search_skips_yfinance_validation_when_ticker_already_selected(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Already selected ticker should not call yfinance validation")

    st.session_state.clear()
    st.session_state.new_ticker = ""
    st.session_state.selected_tickers = ["TSLA"]
    st.session_state.tickers = ["TSLA", "AAPL"]

    monkeypatch.setattr(render_helpers, "validate_ticker_with_yfinance", fail_if_called)

    assert render_helpers.search_and_add_ticker("tsla") is True
    assert st.session_state.selected_tickers == ["TSLA"]
    assert st.session_state.tickers == ["TSLA", "AAPL"]
    assert st.session_state.last_processed_ticker_search == "TSLA"


def test_search_exception_uses_generic_customer_message(monkeypatch):
    messages = []

    def fail_validation(*args, **kwargs):
        raise RuntimeError("secret upstream details")

    st.session_state.clear()
    st.session_state.new_ticker = ""
    st.session_state.selected_tickers = []
    st.session_state.tickers = ["TSLA", "AAPL"]

    monkeypatch.setattr(render_helpers, "validate_ticker_with_yfinance", fail_validation)
    monkeypatch.setattr(render_helpers.st, "error", messages.append)

    assert render_helpers.search_and_add_ticker("ibm") is False
    assert messages == [render_helpers.YFINANCE_PROVIDER_DOWN_MESSAGE]
    assert "secret upstream details" not in messages[0]
