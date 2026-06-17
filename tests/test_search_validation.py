import pandas as pd

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


def test_provider_down_message_text_is_stable():
    assert (
        render_helpers.YFINANCE_PROVIDER_DOWN_MESSAGE
        == "yfinance data provider is down, please try later"
    )
