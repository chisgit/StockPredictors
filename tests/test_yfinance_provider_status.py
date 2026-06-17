import data_handler


def test_provider_down_detects_try_after_exception():
    exc = RuntimeError("YFRateLimitError: Try after a while.")

    assert data_handler.is_yfinance_provider_down("TSLA", exc)


def test_provider_down_detects_shared_yfinance_error(monkeypatch):
    monkeypatch.setattr(
        data_handler.yf_shared,
        "_ERRORS",
        {"IBM": "Failed download: YFRateLimitError('Try after a while.')"},
        raising=False,
    )

    assert data_handler.is_yfinance_provider_down("IBM")


def test_provider_down_ignores_plain_invalid_ticker_error(monkeypatch):
    monkeypatch.setattr(data_handler.yf_shared, "_ERRORS", {"FAKE": ""}, raising=False)
    exc = ValueError("No data retrieved for ticker FAKE from yfinance")

    assert not data_handler.is_yfinance_provider_down("FAKE", exc)


def test_provider_down_message_text_is_stable():
    assert (
        data_handler.YFINANCE_PROVIDER_DOWN_MESSAGE
        == "yfinance data provider is down, please try later"
    )
