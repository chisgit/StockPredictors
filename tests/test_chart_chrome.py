"""Isolated tests for §6 chart chrome strip.

Locks: eyebrow label and OHLC caption removed; ticker title kept.
Tested via generate_chart_widget_html() — pure function, no Streamlit dependency.
"""
from render_helpers import generate_chart_widget_html

DUMMY_JSON = "[]"


def test_eyebrow_label_removed():
    html = generate_chart_widget_html("AAPL", DUMMY_JSON)
    assert "TradingView Style Chart" not in html


def test_ohlc_caption_removed():
    html = generate_chart_widget_html("AAPL", DUMMY_JSON)
    assert "OHLC candles from local market data" not in html


def test_ticker_title_kept():
    html = generate_chart_widget_html("AAPL", DUMMY_JSON)
    assert "AAPL · Last 10 Trading Days" in html


def test_ticker_uppercased():
    html = generate_chart_widget_html("msft", DUMMY_JSON)
    assert "MSFT · Last 10 Trading Days" in html


def test_chart_json_embedded():
    html = generate_chart_widget_html("AAPL", '[{"time": 1, "open": 1}]')
    assert '[{"time": 1, "open": 1}]' in html


def test_chart_container_id_uses_ticker():
    html = generate_chart_widget_html("TSLA", DUMMY_JSON)
    assert 'id="tv_chart_TSLA"' in html
