"""Isolated tests for the single status-driven header (plan section 1).

Smoke-tests the rendered HTML of generate_market_status_header() per market
state. Locks: one left-aligned header, status-driven title, inline icon, and
the after-close trading-day vs weekend/holiday split. Pure function, no
Streamlit / no time dependence, so it is deterministic.

State → header copy:
  BEFORE_MARKET_OPEN          → "Market Closed - Displaying Predictions for <date>"
  MARKET_OPEN                 → "Live — Last Traded" (no date)
  AFTER_MARKET_CLOSE, trading → "Today's Close Predictions and Actuals" (no date)
  AFTER_MARKET_CLOSE, off-day → "Market Closed Weekend/Holiday - Displaying
                                 Predictions for <date>"
"""
import pytest

from display_market_status import generate_market_status_header

DATE = "Friday, June 12"


def test_before_open_shows_last_traded_date():
    html = generate_market_status_header("BEFORE_MARKET_OPEN", DATE)
    assert "🔴" in html
    assert f"Market Closed - Displaying Predictions for {DATE}" in html


def test_market_open_live_no_date():
    html = generate_market_status_header("MARKET_OPEN", DATE)
    assert "🔔" in html
    assert "Live — Last Traded" in html
    assert DATE not in html


def test_after_close_trading_day_predictions_and_actuals():
    html = generate_market_status_header(
        "AFTER_MARKET_CLOSE", DATE, today_is_trading_day=True
    )
    assert "🔴" in html
    assert "Today's Close Predictions and Actuals" in html
    # today's close is final → no last-traded date in the title
    assert DATE not in html


def test_after_close_weekend_holiday_shows_last_traded_date():
    html = generate_market_status_header(
        "AFTER_MARKET_CLOSE", DATE, today_is_trading_day=False
    )
    assert "🔴" in html
    assert f"Market Closed Weekend/Holiday - Displaying Predictions for {DATE}" in html


def test_header_is_left_aligned_single_block():
    html = generate_market_status_header("AFTER_MARKET_CLOSE", DATE)
    assert "text-align: left" in html
    assert "text-align: center" not in html
    # one header element, not a stacked date <h2> + <h3>
    assert html.count("<h3") == 1
    assert "<h2" not in html


def test_hardcoded_finality_string_removed():
    # the old "Today's closing prices are final." copy must be gone
    for status in ("BEFORE_MARKET_OPEN", "MARKET_OPEN", "AFTER_MARKET_CLOSE"):
        html = generate_market_status_header(status, DATE)
        assert "are final" not in html


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        generate_market_status_header("NONSENSE", DATE)
