"""Isolated tests for the single status-driven header (plan section 1).

Smoke-tests the rendered HTML of generate_market_status_header() per market
state. Locks: one left-aligned header, status-driven title, inline icon, and
the after-close trading-day vs weekend/holiday split. Pure function, no
Streamlit / no time dependence, so it is deterministic.

State -> header copy:
  BEFORE_MARKET_OPEN          -> "Market is Closed" + date subtitle
  MARKET_OPEN                 -> "Market is Open" + "Displaying Predictions for Today's Close" subtitle
  AFTER_MARKET_CLOSE, trading -> "Market is Closed" + finality subtitle
  AFTER_MARKET_CLOSE, off-day -> "Market is Closed (Weekend/Holiday)" + date subtitle
"""
from datetime import date

import pytest

from display_market_status import generate_market_status_header, generate_next_day_header
from utils import next_trading_day

DATE = "Friday, June 12"


def test_before_open_shows_last_traded_date():
    html = generate_market_status_header("BEFORE_MARKET_OPEN", DATE)
    assert "🔴" in html
    # status title on its own line; date in a separate muted subtitle (uses theme token)
    assert "Market is Closed" in html
    assert f"Displaying Predictions for {DATE}" in html
    assert "color: #64748b" in html  # dark theme text_delta_label
    assert " - Displaying" not in html


def test_market_open_live_subtitle_no_date():
    html = generate_market_status_header("MARKET_OPEN", DATE)
    assert "🔔" in html
    assert "Market is Open" in html
    assert "Displaying Predictions for Today's Close" in html
    assert DATE not in html


def test_after_close_trading_day_closed_title_and_finality_subtitle():
    html = generate_market_status_header(
        "AFTER_MARKET_CLOSE", DATE, today_is_trading_day=True
    )
    assert "🔴" in html
    assert "Market is Closed" in html
    assert "Today's Close Predictions and Actuals" in html
    # today's close is final → no last-traded date in the title
    assert DATE not in html


def test_after_close_weekend_holiday_shows_last_traded_date():
    html = generate_market_status_header(
        "AFTER_MARKET_CLOSE", DATE, today_is_trading_day=False
    )
    assert "🔴" in html
    assert "Market is Closed (Weekend/Holiday)" in html
    assert f"Displaying Predictions for {DATE}" in html
    assert " - Displaying" not in html


def test_header_is_centered_single_block():
    html = generate_market_status_header("AFTER_MARKET_CLOSE", DATE)
    assert "text-align: center" in html
    assert "text-align: left" not in html
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


# --- next-day (forward) section header --------------------------------------
def test_next_day_header_is_forward_no_displaying_cue():
    # forward section never uses the "Displaying" past-session cue
    html = generate_next_day_header("Monday, June 15")
    assert html == "Next Day's Close Predictions Monday, June 15"
    assert "Displaying" not in html


# --- next_trading_day: concrete target session (real XNYS calendar) ---------
@pytest.mark.parametrize("last_session, expected, desc", [
    (date(2026, 6, 12), date(2026, 6, 15), "Fri -> Mon (skip weekend) = today before open"),
    (date(2026, 6, 15), date(2026, 6, 16), "Mon -> Tue (normal next session)"),
    (date(2026, 5, 22), date(2026, 5, 26), "Fri before Memorial Day -> Tue (skip Mon holiday)"),
    (date(2026, 7, 2),  date(2026, 7, 6),  "Thu -> Mon (skip Jul 3 holiday + weekend)"),
])
def test_next_trading_day(last_session, expected, desc):
    assert next_trading_day(last_session) == expected, desc
