"""Isolated tests for NYSE market-status logic (plan section 0).

These lock the behavior of utils.market_status() / utils.is_trading_day() so a
change in any other branch (header, cards, theme, ...) that accidentally
touches this logic is caught by `pytest`. Time is injected by monkeypatching
get_nyse_datetime, so the suite is deterministic and does not depend on when it
runs.
"""
from datetime import datetime, date
import pytz
import pytest

import utils

NYSE_TZ = pytz.timezone("America/New_York")


def _at(y, m, d, hh, mm):
    """Build a NYSE-tz-aware datetime for injection."""
    return NYSE_TZ.localize(datetime(y, m, d, hh, mm))


@pytest.fixture
def freeze_now(monkeypatch):
    """Return a setter that pins utils.get_nyse_datetime() to a fixed instant."""
    def _set(dt):
        monkeypatch.setattr(utils, "get_nyse_datetime", lambda: dt)
    return _set


# --- is_trading_day: weekends + holidays (real XNYS calendar) ---------------
@pytest.mark.parametrize("d, expected, desc", [
    (date(2026, 6, 15), True,  "Monday, normal trading day"),
    (date(2026, 6, 13), False, "Saturday"),
    (date(2026, 6, 14), False, "Sunday"),
    (date(2026, 5, 25), False, "Memorial Day (Mon)"),
    (date(2026, 1, 1),  False, "New Year's Day"),
    (date(2026, 7, 3),  False, "July 4 observed (Fri)"),
])
def test_is_trading_day(d, expected, desc):
    assert utils.is_trading_day(d) is expected, desc


# --- market_status: trading-day time boundaries -----------------------------
# 2026-06-15 is a Monday (confirmed trading day above).
@pytest.mark.parametrize("hh, mm, expected, desc", [
    (8, 0,   "BEFORE_MARKET_OPEN", "before the 09:30 bell"),
    (9, 29,  "BEFORE_MARKET_OPEN", "one minute before open"),
    (9, 30,  "MARKET_OPEN",        "exactly at open"),
    (12, 0,  "MARKET_OPEN",        "midday"),
    (16, 0,  "MARKET_OPEN",        "exactly at close (inclusive)"),
    (16, 1,  "AFTER_MARKET_CLOSE", "one minute after close"),
    (20, 0,  "AFTER_MARKET_CLOSE", "evening"),
])
def test_status_on_trading_day(freeze_now, hh, mm, expected, desc):
    freeze_now(_at(2026, 6, 15, hh, mm))
    assert utils.market_status() == expected, desc


# --- market_status: non-trading days are always AFTER_MARKET_CLOSE ----------
# (the corrected logic: weekend/holiday != BEFORE_MARKET_OPEN)
@pytest.mark.parametrize("y, m, d, hh, mm, desc", [
    (2026, 6, 13, 8, 0,  "Saturday morning"),
    (2026, 6, 13, 12, 0, "Saturday midday"),
    (2026, 6, 14, 23, 0, "Sunday night"),
    (2026, 5, 25, 10, 0, "Memorial Day during would-be market hours"),
    (2026, 1, 1, 11, 0,  "New Year's Day during would-be market hours"),
])
def test_status_on_non_trading_day(freeze_now, y, m, d, hh, mm, desc):
    freeze_now(_at(y, m, d, hh, mm))
    assert utils.market_status() == "AFTER_MARKET_CLOSE", desc
