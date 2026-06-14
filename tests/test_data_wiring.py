"""Isolated tests for §1f data-wiring — before-open prediction accuracy caption.

Locks: _session_ref_caption() emits the correct text for the closed-market
accuracy recap shown below prediction strips.

State contract:
  MARKET_OPEN        → no caption (not tested here — caption omitted in caller)
  BEFORE_MARKET_OPEN → caption with session date + actual close
  AFTER_MARKET_CLOSE → caption with session date + actual close
"""
from datetime import date

import pytest

from render_ui import _session_ref_caption


FRIDAY = date(2026, 6, 12)
MONDAY = date(2026, 6, 15)
CLOSE = 188.50


# --- caption content ---------------------------------------------------------

def test_caption_includes_delta_symbol():
    text = _session_ref_caption(FRIDAY, CLOSE)
    assert "Δ" in text


def test_caption_includes_actual_close_label():
    text = _session_ref_caption(FRIDAY, CLOSE)
    assert "actual close" in text


def test_caption_includes_close_price():
    text = _session_ref_caption(FRIDAY, CLOSE)
    assert "$188.50" in text


def test_caption_includes_weekday():
    text = _session_ref_caption(FRIDAY, CLOSE)
    assert "Friday" in text


def test_caption_includes_month_and_day():
    text = _session_ref_caption(FRIDAY, CLOSE)
    assert "Jun 12" in text


# --- different session dates -------------------------------------------------

def test_caption_monday_date():
    text = _session_ref_caption(MONDAY, 200.00)
    assert "Monday" in text
    assert "Jun 15" in text


def test_caption_close_formats_two_decimals():
    text = _session_ref_caption(FRIDAY, 99.9)
    assert "$99.90" in text


def test_caption_large_close():
    text = _session_ref_caption(FRIDAY, 1234.56)
    assert "$1234.56" in text


# --- round-trip check --------------------------------------------------------

@pytest.mark.parametrize("session_date,close", [
    (date(2026, 1, 2), 150.00),   # Friday (New Year's Day observed — first trading day)
    (date(2026, 3, 27), 210.75),  # Friday
    (date(2026, 11, 25), 88.30),  # Wednesday before Thanksgiving
])
def test_caption_parametrized(session_date, close):
    text = _session_ref_caption(session_date, close)
    assert "Δ" in text
    assert "actual close" in text
    assert f"${close:.2f}" in text
