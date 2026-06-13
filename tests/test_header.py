"""Isolated tests for the single status-driven header (plan section 1).

Smoke-tests the rendered HTML of generate_market_status_header() per market
status. Locks: one left-aligned header, status-driven title, inline icon, and
the subtitle binary (date shown only when Closed-after-close). Pure function,
no Streamlit / no time dependence, so it is deterministic.
"""
import pytest

from display_market_status import generate_market_status_header

DATE = "Friday, June 12"


def test_before_open_title_names_session_date():
    html = generate_market_status_header("BEFORE_MARKET_OPEN", DATE)
    assert "⏳" in html
    assert f"Predictions for {DATE}" in html
    # title already carries the date → no duplicate subtitle date
    assert "Last traded session" not in html


def test_market_open_live_no_date():
    html = generate_market_status_header("MARKET_OPEN", DATE)
    assert "🔔" in html
    assert "Live — Last Traded" in html
    # Open → no session-date subtitle
    assert DATE not in html
    assert "Last traded session" not in html


def test_after_close_title_and_date_subtitle():
    html = generate_market_status_header("AFTER_MARKET_CLOSE", DATE)
    assert "🔴" in html
    assert "Today's Close Predictions" in html
    # Closed-after-close → date appears in the subtitle
    assert f"Last traded session — {DATE}" in html


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
