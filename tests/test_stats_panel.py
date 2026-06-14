"""Isolated tests for the grouped stats panel (plan section 3).

Smoke-tests the HTML output of create_grid_display() per market status.
Locks: all six metric cards present, close label reflects market status.
"""


from datetime import date

import pytest

from render_helpers import create_grid_display


SAMPLE = dict(
    open_val=100.0,
    high_val=105.0,
    low_val=98.0,
    prev_close_val=99.0,
    close_val=103.0,
    volume=1_000_000,
)
SESSION_DATE = date(2026, 6, 12)  # a Friday


def _html(monkeypatch, status, session_date=SESSION_DATE):
    monkeypatch.setattr("render_helpers.market_status", lambda: status)
    return create_grid_display(**SAMPLE, session_date=session_date)


# --- metric cards present ----------------------------------------------------

@pytest.mark.parametrize("status", ["MARKET_OPEN", "BEFORE_MARKET_OPEN", "AFTER_MARKET_CLOSE"])
def test_all_metrics_present(monkeypatch, status):
    html = _html(monkeypatch, status)
    for label in ("OPEN", "HIGH", "LOW", "PREV CLOSE", "VOLUME"):
        assert label in html.upper()


def test_close_label_open(monkeypatch):
    html = _html(monkeypatch, "MARKET_OPEN")
    assert "LAST TRADED" in html.upper()


def test_close_label_closed(monkeypatch):
    html = _html(monkeypatch, "BEFORE_MARKET_OPEN")
    assert "CLOSE" in html.upper()
    assert "LAST TRADED" not in html.upper()
