"""Isolated tests for the grouped stats panel (plan section 3).

Smoke-tests the HTML output of create_grid_display() per market status.
Locks: bordered panel wrapper present, status-aware header text correct,
all six metric cards present.

State → panel header:
  MARKET_OPEN        → "Today's Data"
  BEFORE_MARKET_OPEN → session date string  (e.g. "Friday, June 12")
  AFTER_MARKET_CLOSE → session date string
  session_date=None  → "Last Session" fallback
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


# --- panel wrapper -----------------------------------------------------------

def test_panel_border_present(monkeypatch):
    html = _html(monkeypatch, "MARKET_OPEN")
    assert "border-radius: 18px" in html


# --- header text -------------------------------------------------------------

def test_header_market_open(monkeypatch):
    html = _html(monkeypatch, "MARKET_OPEN")
    assert "Today's Data" in html


def test_header_before_open(monkeypatch):
    html = _html(monkeypatch, "BEFORE_MARKET_OPEN")
    assert "Friday, June 12" in html
    assert "Today's Data" not in html


def test_header_after_close(monkeypatch):
    html = _html(monkeypatch, "AFTER_MARKET_CLOSE")
    assert "Friday, June 12" in html
    assert "Today's Data" not in html


def test_header_no_session_date_fallback(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "BEFORE_MARKET_OPEN")
    html = create_grid_display(**SAMPLE, session_date=None)
    assert "Last Session" in html


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
