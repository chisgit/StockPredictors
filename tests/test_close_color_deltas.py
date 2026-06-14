"""Tests for §4 close-card number color + §5 bold deltas.

§4: Close/Last-Traded card has neutral bg/border; value text colored
up/down/equal.  §5: deltas in prediction cards are bold + equal case neutral.
"""
from render_helpers import _close_color, create_grid_display, generate_prediction_cards_html


# ── §4: _close_color helper ────────────────────────────────────────────────

def test_close_color_up():
    assert _close_color(105.0, 100.0) == "#166534"


def test_close_color_down():
    assert _close_color(95.0, 100.0) == "#991b1b"


def test_close_color_equal():
    assert _close_color(100.0, 100.0) == "#475569"


# ── §4: neutral card background for Close/Last-Traded ───────────────────────

def test_close_card_bg_neutral(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "MARKET_OPEN")
    html = create_grid_display(
        open_val=100.0, high_val=105.0, low_val=98.0,
        prev_close_val=99.0, close_val=103.0, volume=1_000_000,
    )
    assert "LAST TRADED" in html.upper()
    assert "rgba(34,197,94,0.12)" not in html
    assert "rgba(239,68,68,0.12)" not in html


def test_close_card_down_value_red(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "AFTER_MARKET_CLOSE")
    html = create_grid_display(
        open_val=100.0, high_val=105.0, low_val=98.0,
        prev_close_val=103.0, close_val=95.0, volume=1_000_000,
    )
    assert "#991b1b" in html


def test_close_card_eq_value_neutral(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "AFTER_MARKET_CLOSE")
    html = create_grid_display(
        open_val=100.0, high_val=100.0, low_val=100.0,
        prev_close_val=100.0, close_val=100.0, volume=1_000_000,
    )
    assert "#475569" in html


# ── §5: delta bold styling ─────────────────────────────────────────────────

def test_delta_has_bold_weight():
    html = generate_prediction_cards_html([105.0, 95.0], 100.0, "Δ from close")
    assert 'font-weight: 700' in html


def test_delta_equal_neutral_color():
    html = generate_prediction_cards_html([100.0, 100.0], 100.0, "Δ from close")
    assert "#475569" in html
