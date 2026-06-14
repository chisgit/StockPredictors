"""Isolated tests for §1f/§1f-δ — prediction card delta label.

Locks: _delta_caption(is_open) emits the correct inline label shown
left of the price diff in each model card.

State contract:
  MARKET_OPEN  → "Δ from last traded"
  closed       → "Δ from close"
"""
from render_ui import _delta_caption


def test_closed_market_label():
    assert _delta_caption(is_open=False) == "Δ from close"


def test_open_market_label():
    assert _delta_caption(is_open=True) == "Δ from last traded"


def test_closed_contains_delta_symbol():
    assert "Δ" in _delta_caption(is_open=False)


def test_open_contains_delta_symbol():
    assert "Δ" in _delta_caption(is_open=True)


def test_closed_contains_from():
    assert "from" in _delta_caption(is_open=False)


def test_open_contains_from():
    assert "from" in _delta_caption(is_open=True)


def test_labels_are_distinct():
    assert _delta_caption(is_open=False) != _delta_caption(is_open=True)


def test_no_price_value_in_label():
    for label in (_delta_caption(True), _delta_caption(False)):
        assert "$" not in label


def test_no_date_in_label():
    for label in (_delta_caption(True), _delta_caption(False)):
        assert "Jun" not in label
        assert "2026" not in label
