"""Isolated tests for the two-model-card prediction display (plan section 2).

Smoke-tests the HTML output of generate_prediction_cards_html(). Locks:
two model cards (Linear Regression, XGBoost) present, predicted prices
formatted, delta coloring per up/down/equal. Pure function — no Streamlit
dependency.
"""
import pytest
from render_helpers import generate_prediction_cards_html


def test_both_model_cards_present():
    html = generate_prediction_cards_html([105.0, 103.5], 100.0)
    assert "Linear Regression" in html
    assert "XGBoost" in html


def test_prices_formatted():
    html = generate_prediction_cards_html([105.0, 103.5], 100.0)
    assert "$105.00" in html
    assert "$103.50" in html


def test_delta_up_green():
    html = generate_prediction_cards_html([105.0, 100.0], 100.0)
    assert "#4CAF50" in html
    assert "(+$5.00)" in html


def test_delta_down_red():
    html = generate_prediction_cards_html([95.0, 98.0], 100.0)
    assert "#FF5252" in html
    assert "(-$5.00)" in html
    assert "(-$2.00)" in html


def test_delta_equal():
    html = generate_prediction_cards_html([100.0, 100.0], 100.0)
    assert "($0.00)" in html


def test_card_has_border_radius():
    html = generate_prediction_cards_html([100.0, 100.0], 100.0)
    assert "border-radius: 14px" in html


def test_model_order_lr_then_xgb():
    html = generate_prediction_cards_html([100.0, 100.0], 100.0)
    assert html.index("Linear Regression") < html.index("XGBoost")


def test_card_min_width_set():
    html = generate_prediction_cards_html([100.0, 100.0], 100.0)
    assert "min-width: 160px" in html


def test_mixed_deltas():
    html = generate_prediction_cards_html([110.0, 90.0], 100.0)
    assert "(+$10.00)" in html
    assert "(-$10.00)" in html
    assert "(+$10.00)" in html
