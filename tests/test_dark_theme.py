"""Tests for §7 dark theme + light toggle.

Locks: THEME dict has both keys; dark/light tokens drive HTML in
generate_prediction_cards_html, create_grid_display, generate_chart_widget_html,
section_container_html, and ticker_header_html. Pure-function tests — no Streamlit.
"""
import pytest
from render_helpers import (
    THEME,
    generate_prediction_cards_html,
    create_grid_display,
    generate_chart_widget_html,
    section_container_html,
    section_container_css,
    _safe_key,
    ticker_header_html,
    _close_color,
)


# ── THEME dict structure ───────────────────────────────────────────────────────

def test_theme_has_dark_and_light():
    assert "dark" in THEME
    assert "light" in THEME


def test_theme_keys_complete():
    required = {
        "card_bg", "card_border", "text_price", "text_model_label",
        "text_delta_label", "metric_label", "metric_emphasis",
        "section_bg", "section_border", "ticker_color",
        "chart_bg", "chart_text", "chart_grid", "chart_border",
        "close_up", "close_down", "close_neutral",
        "delta_up", "delta_down", "delta_neutral",
    }
    for variant in ("dark", "light"):
        assert required.issubset(THEME[variant].keys()), f"Missing keys in THEME['{variant}']"


# ── dark prediction cards ─────────────────────────────────────────────────────

def test_dark_cards_use_dark_card_bg():
    html = generate_prediction_cards_html([105.0, 103.0], 100.0, "Δ from close", "dark")
    assert THEME["dark"]["card_bg"] in html


def test_light_cards_use_light_card_bg():
    html = generate_prediction_cards_html([105.0, 103.0], 100.0, "Δ from close", "light")
    assert THEME["light"]["card_bg"] in html


def test_dark_cards_price_color():
    html = generate_prediction_cards_html([105.0, 103.0], 100.0, "Δ from close", "dark")
    assert THEME["dark"]["text_price"] in html


def test_light_cards_price_color():
    html = generate_prediction_cards_html([105.0, 103.0], 100.0, "Δ from close", "light")
    assert THEME["light"]["text_price"] in html


def test_dark_delta_label_color():
    html = generate_prediction_cards_html([105.0, 103.0], 100.0, "Δ from close", "dark")
    assert THEME["dark"]["text_delta_label"] in html


# ── dark stats grid ───────────────────────────────────────────────────────────

def test_dark_grid_card_bg(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "AFTER_MARKET_CLOSE")
    html = create_grid_display(100.0, 105.0, 98.0, 99.0, 103.0, 1_000_000, theme_name="dark")
    assert THEME["dark"]["card_bg"] in html


def test_light_grid_card_bg(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "AFTER_MARKET_CLOSE")
    html = create_grid_display(100.0, 105.0, 98.0, 99.0, 103.0, 1_000_000, theme_name="light")
    assert THEME["light"]["card_bg"] in html


def test_dark_close_up_color(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "AFTER_MARKET_CLOSE")
    html = create_grid_display(100.0, 105.0, 98.0, 99.0, 103.0, 1_000_000, theme_name="dark")
    assert THEME["dark"]["close_up"] in html


def test_dark_close_down_color(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "AFTER_MARKET_CLOSE")
    html = create_grid_display(100.0, 105.0, 98.0, 103.0, 95.0, 1_000_000, theme_name="dark")
    assert THEME["dark"]["close_down"] in html


# ── DM5: responsive even-distribution grid ────────────────────────────────────

def test_grid_responsive_breakpoints(monkeypatch):
    monkeypatch.setattr("render_helpers.market_status", lambda: "AFTER_MARKET_CLOSE")
    html = create_grid_display(100.0, 105.0, 98.0, 99.0, 103.0, 1_000_000, theme_name="dark")
    assert "@container stats-grid (min-width:520px)" in html
    assert "@container stats-grid (min-width:960px)" in html
    assert "container-name:stats-grid" in html
    assert "container-type:inline-size" in html
    assert "repeat(2,1fr)" in html
    assert "repeat(3,1fr)" in html
    assert "repeat(6,1fr)" in html
    assert "auto-fit" not in html


# ── _close_color theme-aware ──────────────────────────────────────────────────

def test_close_color_up_dark():
    assert _close_color(105.0, 100.0, "dark") == THEME["dark"]["close_up"]


def test_close_color_down_dark():
    assert _close_color(95.0, 100.0, "dark") == THEME["dark"]["close_down"]


def test_close_color_equal_dark():
    assert _close_color(100.0, 100.0, "dark") == THEME["dark"]["close_neutral"]


def test_close_color_up_light():
    assert _close_color(105.0, 100.0, "light") == "#166534"


def test_close_color_down_light():
    assert _close_color(95.0, 100.0, "light") == "#991b1b"


# ── dark chart ────────────────────────────────────────────────────────────────

def test_dark_chart_bg():
    html = generate_chart_widget_html("AAPL", "[]", "dark")
    assert THEME["dark"]["chart_bg"] in html


def test_light_chart_bg():
    html = generate_chart_widget_html("AAPL", "[]", "light")
    assert THEME["light"]["chart_bg"] in html


def test_dark_chart_text_color():
    html = generate_chart_widget_html("AAPL", "[]", "dark")
    assert THEME["dark"]["chart_text"] in html


# ── section container + ticker header ────────────────────────────────────────
# Section containers always dark to match Streamlit chrome (set via config.toml).

def test_section_always_dark_bg():
    for theme in ("dark", "light"):
        html = section_container_html(theme)
        assert THEME["dark"]["section_bg"] in html


def test_ticker_header_always_dark_text():
    for theme in ("dark", "light"):
        html = ticker_header_html("TSLA", theme)
        assert THEME["dark"]["ticker_color"] in html
        assert "TSLA" in html


# ── DM3 native grouping container ────────────────────────────────────────────
# render_section_container mounts a native st.container styled via this scoped
# CSS (pure-string, testable). Replaces the broken lone-<div> wrapper.

def test_section_css_targets_keyed_container():
    css = section_container_css("ticker_section_AAPL", "dark")
    assert ".st-key-ticker_section_AAPL" in css


def test_section_css_dark_bg():
    css = section_container_css("ticker_section_AAPL", "dark")
    assert THEME["dark"]["section_bg"] in css
    assert THEME["dark"]["section_border"] in css


def test_section_css_light_bg():
    css = section_container_css("ticker_section_AAPL", "light")
    assert THEME["light"]["section_bg"] in css


def test_safe_key_sanitizes_special_chars():
    assert _safe_key("ticker_section_BRK-B") == "ticker_section_BRK_B"
    assert _safe_key("^GSPC") == "_GSPC"


def test_section_css_uses_safe_key():
    css = section_container_css("ticker_section_BRK-B", "dark")
    assert ".st-key-ticker_section_BRK_B" in css
