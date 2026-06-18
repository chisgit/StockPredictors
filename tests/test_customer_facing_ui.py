from pathlib import Path

from render_ui import app_scrollbar_css


def test_cache_debug_controls_are_not_customer_facing():
    source = Path("render_ui.py").read_text(encoding="utf-8")

    assert "Cache Debug" not in source
    assert "Show yfinance cache path" not in source
    assert "Clear yfinance cache now" not in source


def test_app_scrollbar_css_uses_dark_theme_styling():
    css = app_scrollbar_css()

    assert "scrollbar-width: thin" in css
    assert "scrollbar-color: #475569 #0f172a" in css
    assert "::-webkit-scrollbar-thumb" in css
    assert "border-radius: 999px" in css
