from pathlib import Path

from render_ui import app_scrollbar_css


def test_cache_debug_controls_are_not_customer_facing():
    source = Path("render_ui.py").read_text(encoding="utf-8")

    assert "Cache Debug" not in source
    assert "Show yfinance cache path" not in source
    assert "Clear yfinance cache now" not in source


def test_app_scrollbar_css_is_well_formed_dark_styling():
    css = app_scrollbar_css()

    # Structural, not value-coupled: a palette/token tweak shouldn't break this.
    assert "<style>" in css and "</style>" in css
    assert "scrollbar-width: thin" in css
    assert "::-webkit-scrollbar-thumb" in css
