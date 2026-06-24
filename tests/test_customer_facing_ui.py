from pathlib import Path

import render_ui
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


def test_provider_down_skipped_ticker_renders_customer_message(monkeypatch):
    messages = []

    class DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(render_ui, "enforce_max_tickers", lambda: None)
    monkeypatch.setattr(render_ui, "market_status", lambda: "MARKET_CLOSED")
    monkeypatch.setattr(
        render_ui,
        "render_section_container",
        lambda *args, **kwargs: DummyContainer(),
    )
    monkeypatch.setattr(render_ui, "ticker_header_html", lambda *args, **kwargs: "")
    monkeypatch.setattr(render_ui.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(render_ui.st, "error", messages.append)

    render_ui.display_results(
        [[], []],
        [("IBM", render_ui.YFINANCE_PROVIDER_DOWN_MESSAGE)],
    )

    assert messages == [render_ui.YFINANCE_PROVIDER_DOWN_MESSAGE]
