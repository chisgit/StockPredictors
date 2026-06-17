from pathlib import Path


def test_cache_debug_controls_are_not_customer_facing():
    source = Path("render_ui.py").read_text(encoding="utf-8")

    assert "Cache Debug" not in source
    assert "Show yfinance cache path" not in source
    assert "Clear yfinance cache now" not in source
