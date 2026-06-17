import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
from utils import market_status
from trace_utils import trace_event


THEME = {
    "dark": {
        "card_bg": "linear-gradient(180deg, #192638, #0c1728)",
        "card_border": "rgba(148,163,184,0.18)",
        "card_shadow": "0 8px 24px rgba(0,0,0,0.25)",
        "text_price": "#f8fafc",
        "text_model_label": "#94a3b8",
        "text_delta_label": "#64748b",
        "metric_label": "#94a3b8",
        "metric_emphasis": "#cbd5e1",
        "section_bg": "#132030",
        "section_border": "rgba(148,163,184,0.12)",
        "section_shadow": "0 4px 16px rgba(0,0,0,0.2)",
        "ticker_color": "#f8fafc",
        "divider_grad": "linear-gradient(90deg, rgba(148,163,184,0.2), transparent)",
        "chart_bg": "#0f172a",
        "chart_text": "#cbd5e1",
        "chart_grid": "rgba(148,163,184,0.08)",
        "chart_border": "rgba(148,163,184,0.18)",
        "close_up": "#4ade80",
        "close_down": "#f87171",
        "close_neutral": "#94a3b8",
        "delta_up": "#4CAF50",
        "delta_down": "#FF5252",
        "delta_neutral": "#475569",
    },
    "light": {
        "card_bg": "linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247,250,252,0.9))",
        "card_border": "rgba(148,163,184,0.18)",
        "card_shadow": "0 8px 24px rgba(15,23,42,0.06)",
        "text_price": "#0f172a",
        "text_model_label": "#475569",
        "text_delta_label": "#94a3b8",
        "metric_label": "#475569",
        "metric_emphasis": "#334155",
        "section_bg": "#f1f5f9",
        "section_border": "rgba(148,163,184,0.18)",
        "section_shadow": "0 4px 16px rgba(15,23,42,0.04)",
        "ticker_color": "#0f172a",
        "divider_grad": "linear-gradient(90deg, rgba(148,163,184,0.3), transparent)",
        "chart_bg": "#ffffff",
        "chart_text": "#334155",
        "chart_grid": "rgba(148,163,184,0.2)",
        "chart_border": "rgba(148,163,184,0.3)",
        "close_up": "#166534",
        "close_down": "#991b1b",
        "close_neutral": "#475569",
        "delta_up": "#4CAF50",
        "delta_down": "#FF5252",
        "delta_neutral": "#475569",
    },
}


def _resolve_theme(theme_name):
    if theme_name is not None:
        return THEME.get(theme_name, THEME["light"])
    try:
        name = st.session_state.get("theme", "light")
    except Exception:
        name = "light"
    return THEME.get(name, THEME["light"])


@st.cache_data(ttl=300, show_spinner=False)
def get_recent_data(ticker):
    """Download recent stock data for the given ticker."""
    try:
        data = yf.download(ticker, period="10d", interval="1d", progress=False)
        if data.empty or len(data) < 2:
            raise ValueError(
                f"Insufficient data available for {ticker}. Need at least 2 days of data."
            )
        return data
    except Exception as e:
        print(f"Error getting data for {ticker}: {str(e)}")
        return None


def group_predictions_by_ticker(todays_close_predictions, next_day_close_predictions):
    grouped_predictions = {}
    grouped_next_day_predictions = {}

    for ticker, prediction in todays_close_predictions:
        if ticker not in grouped_predictions:
            grouped_predictions[ticker] = []
        grouped_predictions[ticker].append(prediction)

    for ticker, prediction in next_day_close_predictions:
        if ticker not in grouped_next_day_predictions:
            grouped_next_day_predictions[ticker] = []
        grouped_next_day_predictions[ticker].append(prediction)

    return grouped_predictions, grouped_next_day_predictions


def extract_and_format_ticker_data(latest_data):
    """Extract and format ticker data from the latest data."""
    if latest_data is None or len(latest_data) < 2:
        return None  # Or handle error as needed

    current_data = latest_data.iloc[-1]
    prev_data = latest_data.iloc[-2]

    # Ensure that the volume is correctly extracted as a scalar
    volume = current_data["Volume"].item()

    # Prepare formatted data as a dictionary
    formatted_data = {
        "Open": round(current_data["Open"].item(), 2),
        "High": round(current_data["High"].item(), 2),
        "Low": round(current_data["Low"].item(), 2),
        "Prev Close": round(prev_data["Close"].item(), 2),
        "Close": round(current_data["Close"].item(), 2),
        "Volume": int(volume),
    }
    return formatted_data


def _model_card_html(model_name, predicted_price, delta, delta_color, delta_sign, delta_label, t):
    return (
        f'<div style="flex: 1; min-width: 160px; padding: 16px; border-radius: 14px; '
        f'border: 1px solid {t["card_border"]}; '
        f'background: {t["card_bg"]}; '
        f'box-shadow: {t["card_shadow"]}; text-align: center;">'
        f'<div style="font-size: 0.76em; font-weight: 700; letter-spacing: 0.08em; '
        f'text-transform: uppercase; color: {t["text_model_label"]}; margin-bottom: 8px;">{model_name}</div>'
        f'<div style="font-size: 1.5em; font-weight: 800; color: {t["text_price"]}; line-height: 1.2; '
        f'margin-bottom: 4px;">${predicted_price:.2f}</div>'
        f'<div style="display: flex; align-items: center; justify-content: center; gap: 5px;">'
        f'<span style="font-size: 0.7em; font-weight: 600; color: {t["text_delta_label"]}; white-space: nowrap;">{delta_label}</span>'
        f'<span style="font-size: 1.05em; font-weight: 700; color: {delta_color};">({delta_sign}${abs(delta):.2f})</span>'
        f'</div>'
        f'</div>'
    )


def _prediction_cards_container_html(cards_html):
    return (
        '<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px;">'
        f'{cards_html}'
        '</div>'
    )


def generate_prediction_cards_html(predictions, current_val, delta_label, theme_name=None):
    """Build HTML for two side-by-side model prediction cards."""
    t = _resolve_theme(theme_name)
    cards_html = ""
    for i, prediction in enumerate(predictions):
        model_name = "Linear Regression" if i == 0 else "XGBoost"
        delta = prediction - current_val
        if delta > 0:
            delta_color = t["delta_up"]
            delta_sign = "+"
        elif delta < 0:
            delta_color = t["delta_down"]
            delta_sign = "-"
        else:
            delta_color = t["delta_neutral"]
            delta_sign = ""
        cards_html += _model_card_html(model_name, prediction, delta, delta_color, delta_sign, delta_label, t)
    return _prediction_cards_container_html(cards_html)


def display_predictions(predictions, current_val, delta_label, theme_name=None):
    """Display predictions as two side-by-side model cards."""
    st.markdown(
        generate_prediction_cards_html(predictions, current_val, delta_label, theme_name),
        unsafe_allow_html=True,
    )


def display_tradingview_chart_from_data(ticker, latest_data, theme_name=None):
    """Render a TradingView Lightweight Charts candlestick chart from local OHLC data."""
    if latest_data is None or latest_data.empty:
        st.warning(f"No chart data available for {ticker}.")
        return

    chart_data = latest_data.tail(10).copy()
    if len(chart_data) < 2:
        st.warning(f"Not enough recent data to chart {ticker}.")
        return

    rows = []
    for idx, row in chart_data.iterrows():
        rows.append(
            {
                "time": int(idx.timestamp()),
                "open": float(row["Open"].item()),
                "high": float(row["High"].item()),
                "low": float(row["Low"].item()),
                "close": float(row["Close"].item()),
            }
        )

    chart_json = str(rows).replace("'", '"')
    components.html(generate_chart_widget_html(ticker, chart_json, theme_name), height=500, scrolling=False)


def generate_chart_widget_html(ticker, chart_json, theme_name=None):
    """Build TradingView chart HTML. Pure function for testability."""
    t = _resolve_theme(theme_name)
    return f"""
    <div style="padding: 16px; background: {t["card_bg"]}; border: 1px solid {t["card_border"]}; border-radius: 14px; box-shadow: {t["card_shadow"]};">
      <div style="padding-bottom: 12px; display: flex; justify-content: space-between; align-items: center; gap: 12px;">
        <div style="font-weight: 800; font-size: 1.1em; color: {t["text_price"]};">{ticker.upper()} · Last 10 Trading Days</div>
      </div>
      <div id="tv_chart_{ticker}" style="height: 420px; width: 100%; background: {t["chart_bg"]}; border-radius: 8px;"></div>
    </div>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script>
      (function() {{
        const container = document.getElementById("tv_chart_{ticker}");
        if (!container || !window.LightweightCharts) {{
          if (container) {{
            container.innerHTML = '<div style="padding:16px;color:#ef4444;">Chart library failed to load.</div>';
          }}
          return;
        }}
        const chart = LightweightCharts.createChart(container, {{
          layout: {{
            background: {{ color: '{t["chart_bg"]}' }},
            textColor: '{t["chart_text"]}',
          }},
          grid: {{
            vertLines: {{ color: '{t["chart_grid"]}' }},
            horzLines: {{ color: '{t["chart_grid"]}' }},
          }},
          rightPriceScale: {{
            borderColor: '{t["chart_border"]}',
          }},
          timeScale: {{
            borderColor: '{t["chart_border"]}',
            timeVisible: false,
            secondsVisible: false,
          }},
          width: container.clientWidth,
          height: 420,
        }});

        const candleSeries = chart.addSeries
          ? chart.addSeries(LightweightCharts.CandlestickSeries, {{
              upColor: '#22c55e',
              downColor: '#f87171',
              borderUpColor: '#22c55e',
              borderDownColor: '#f87171',
              wickUpColor: '#22c55e',
              wickDownColor: '#f87171',
            }})
          : chart.addCandlestickSeries({{
          upColor: '#22c55e',
          downColor: '#f87171',
          borderUpColor: '#22c55e',
          borderDownColor: '#f87171',
          wickUpColor: '#22c55e',
          wickDownColor: '#f87171',
        }});

        const data = {chart_json};
        candleSeries.setData(data);
        chart.timeScale().fitContent();

        window.addEventListener('resize', () => {{
          chart.applyOptions({{ width: container.clientWidth }});
        }});
      }})();
    </script>
    """


def _close_color(close_val, prev_close_val, theme_name=None):
    t = _resolve_theme(theme_name)
    if close_val > prev_close_val:
        return t["close_up"]
    if close_val < prev_close_val:
        return t["close_down"]
    return t["close_neutral"]


def create_grid_display(open_val, high_val, low_val, prev_close_val, close_val, volume, session_date=None, theme_name=None):
    t = _resolve_theme(theme_name)

    def format_price(value):
        return f"${value:,.2f}"

    close_val_color = _close_color(close_val, prev_close_val, theme_name)

    metrics = [
        "Open",
        "High",
        "Low",
        "Prev Close",
        "Last Traded" if market_status() == "MARKET_OPEN" else "Close",
        "Volume",
    ]
    values = [
        open_val,
        high_val,
        low_val,
        prev_close_val,
        close_val,
        f"{int(volume):,}",
    ]

    grid_items = []
    for metric, value in zip(metrics, values):
        is_emphasis = metric in ("Prev Close", "Last Traded", "Close")
        card_bg = t["card_bg"]
        card_border = t["card_border"]
        value_color = close_val_color if metric in ("Last Traded", "Close") else t["text_price"]
        metric_color = t["metric_label"] if not is_emphasis else t["metric_emphasis"]
        grid_items.append(
            f'<div style="position: relative; overflow: hidden; padding: 14px 14px 13px; border-radius: 16px; border: 1px solid {card_border}; background: {card_bg}; box-shadow: {t["card_shadow"]}; text-align: left;">'
            f'<div style="font-size: 0.76em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {metric_color}; margin-bottom: 8px;">{metric}</div>'
            f'<div style="font-size: 1.08em; font-weight: 800; color: {value_color}; line-height: 1.1;">{format_price(value) if metric != "Volume" else value}</div>'
            f'</div>'
        )

    return (
        '<style>'
        '.stats-grid-6{display:grid;gap:12px;margin:8px 0 12px 0;grid-template-columns:repeat(2,1fr)}'
        '@container stats-grid (min-width:520px){.stats-grid-6{margin:8px 0 16px 0;grid-template-columns:repeat(3,1fr)}}'
        '@container stats-grid (min-width:960px){.stats-grid-6{margin:8px 0 20px 0;grid-template-columns:repeat(6,1fr)}}'
        '</style>'
        '<div style="container-type:inline-size;container-name:stats-grid">'
        '<div class="stats-grid-6">'
        + ''.join(grid_items)
        + '</div></div>'
    )


def section_container_html(theme_name=None):
    """Return open-tag HTML for a ticker section container. Always dark to match Streamlit chrome.

    DEPRECATED (DM3): a lone opening <div> from st.markdown can't wrap sibling
    widgets or the chart iframe — Streamlit sanitizes each block independently
    and force-closes it. Use render_section_container() instead. Retained until
    DM6 migrates the next-day section.
    """
    dark = THEME["dark"]
    return (
        f'<div style="border: 1px solid {dark["section_border"]}; border-radius: 20px; '
        f'padding: 18px 18px 16px; margin: 20px 0 14px 0; '
        f'background: {dark["section_bg"]}; box-shadow: {dark["section_shadow"]};">'
    )


def _safe_key(key):
    """CSS-class-safe key: tickers like BRK-B / ^GSPC -> underscores."""
    return "".join(c if c.isalnum() else "_" for c in str(key))


def section_container_css(key, theme_name=None, padding_bottom=16, margin_bottom=0):
    """Scoped <style> styling the native st.container(key=...) as a ticker
    grouping card (DM3). Pure string so it's unit-testable; the actual mount
    happens in render_section_container."""
    t = _resolve_theme(theme_name)
    safe = _safe_key(key)
    margin_rule = f"margin-bottom: {margin_bottom}px; " if margin_bottom else ""
    return (
        f"<style>.st-key-{safe} {{ "
        f"background: {t['section_bg']}; "
        f"border: 1px solid {t['section_border']}; "
        f"border-radius: 20px; "
        f"padding: 18px 18px {padding_bottom}px; "
        f"{margin_rule}"
        f"box-shadow: {t['section_shadow']}; }}</style>"
    )


def render_section_container(key, theme_name=None, padding_bottom=16, margin_bottom=0):
    """Native bordered grouping card for a ticker section (DM3).

    Returns a native st.container — use it as a `with` block so the header,
    prediction cards, chart iframe, and stats grid all nest inside one visually
    grouped surface. Reusable: DM6 applies it to the next-day section too.
    """
    safe = _safe_key(key)
    st.markdown(section_container_css(safe, theme_name, padding_bottom, margin_bottom), unsafe_allow_html=True)
    return st.container(key=safe)


def ticker_header_html(ticker, theme_name=None):
    """Return HTML for the ticker name + accent bar row. Always light text on dark section bg."""
    dark = THEME["dark"]
    return (
        f'<div style="display: flex; align-items: center; gap: 10px;">'
        f'<div style="width: 4px; height: 22px; background: #3b82f6; border-radius: 3px;"></div>'
        f'<span style="font-size: 1.15em; font-weight: 800; color: {dark["ticker_color"]};">{ticker}</span>'
        f'<div style="flex: 1; height: 1px; background: {dark["divider_grad"]};"></div>'
        f'</div>'
    )


def search_and_add_ticker(new_ticker):
    # Returns True  → ticker added/selected: caller should clear bar and rerun
    # Returns None  → guard fired (already processed): caller should clear bar, no forced rerun
    # Returns False → invalid/empty: keep bar so user sees warning
    if new_ticker != st.session_state.new_ticker:
        st.session_state.new_ticker = new_ticker
        trace_event("search.state_updated", new_ticker=new_ticker)

    if new_ticker:
        try:
            new_ticker_upper = new_ticker.strip().upper()
            if not new_ticker_upper:
                trace_event("search.invalid_or_empty", new_ticker=new_ticker)
                return False

            already_selected = new_ticker_upper in [
                t.upper() for t in st.session_state.selected_tickers
            ]
            if already_selected:
                trace_event("search.already_selected_early", ticker=new_ticker_upper)
                st.info(f"{new_ticker_upper} is already selected.")
                st.session_state.last_processed_ticker_search = new_ticker_upper
                return True

            trace_event(
                "search.enter",
                new_ticker=new_ticker,
                selected=st.session_state.get("selected_tickers", []),
                tickers=st.session_state.get("tickers", []),
            )

            with st.spinner(f"Validating {new_ticker_upper}..."):
                stock_data = get_recent_data(new_ticker_upper)
            if stock_data is None or stock_data.empty:
                trace_event("search.invalid_or_empty", new_ticker=new_ticker_upper)
                st.session_state.last_processed_ticker_search = new_ticker_upper
                st.warning(f"Ticker '{new_ticker_upper}' is not valid or does not exist.")
                return False

            st.session_state.last_processed_ticker_search = new_ticker_upper
            if new_ticker_upper not in [
                t.upper() for t in st.session_state.tickers
            ]:
                trace_event("search.add_new_ticker", ticker=new_ticker_upper)
                st.session_state.tickers.insert(0, new_ticker_upper)
            else:
                trace_event("search.ticker_option_exists", ticker=new_ticker_upper)

            if not already_selected:
                trace_event("search.select_ticker", ticker=new_ticker_upper)
                st.session_state.selected_tickers.append(new_ticker_upper)
                st.session_state.pending_autoselect_ticker = new_ticker_upper
                trace_event(
                    "search.state_mutated",
                    reason="selected_ticker",
                    selected=st.session_state.selected_tickers,
                    tickers=st.session_state.tickers,
                )
            else:
                trace_event("search.already_selected", ticker=new_ticker_upper)
            return True
        except Exception as e:
            trace_event("search.exception", new_ticker=new_ticker, error=str(e))
            st.error(f"Error: {e}")
            return False
    return False
