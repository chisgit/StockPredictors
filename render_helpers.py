import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
from utils import market_status


def get_recent_data(ticker):
    """Download recent stock data for the given ticker."""
    try:
        data = yf.download(ticker, period="10d", interval="1d")
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


def preds_sameline(predictions, current_val):
    predictions_html = ""
    for i, prediction in enumerate(predictions):
        model_type = "Linear Regression" if i == 0 else "XGBoost"
        price_diff = prediction - current_val
        diff_color = "#4CAF50" if price_diff >= 0 else "#FF5252"

        if price_diff > 0:
            diff_sign = "+"
        elif price_diff < 0:
            diff_sign = "-"
        else:
            diff_sign = ""

        diff_str = f'<span style="color: {diff_color}; margin-left: 8px;">({diff_sign}${abs(price_diff):.2f})</span>'
        predictions_html += f'{model_type}: <span style="font-size: 1.1em;">${prediction:.2f}</span>{diff_str} &nbsp;&nbsp;'
    return predictions_html


def display_predictions(ticker, predictions_html):
    """Display predictions for a given ticker."""
    st.markdown(
        f'<div style="margin: 14px 0 8px 0; padding: 12px 14px; border: 1px solid rgba(148,163,184,0.18); border-radius: 14px; background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(247,250,252,0.96)); box-shadow: 0 8px 24px rgba(15,23,42,0.06);">'
        f'<div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;">'
        f'<span style="font-size: 1.05em; font-weight: 800; letter-spacing: 0.04em; color: #0f172a;">{ticker}</span>'
        f'<span style="line-height: 1.5;">{predictions_html}</span>'
        f'</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def display_tradingview_chart_from_data(ticker, latest_data):
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
    widget_html = f"""
    <div style="margin: 8px 0 18px 0; border: 1px solid rgba(51,65,85,0.8); border-radius: 18px; overflow: hidden; box-shadow: 0 16px 36px rgba(2,6,23,0.35); background: linear-gradient(180deg, #0f172a 0%, #111827 100%);">
      <div style="padding: 14px 16px; border-bottom: 1px solid rgba(148,163,184,0.14); display: flex; justify-content: space-between; align-items: center; gap: 12px; background: linear-gradient(90deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));">
        <div>
          <div style="font-size: 0.72em; letter-spacing: 0.16em; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">TradingView Style Chart</div>
          <div style="font-weight: 800; font-size: 1.1em; color: #f8fafc;">{ticker.upper()} · Last 10 Trading Days</div>
        </div>
        <div style="font-size: 0.85em; color: #cbd5e1;">OHLC candles from local market data</div>
      </div>
      <div id="tv_chart_{ticker}" style="height: 420px; width: 100%;"></div>
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
            background: {{ color: '#0f172a' }},
            textColor: '#cbd5e1',
          }},
          grid: {{
            vertLines: {{ color: 'rgba(148,163,184,0.08)' }},
            horzLines: {{ color: 'rgba(148,163,184,0.08)' }},
          }},
          rightPriceScale: {{
            borderColor: 'rgba(148,163,184,0.18)',
          }},
          timeScale: {{
            borderColor: 'rgba(148,163,184,0.18)',
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
    components.html(widget_html, height=500, scrolling=False)


def create_grid_display(open_val, high_val, low_val, prev_close_val, close_val, volume, session_date=None):
    def format_price(value):
        return f"${value:,.2f}"

    status = market_status()

    close_is_up = close_val >= prev_close_val
    close_bg = "rgba(34,197,94,0.12)" if close_is_up else "rgba(239,68,68,0.12)"
    close_border = "rgba(34,197,94,0.28)" if close_is_up else "rgba(239,68,68,0.28)"
    close_color = "#166534" if close_is_up else "#991b1b"

    metrics = [
        "Open",
        "High",
        "Low",
        "Prev Close",
        "Last Traded" if status == "MARKET_OPEN" else "Close",
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
        card_bg = close_bg if metric == "Last Traded" or metric == "Close" else "linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.9))"
        card_border = close_border if metric == "Last Traded" or metric == "Close" else "rgba(148,163,184,0.18)"
        value_color = close_color if metric == "Last Traded" or metric == "Close" else "#0f172a"
        metric_color = "#475569" if not is_emphasis else "#334155"
        grid_items.append(
            f'<div style="position: relative; overflow: hidden; padding: 14px 14px 13px; border-radius: 16px; border: 1px solid {card_border}; background: {card_bg}; box-shadow: 0 10px 24px rgba(15,23,42,0.05); text-align: left;">'
            f'<div style="font-size: 0.76em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {metric_color}; margin-bottom: 8px;">{metric}</div>'
            f'<div style="font-size: 1.08em; font-weight: 800; color: {value_color}; line-height: 1.1;">{format_price(value) if metric != "Volume" else value}</div>'
            f'</div>'
        )

    if status == "MARKET_OPEN":
        panel_header = "Today's Data"
    else:
        panel_header = session_date.strftime('%A, %B %d') if session_date else "Last Session"

    grid = (
        '<div style="display: grid; '
        'grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 12px;">'
        + ''.join(grid_items)
        + '</div>'
    )
    return (
        '<div style="border: 1px solid rgba(148,163,184,0.25); border-radius: 18px; '
        'padding: 16px 16px 14px; margin: 8px 0 2px 0;">'
        f'<div style="font-size: 0.78em; font-weight: 700; letter-spacing: 0.06em; '
        f'text-transform: uppercase; color: #475569; margin-bottom: 10px;">{panel_header}</div>'
        + grid
        + '</div>'
    )


def search_and_add_ticker(new_ticker):
    # Clears out the ticker if there's a change in removing the ticker
    # This way it doesn't keep appearing in the search bar after it's been removed
    if new_ticker != st.session_state.new_ticker:
        st.session_state.new_ticker = new_ticker

    if new_ticker:
        try:
            stock_data = get_recent_data(new_ticker)
            if stock_data is None or stock_data.empty:
                st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
            else:
                new_ticker_upper = new_ticker.upper()
                if new_ticker_upper not in [
                    t.upper() for t in st.session_state.tickers
                ]:
                    st.session_state.tickers.insert(0, new_ticker_upper)
                    st.session_state.selected_tickers.append(new_ticker_upper)
                    st.rerun()
                else:
                    if new_ticker_upper not in [
                        t.upper() for t in st.session_state.selected_tickers
                    ]:
                        st.session_state.selected_tickers.append(new_ticker_upper)
                        st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
