import streamlit as st
from utils import market_status, is_trading_day, get_nyse_date


def generate_market_status_header(status, last_date_str, today_is_trading_day=True):
    """Single status-driven header (icon + title, left-aligned).

    The title carries everything inline (no separate subtitle). `last_date_str`
    is the last completed session's date (data.index[-1]).

    Title by state:
      - BEFORE_MARKET_OPEN (trading day, pre-bell):
            "Market Closed - Displaying Predictions for <last traded date>"
      - MARKET_OPEN:
            "Live — Last Traded"
      - AFTER_MARKET_CLOSE on a trading day (today's close is final):
            "Today's Close Predictions and Actuals"
      - AFTER_MARKET_CLOSE on a weekend/holiday (today never traded):
            "Market Closed Weekend/Holiday - Displaying Predictions for
             <last traded date>"

    `today_is_trading_day` only matters for AFTER_MARKET_CLOSE — it splits the
    trading-day-after-close case from the weekend/holiday case, which
    market_status() collapses into one status.
    """
    subtitle = ""
    if status == "MARKET_OPEN":
        icon, title = "🔔", "Live — Last Traded"
    elif status == "BEFORE_MARKET_OPEN":
        icon, title = "🔴", "Market Closed"
        subtitle = f"Displaying Predictions for {last_date_str}"
    elif status == "AFTER_MARKET_CLOSE":
        icon = "🔴"
        if today_is_trading_day:
            title = "Today's Close Predictions and Actuals"
        else:
            title = "Market Closed Weekend/Holiday"
            subtitle = f"Displaying Predictions for {last_date_str}"
    else:
        raise ValueError(f"Unknown market status: {status}")

    # subtitle ~65% of title size (1.05rem vs 1.6rem), dark grey, snug under title
    subtitle_html = (
        f"<div style='font-size: 1.05rem; line-height: 1.1; color: #555555; margin-top: -10px;'>{subtitle}</div>"
        if subtitle else ""
    )
    return (
        "<div style='text-align: center; margin-bottom: 8px;'>"
        f"<h3 style='margin: 0; font-size: 1.6rem; line-height: 1.2;'>{icon} {title}</h3>"
        f"{subtitle_html}"
        "</div>"
    )


def generate_next_day_header(next_date_str):
    """Header for the forward next-day-close section.

    Always forward-looking, so it never uses the "Displaying" cue (that cue is
    reserved for the main header's past-session accuracy recaps). `next_date_str`
    is the concrete next trading session (utils.next_trading_day of the last
    completed session) — before the bell this is today.
    """
    return f"Predictions for {next_date_str}"


def display_market_status(last_available_date):
    """
    Display the single status-driven header.
    Args:
        last_available_date: date of the last completed session (data.index[-1]).
            Required — the caller only invokes this once it has a real session
            date, so there is no meaningful None default.
    """
    status = market_status()
    today_is_trading_day = is_trading_day(get_nyse_date())
    last_date_str = last_available_date.strftime('%A, %B %d')

    st.markdown(
        generate_market_status_header(status, last_date_str, today_is_trading_day),
        unsafe_allow_html=True,
    )
    st.markdown("---")  # Add a separator line
