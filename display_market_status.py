import streamlit as st
from utils import market_status

# Inline status icon + title. Title is driven by market_status() so the single
# header replaces the old centered date <h2> + the separate "Today's Close
# Predictions" subheader (plan §1). User only ever distinguishes Open vs Closed.
_STATUS = {
    "BEFORE_MARKET_OPEN": {"icon": "⏳", "title": "Predictions for {date}"},
    "MARKET_OPEN":        {"icon": "🔔", "title": "Live — Last Traded"},
    "AFTER_MARKET_CLOSE": {"icon": "🔴", "title": "Today's Close Predictions"},
}


def generate_market_status_header(status, last_date_str):
    """Single status-driven header (icon + title, left-aligned) with an optional
    session-date subtitle.

    Subtitle binary on market status:
      - MARKET_OPEN          → no subtitle (data is today's, self-explanatory).
      - AFTER_MARKET_CLOSE   → "Last traded session — <date>".
      - BEFORE_MARKET_OPEN   → no subtitle; the title already names the session
                               date, so a date subtitle would just duplicate it.
    """
    if status not in _STATUS:
        raise ValueError(f"Unknown market status: {status}")

    icon = _STATUS[status]["icon"]
    title = _STATUS[status]["title"].format(date=last_date_str)

    subtitle_html = ""
    if status == "AFTER_MARKET_CLOSE":
        subtitle_html = (
            "<div style='font-size: 10pt; color: #64748b; margin-top: 2px;'>"
            f"Last traded session — {last_date_str}</div>"
        )

    return f"""<div style='text-align: left; margin-bottom: 8px;'>
        <h3 style='margin: 0;'>{icon} {title}</h3>
        {subtitle_html}
    </div>"""


def display_market_status(last_available_date=None):
    """
    Display the single status-driven header.
    Args:
        last_available_date: date of the last completed session (data.index[-1]).
    """
    status = market_status()
    last_date_str = last_available_date.strftime('%A, %B %d')

    st.markdown(
        generate_market_status_header(status, last_date_str),
        unsafe_allow_html=True,
    )
    st.markdown("---")  # Add a separator line
