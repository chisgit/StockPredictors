import streamlit as st
from utils import get_nyse_datetime, market_status

def generate_market_status_header(date_str):
    """Generate the header with the current date."""
    return f"<h2 style='text-align: center; margin-bottom: 0;'>{date_str}</h2>"

def generate_market_status_message(status, last_date_str, time_str):
    """Generate the market status message based on the current market status."""
    status_messages = {
        "BEFORE_MARKET_OPEN": {
            "icon": "⏳",
            "title": "Market hasn't opened yet",
            "subtitle": f"Predicted closing prices are for {last_date_str} based on the latest available data"
        },
        "MARKET_OPEN": {
            "icon": "🔔",
            "title": "Market is Open",
            "subtitle": f"Predicted closing price for {last_date_str} based on current time: {time_str}"
        },
        "AFTER_MARKET_CLOSE": {
            "icon": "🔴",
            "title": "Market is Closed",
            "subtitle": f"Predicted closing price for {last_date_str}. Today's closing prices are final."
        }
    }
    
    # Raise an error if the status is not found, which helps catch potential issues
    if status not in status_messages:
        raise ValueError(f"Unknown market status: {status}")
    
    message = status_messages[status]
    
    return f"""<div style='text-align: center; margin-top: -10px;'>
        <h3 style='margin-bottom: 0;'>{message['icon']} {message['title']}</h3>
        <div style='font-size: 10pt; margin-top: -10px;'>{message['subtitle']}</div>
    </div>"""

def display_market_status(last_available_date=None):
    """
    Display market status with icon and description.
    Args:
        last_available_date: Last available trading date (for BEFORE_MARKET_OPEN state)
    """
    # Get current time in Eastern Time
    current_time = get_nyse_datetime()
    status = market_status()
    
    # Format and display current date
    date_str = current_time.strftime('%A, %B %d, %Y')
    st.markdown(generate_market_status_header(date_str), unsafe_allow_html=True)

    last_date_str = last_available_date.strftime('%A, %B %d')
    time_str = current_time.strftime('%I:%M %p EST')  # Add EST to the time

    st.markdown(generate_market_status_message(status, last_date_str, time_str), unsafe_allow_html=True)
    
    st.markdown("---")  # Add a separator line