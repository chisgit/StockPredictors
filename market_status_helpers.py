import streamlit as st
from utils import get_nyse_datetime, market_status
from ui_helpers import generate_market_status_header, generate_market_status_message


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