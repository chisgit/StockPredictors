import streamlit as st
from utils import get_nyse_datetime, market_status

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
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 0;'>{date_str}</h2>", unsafe_allow_html=True)

    last_date_str = last_available_date.strftime('%A, %B %d')
    time_str = current_time.strftime('%I:%M %p EST')  # Add EST to the time

    if status == "BEFORE_MARKET_OPEN":
        st.markdown(f"""<div style='text-align: center; margin-top: -10px;'>
            <h3 style='margin-bottom: 0;'>⏳ Market hasn't opened yet</h3>
            <div style='font-size: 10pt; margin-top: -10px;'>Predicted closing prices are for {last_date_str} based on the latest available data</div>
        </div>""", unsafe_allow_html=True)
    elif status == "MARKET_OPEN":
        st.markdown(f"""<div style='text-align: center; margin-top: -10px;'>
            <h3 style='margin-bottom: 0;'>🔔 Market is Open</h3>
            <div style='font-size: 10pt; margin-top: -10px;'>Predicted closing price for {last_date_str} based on current time: {time_str}</div>
        </div>""", unsafe_allow_html=True)
    else:  # AFTER_MARKET_CLOSE
        st.markdown(f"""<div style='text-align: center; margin-top: -10px;'>
            <h3 style='margin-bottom: 0;'>🔴 Market is Closed</h3>
            <div style='font-size: 10pt; margin-top: -10px;'>Predicted closing price for {last_date_str}. Today's closing prices are final.</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")  # Add a separator line