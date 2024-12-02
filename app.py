# app.py
# Import necessary modules
import streamlit as st
import streamlit as st
from gen_utils import get_todays_date, message_handler

# Function to initialize session state or other global setup
def initialize_session():
    initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
    if 'tickers' not in st.session_state:
        st.session_state.tickers = initial_tickers.copy()

    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []

# Utility function to handle ticker selection
def handle_ticker_selection():
    tickers = st.multiselect(
        "Select stocks to predict:",
        st.session_state.tickers,
        default=st.session_state.selected_tickers,
        key='stock_multiselect'
    )
    st.session_state.selected_tickers = tickers  # Update selected tickers

# Other utility functions like adding tickers or displaying messages
def handle_search_bar():
    new_ticker = st.text_input("Search for a stock ticker:", key="new_ticker_input")
    if new_ticker and new_ticker not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker)  # Add new ticker to list
        st.success(f"Added {new_ticker} to stock list.")

# Function to display today's date and other messages
def display_date_and_message():
    st.write(f"Today's date: {get_todays_date()}")
    message_handler("Welcome to the stock price prediction app!")

# Render the entire UI (UI layout)
def render_ui():
    # Display Title and UI components
    st.title("Stock Price Predictor")

    # Call functions to handle UI components
    '''
    display_date_and_message()  # Display the date and message
    handle_ticker_selection()  # Handle the multiselect of tickers
    handle_search_bar()  # Handle the search bar for new tickers
    '''
