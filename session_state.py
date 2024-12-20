import streamlit as st

def initialize_session_state():
    # List of initial stock tickers
    initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

    # Initialize session state variables
    if 'tickers' not in st.session_state:
        st.session_state.tickers = initial_tickers.copy()

    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []

    # Track the previous value of selected_tickers to detect changes
    if 'prev_selected_tickers' not in st.session_state:
        st.session_state.prev_selected_tickers = st.session_state.selected_tickers

    # Initialize new_ticker for search bar functionality if not already initialized
    if 'new_ticker' not in st.session_state:
        st.session_state.new_ticker = ''

    # Initialize session state for results
    if 'results_key' not in st.session_state:
        st.session_state.results_key = 0

    return st.session_state
