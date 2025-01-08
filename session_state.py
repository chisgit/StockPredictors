import streamlit as st

def initialize_session_state():
    """Initialize all session state variables."""
    
    # List of initial stock tickers
    initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

    # Initialize session state variables if they don't exist
    if 'tickers' not in st.session_state:
        st.session_state.tickers = initial_tickers.copy()

    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []

    if 'new_ticker' not in st.session_state:
        st.session_state.new_ticker = ''

    if 'run_prediction' not in st.session_state:
        st.session_state.run_prediction = False

    if 'results_key' not in st.session_state:
        st.session_state.results_key = 0
        
    return st.session_state

def is_ticker_selected(ticker):
    """Check if a ticker is already selected."""
    return ticker.upper() in [t.upper() for t in st.session_state.selected_tickers]

def add_ticker(ticker):
    """Add a ticker if it's not already in the list."""
    ticker = ticker.upper()
    if ticker not in [t.upper() for t in st.session_state.tickers]:
        st.session_state.tickers.insert(0, ticker)
    if not is_ticker_selected(ticker):
        st.session_state.selected_tickers.append(ticker)
        return True
    return False
