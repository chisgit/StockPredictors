import streamlit as st
import time as time_module
import yfinance as yf
from app import display_results, update_selected_tickers
from pipeline import execute_pipeline

def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    
    # Get the current multiselect state
    updated_sel_tickers = st.session_state.stock_multiselect
    
    # Update selected_tickers to match the multiselect state
    st.session_state.selected_tickers = updated_sel_tickers
    
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")

# Title of the app
st.title("Stock Price Predictor")

# List of initial stock tickers
initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Initialize session state variables
if 'tickers' not in st.session_state:
    st.session_state.tickers = initial_tickers.copy()  # Example: ['TSLA', 'NVDA', 'AAPL']

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

tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,
    default=st.session_state.selected_tickers,  # Preserve selected tickers
    key='stock_multiselect',
    on_change=update_selected_tickers,
    args=['stock_multiselect']
)

# Update selected tickers based on multiselect
st.session_state.selected_tickers = tickers

# Search bar for new tickers
new_ticker = st.text_input("Search for a stock ticker:", value=st.session_state.new_ticker, key="new_ticker_input")

# This clears out the ticker if there's a change in removing the ticker
# this way it doesn't keep appearing in the search bar after it's been removed
if new_ticker != st.session_state.new_ticker:
    st.session_state.new_ticker = new_ticker

if new_ticker:
    try:
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            new_ticker_upper = new_ticker.upper()
            if new_ticker_upper not in [t.upper() for t in st.session_state.tickers]:
                st.session_state.tickers.insert(0, new_ticker_upper)
                st.session_state.selected_tickers.append(new_ticker_upper)
                st.rerun()
            else:
                if new_ticker_upper not in [t.upper() for t in st.session_state.selected_tickers]:
                    st.session_state.selected_tickers.append(new_ticker_upper)
                    st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")
st.session_state.new_ticker = ''
    

# Predict button and results area
predict_button = st.button("Predict")

# Create containers for processing messages and results
processing_container = st.container(key=f"processing_{st.session_state.results_key}")
results_container = st.container(key=f"results_{st.session_state.results_key}")

if predict_button:
    # Increment the key to force new containers on next render
    st.session_state.results_key += 1
    
    # Clear both containers
    processing_container.empty()
    results_container.empty()
    time_module.sleep(2)  # Wait for 2 seconds
    
    # Execute predictions
    tickers = st.session_state.selected_tickers
    
    # Show processing messages in processing container
    with processing_container:
        print(f"Tickers {tickers}")
        close_price_prediction = execute_pipeline(tickers)
    
    # Display results table in results container
    with results_container:
        display_results(close_price_prediction)