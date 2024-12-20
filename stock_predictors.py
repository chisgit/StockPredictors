import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta, time as dt_time
import pytz  
import time as time_module  
from app import display_market_status
from utils import get_nyse_datetime, get_nyse_date, get_nyse_time, market_status
from data_handler import fetch_data, fetch_features
from data_processor import preprocess_data

def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    
    # Get the current multiselect state
    updated_sel_tickers = st.session_state.stock_multiselect
    
    # Update selected_tickers to match the multiselect state
    st.session_state.selected_tickers = updated_sel_tickers
    
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")

def train_model(train_ready_data, model_type):
    # Features and target
    try:
        X = train_ready_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']]
        y = train_ready_data['Close']

    except KeyError:
        st.warning(f"Missing required columns. Skipping.")

    if model_type == "linear_regression":
        model = LinearRegression()
    else:
        raise ValueError("Unknown model type")
    model.fit(X, y)
    return model

def execute_pipeline(tickers):
    results = []
    status = market_status()
    print(f"Market status: {status}")
    
    for each_ticker in tickers:
        st.markdown(f"Processing {each_ticker}...")

        try:
            # Get training data using NYSE timezone
            nyse_date = get_nyse_date()
            stock_data = fetch_data(each_ticker, nyse_date + timedelta(days=1))
            train_ready_data = preprocess_data(stock_data)
            
            # Train model
            model_type = "linear_regression"
            model = train_model(train_ready_data, model_type)
            
            # Get prediction features and make prediction
            prediction_features = fetch_features(each_ticker)
            prediction = model.predict(prediction_features)
            
            # Convert prediction to scalar if needed
            if isinstance(prediction, (list, np.ndarray)):
                prediction = prediction.flatten()[0] if isinstance(prediction, np.ndarray) else prediction[0]
            
            results.append((each_ticker, prediction))

        except Exception as e:
            print(f"Error processing {each_ticker}: {e}")
            results.append((each_ticker, None))
            continue

    return results

def display_results(results):
    """
    Display latest market data and predictions for each ticker.
    Args:
        results: List of tuples containing (ticker, prediction)
    """
    # Initialize last_available_date as None
    last_available_date = None
    
    for ticker, prediction in results:
        try:
            # Fetch latest data including previous day for previous close
            latest_data = yf.download(ticker, period='5d', interval='1d')
            
            if len(latest_data) < 2:  # Check length first
                st.warning(f"Insufficient data available for {ticker}. Need at least 2 days of data to show previous close.")
                continue
            
            # Get the last available date from the first valid ticker data
            if last_available_date is None:
                last_available_date = latest_data.index[-1].date()
                # Display market status after we have the last available date
                display_market_status(last_available_date)
            
            # Get current day's data
            current_data = latest_data.iloc[-1]
            prev_data = latest_data.iloc[-2]
            
            # Extract values using .item() for all Series
            try:
                current_close = current_data['Close'].item()
                prev_close = prev_data['Close'].item()
                open_price = current_data['Open'].item()
                high_price = current_data['High'].item()
                low_price = current_data['Low'].item()
                volume = current_data['Volume'].item()
            except Exception as e:
                st.error(f"Error extracting data for {ticker}: {str(e)}")
                continue

            # Calculate price difference if close price is available and market is not open
            diff_str = ""
            diff_color = ""  # Initialize with empty string
            if market_status() != "MARKET_OPEN":
                price_diff = prediction - current_close
                if not pd.isna(price_diff):
                    diff_color = "#4CAF50" if price_diff >= 0 else "#FF5252"  # Green if positive, red if negative
                    diff_sign = "+" if price_diff >= 0 else ""
                    diff_str = f'<span style="color: {diff_color}; margin-left: 8px;">({diff_sign}${price_diff:.2f})</span>'

            # Display ticker with predicted close and difference
            st.markdown(
                f'<div style="margin-bottom: 5px; font-size: 1.1em; font-weight: bold;">'
                f'{ticker} - Predicted Close: <span style="font-size: 1.15em;">${prediction:.2f}</span>{diff_str}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Format values
            open_val = f"${open_price:.2f}" if not pd.isna(open_price) else "N/A"
            high_val = f"${high_price:.2f}" if not pd.isna(high_price) else "N/A"
            low_val = f"${low_price:.2f}" if not pd.isna(low_price) else "N/A"
            prev_close_val = f"${prev_close:.2f}" if not pd.isna(prev_close) else "N/A"
            current_val = f"${current_data['Close'].item():.2f}" if market_status() == "MARKET_OPEN" else (f"${current_close:.2f}" if not pd.isna(current_close) else "N/A")
            volume_val = f"{int(volume):,}" if not pd.isna(volume) else "N/A"
            
            # Create grid display
            current_label = "Last Traded" if market_status() == "MARKET_OPEN" else "Close"
            metrics = ['Open', 'High', 'Low', 'Prev Close', current_label, 'Volume']
            values = [open_val, high_val, low_val, prev_close_val, current_val, volume_val]
            
            # Create HTML table with refined styling and theme-appropriate colors
            html = f"""
            <div style="margin: 10px 0;">
                <table style="width: 100%; text-align: center; border-collapse: collapse;">
                    <tr>
                        {''.join(f'<td style="width: 16.66%; padding: 2px;"><small style="opacity: 0.6;">{metric}</small></td>' for metric in metrics)}
                    </tr>
                    <tr>
                        {''.join(f'<td style="width: 16.66%; padding: 2px;"><span style="font-size: 1.1em; opacity: 0.8;">{value}</span></td>' for value in values)}
                    </tr>
                </table>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error displaying data for {ticker}: {str(e)}")
            continue

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