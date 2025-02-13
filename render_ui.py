import streamlit as st
from utils import get_nyse_datetime, market_status
from render_helpers import get_recent_data, group_predictions_by_ticker, format_ticker_data, display_predictions, preds_sameline, create_grid_display, search_and_add_ticker
import yfinance as yf
import pandas as pd
import time as time_module
from rules import UI_RULES, MARKET_HOURS
from display_market_status import display_market_status
from render_preds_processor import process_todays_predictions

def enforce_max_tickers():
    if len(st.session_state.selected_tickers) > UI_RULES['max_tickers']:
        st.warning(f"Maximum {UI_RULES['max_tickers']} tickers can be selected at once.")
        st.session_state.selected_tickers = st.session_state.selected_tickers[:UI_RULES['max_tickers']]

def display_results(predictions):
    """Display latest market data and predictions for each ticker."""
    enforce_max_tickers()
    
    last_available_date = None
    todays_close_predictions = predictions[0]
    next_day_close_predictions = predictions[1]
    
    print(f"Today's Close and Next_Day Predictions: {predictions}")
    
    # Group predictions by ticker while preserving order
    grouped_predictions, grouped_next_day_predictions = group_predictions_by_ticker(todays_close_predictions, next_day_close_predictions)
    ticker_data = {}  # Cache for storing downloaded data per ticker
    
    for ticker in dict.fromkeys(t for t, _ in todays_close_predictions):
        try:
            # Download data once per ticker
            recent_data = get_recent_data(ticker)
            if recent_data is None:
                continue
            ticker_data[ticker] = recent_data
            if len(ticker_data[ticker]) >= 2 and last_available_date is None:
                last_available_date = ticker_data[ticker].index[-1].date()
                display_market_status(last_available_date)
                st.subheader("Today's Close Predictions")
        except Exception as e:
            st.error(f"Error in the downloading of current data for {ticker}: {str(e)}")
            continue

    # Process each ticker once for today's close
    for ticker in dict.fromkeys(t for t, _ in todays_close_predictions):
        try:
            latest_data = ticker_data.get(ticker)
            if latest_data is None or len(latest_data) < 2:
                st.warning(f"Insufficient data available for {ticker}. Need at least 2 days of data to show previous close.")
                continue
            
            current_data = latest_data.iloc[-1]
            prev_data = latest_data.iloc[-2]

            # Ensure that the volume is correctly extracted as a scalar
            volume = current_data['Volume'].item()

            # Call the formatting function with the correct parameters
            formatted_data = format_ticker_data(current_data, prev_data, volume)
            open_val = formatted_data['open_price']
            high_val = formatted_data['high_price']
            low_val = formatted_data['low_price']
            prev_close_val = formatted_data['prev_close']
            current_val = formatted_data['current_close']
            
            # Combine ticker header with predictions on same line
            predictions_html = preds_sameline(grouped_predictions[ticker], current_val)
            
            # Replace the existing display logic with a call to display_predictions
            display_predictions(ticker, predictions_html)

            # Create grid display
            grid_html = create_grid_display(open_val, high_val, low_val, prev_close_val, current_val, volume)
            st.markdown(grid_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
            continue

    # Display next day predictions if available
    if next_day_close_predictions:
        st.markdown("---")
        st.subheader("Next Day's Close Predictions")
        
        # Process each ticker for next day's predictions
        for ticker in dict.fromkeys(t for t, _ in next_day_close_predictions):
            if ticker in grouped_next_day_predictions and ticker in ticker_data:
                try:
                    latest_data = ticker_data[ticker]  # Use cached data
                    if not latest_data.empty and len(latest_data) >= 1:
                        current_close = latest_data['Close'].iloc[-1].item()  # Added .item()
                        
                        # Use preds_sameline to format the predictions
                        predictions_html = preds_sameline(grouped_next_day_predictions[ticker], current_close)
                        
                        # Display the predictions
                        display_predictions(ticker, predictions_html)
                except Exception as e:
                    st.error(f"Error processing next day predictions for {ticker}: {str(e)}")
                    continue

def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    
    # Get the current multiselect state
    updated_sel_tickers = st.session_state.stock_multiselect
    
    # Update selected_tickers to match the multiselect state
    st.session_state.selected_tickers = updated_sel_tickers
    
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")

def render_ui():
    st.title("Stock Price Predictor")

    # Initialize default tickers if not already in session state
    if 'tickers' not in st.session_state:
        st.session_state.tickers = UI_RULES['default_tickers']
    
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = UI_RULES['default_tickers'].copy()

    # Add warning if max tickers limit is reached
    if len(st.session_state.selected_tickers) >= UI_RULES['max_tickers']:
        st.warning(f"Maximum {UI_RULES['max_tickers']} tickers can be selected")

    tickers = st.multiselect(
        "Select stocks to predict:",
        st.session_state.tickers,
        default=st.session_state.selected_tickers,
        key='stock_multiselect',
        on_change=update_selected_tickers,
        args=['stock_multiselect'],
        max_selections=UI_RULES['max_tickers']
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
        search_and_add_ticker(new_ticker)
    st.session_state.new_ticker = ''

    # Create two distinct containers
    processing_container = st.container(key=f"processing_container_{st.session_state.results_key}")
    results_container = st.container(key=f"results_container_{st.session_state.results_key}")

    # Predict button and results area
    predict_button = st.button("Predict")

    if predict_button:
        # Set a flag in session state to indicate prediction is requested
        st.session_state.run_prediction = True
        # Trigger a rerun to allow the controller to execute the pipeline
        st.rerun()
