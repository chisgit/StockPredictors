import streamlit as st
from utils import get_nyse_datetime, market_status
from render_helpers import get_recent_data, group_predictions_by_ticker
import yfinance as yf
import pandas as pd
import time as time_module
from rules import UI_RULES, MARKET_HOURS
from display_market_status import display_market_status

def display_results(predictions):
    """Display latest market data and predictions for each ticker."""
    # Enforce maximum tickers rule
    if len(st.session_state.selected_tickers) > UI_RULES['max_tickers']:
        st.warning(f"Maximum {UI_RULES['max_tickers']} tickers can be selected at once. Only showing first {UI_RULES['max_tickers']} tickers.")
        st.session_state.selected_tickers = st.session_state.selected_tickers[:UI_RULES['max_tickers']]
    
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
            st.error(f"Error processing {ticker}: {str(e)}")
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
            
            try:
                current_close = current_data['Close'].item()
                prev_close = prev_data['Close'].item()
                open_price = current_data['Open'].item()
                high_price = current_data['High'].item()
                low_price = current_data['Low'].item()
                volume = current_data['Volume'].item()

                # Format numbers according to UI rules
                current_close = round(current_close, UI_RULES['price_decimals'])
                prev_close = round(prev_close, UI_RULES['price_decimals'])
                open_price = round(open_price, UI_RULES['price_decimals'])
                high_price = round(high_price, UI_RULES['price_decimals'])
                low_price = round(low_price, UI_RULES['price_decimals'])
            except Exception as e:
                st.error(f"Error extracting data for {ticker}: {str(e)}")
                continue

            # Combine ticker header with predictions on same line
            predictions_html = ""
            for i, prediction in enumerate(grouped_predictions[ticker]):
                model_type = "Linear Regression" if i == 0 else "XGBoost"
                price_diff = prediction - current_close
                diff_color = "#4CAF50" if price_diff >= 0 else "#FF5252"
                
                if price_diff > 0:
                    diff_sign = "+"
                elif price_diff < 0:
                    diff_sign = "-"
                else:
                    diff_sign = ""
                    
                diff_str = f'<span style="color: {diff_color}; margin-left: 8px;">({diff_sign}${abs(price_diff):.2f})</span>'
                predictions_html += f'{model_type}: <span style="font-size: 1.1em;">${prediction:.2f}</span>{diff_str} &nbsp;&nbsp;'
            
            # Combined ticker and predictions display with reduced whitespace
            st.markdown(
                f'<div style="margin: 10px 0 5px 0;">'
                f'<span style="font-size: 1.2em; font-weight: bold;">{ticker}</span>'
                f'<span style="margin-left: 10px;">{predictions_html}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Format and display single grid
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
            
            html = f"""
            <div style="margin: 5px 0;">
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
            st.error(f"Error processing {ticker}: {str(e)}")
            continue

    # Display next day predictions if available
    if next_day_close_predictions:
        st.markdown("---")
        st.subheader("Next Day's Close Predictions")
        
        # Process each ticker for next day predictions
        for ticker in dict.fromkeys(t for t, _ in next_day_close_predictions):
            if ticker in grouped_next_day_predictions and ticker in ticker_data:
                try:
                    latest_data = ticker_data[ticker]  # Use cached data
                    if not latest_data.empty and len(latest_data) >= 1:  # Fixed condition
                        current_close = latest_data['Close'].iloc[-1].item()  # Added .item()
                        
                        # Combine ticker header with predictions on same line
                        predictions_html = ""
                        for i, prediction in enumerate(grouped_next_day_predictions[ticker]):
                            model_type = "Linear Regression" if i == 0 else "XGBoost"
                            price_diff = prediction - current_close
                            diff_color = "#4CAF50" if price_diff >= 0 else "#FF5252"
                            
                            if price_diff > 0:
                                diff_sign = "+"
                            elif price_diff < 0:
                                diff_sign = "-"
                            else:
                                diff_sign = ""
                                
                            diff_str = f'<span style="color: {diff_color}; margin-left: 8px;">({diff_sign}${abs(price_diff):.2f})</span>'
                            predictions_html += f'{model_type}: <span style="font-size: 1.1em;">${prediction:.2f}</span>{diff_str} &nbsp;&nbsp;'
                        
                        # Combined ticker and predictions display with reduced whitespace
                        st.markdown(
                            f'<div style="margin: 10px 0 5px 0;">'
                            f'<span style="font-size: 1.2em; font-weight: bold;">{ticker}</span>'
                            f'<span style="margin-left: 10px;">{predictions_html}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
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
