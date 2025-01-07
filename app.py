import streamlit as st
from utils import get_nyse_datetime, market_status
import yfinance as yf
import pandas as pd
import time as time_module

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

def display_results(close_price_prediction):
    """
    Display latest market data and predictions for each ticker.
    Args:
        close_price_prediction: List of tuples containing (ticker, prediction)
    """
    # Initialize last_available_date as None
    last_available_date = None
    
    for ticker, prediction in close_price_prediction:
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
            # if market_status() != "MARKET_OPEN":
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
            st.error(f"Error processing {ticker}: {str(e)}")
            continue

def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    
    # Get the current multiselect state
    updated_sel_tickers = st.session_state.stock_multiselect
    
    # Update selected_tickers to match the multiselect state
    st.session_state.selected_tickers = updated_sel_tickers
    
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")

def render_ui():
    # Title of the app
    st.title("Stock Price Predictor")

    tickers = st.multiselect(
        "Select stocks to predict:",
        st.session_state.tickers,
        default=st.session_state.selected_tickers,
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
