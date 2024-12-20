import streamlit as st
from utils import get_nyse_datetime, market_status
import yfinance as yf
import pandas as pd

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
            st.error(f"Error processing {ticker}: {str(e)}")
            continue

def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    
    # Get the current multiselect state
    updated_sel_tickers = st.session_state.stock_multiselect
    
    # Update selected_tickers to match the multiselect state
    st.session_state.selected_tickers = updated_sel_tickers
    
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")