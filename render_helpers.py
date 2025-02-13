import yfinance as yf
import streamlit as st

def get_recent_data(ticker):
    """Download recent stock data for the given ticker."""
    try:
        data = yf.download(ticker, period='5d', interval='1d')
        if data.empty or len(data) < 2:
            raise ValueError(f"Insufficient data available for {ticker}. Need at least 2 days of data.")
        return data
    except Exception as e:
        print(f"Error getting data for {ticker}: {str(e)}")
        return None

def group_predictions_by_ticker(todays_close_predictions, next_day_close_predictions):
    grouped_predictions = {}
    grouped_next_day_predictions = {}
    
    for ticker, prediction in todays_close_predictions:
        if ticker not in grouped_predictions:
            grouped_predictions[ticker] = []
        grouped_predictions[ticker].append(prediction)

    for ticker, prediction in next_day_close_predictions:
        if ticker not in grouped_next_day_predictions:
            grouped_next_day_predictions[ticker] = []
        grouped_next_day_predictions[ticker].append(prediction)

    return grouped_predictions, grouped_next_day_predictions

def extract_and_format_ticker_data(latest_data):
    """Extract and format ticker data from the latest data."""
    if latest_data is None or len(latest_data) < 2:
        return None  # Or handle error as needed

    current_data = latest_data.iloc[-1]
    prev_data = latest_data.iloc[-2]

    # Ensure that the volume is correctly extracted as a scalar
    volume = current_data['Volume'].item()

    # Prepare formatted data as a list of tuples (label, value)
    formatted_data = [
        ('Open', round(current_data['Open'].item(), 2)),
        ('High', round(current_data['High'].item(), 2)),
        ('Low', round(current_data['Low'].item(), 2)),
        ('Prev Close', round(prev_data['Close'].item(), 2)),
        ('Current Close', round(current_data['Close'].item(), 2)),
        ('Volume', int(volume))
    ]
    return formatted_data

def preds_sameline(predictions, current_val):
    predictions_html = ''
    for i, prediction in enumerate(predictions):
        model_type = "Linear Regression" if i == 0 else "XGBoost"
        price_diff = prediction - current_val
        diff_color = "#4CAF50" if price_diff >= 0 else "#FF5252"
        
        if price_diff > 0:
            diff_sign = "+"
        elif price_diff < 0:
            diff_sign = "-"
        else:
            diff_sign = ""
            
        diff_str = f'<span style="color: {diff_color}; margin-left: 8px;">({diff_sign}${abs(price_diff):.2f})</span>'
        predictions_html += f'{model_type}: <span style="font-size: 1.1em;">${prediction:.2f}</span>{diff_str} &nbsp;&nbsp;'
    return predictions_html

def display_predictions(ticker, predictions_html):
    """Display predictions for a given ticker."""
    st.markdown(
        f'<div style="margin: 10px 0 5px 0;">'
        f'<span style="font-size: 1.2em; font-weight: bold;">{ticker}</span>'
        f'<span style="margin-left: 10px;">{predictions_html}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

def create_grid_display(open_val, high_val, low_val, prev_close_val, current_val, volume):
    metrics = ['Open', 'High', 'Low', 'Prev Close', 'Current Close', 'Volume']
    values = [open_val, high_val, low_val, prev_close_val, current_val, f"{int(volume):,}"]
    
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
    return html

def search_and_add_ticker(new_ticker):
    # Clears out the ticker if there's a change in removing the ticker
    # This way it doesn't keep appearing in the search bar after it's been removed
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