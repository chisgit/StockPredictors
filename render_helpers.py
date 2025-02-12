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

def format_ticker_data(current_data, prev_data, volume):
    """Format the ticker data for display."""
    formatted_data = {
        'current_close': round(current_data['Close'].item(), 2),  # Round to 2 decimals
        'prev_close': round(prev_data['Close'].item(), 2),        # Round to 2 decimals
        'open_price': round(current_data['Open'].item(), 2),      # Round to 2 decimals
        'high_price': round(current_data['High'].item(), 2),      # Round to 2 decimals
        'low_price': round(current_data['Low'].item(), 2),        # Round to 2 decimals
        'volume': int(volume)                                       # Return as int for calculations
    }
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