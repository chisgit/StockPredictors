import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time  # time module for time-related functions
from datetime import datetime, time as datetime_time  # datetime and time from datetime

# Function to fetch stock data
def fetch_stock_data(ticker):
    try:
        data = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to preprocess stock data
def preprocess_data(data):
    data['Prev Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    return data

# Function to train the model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict stock price
def predict_price(model, latest_data):
    input_features = pd.DataFrame([{
        'Prev Close': latest_data['Prev Close'].iloc[0],
        'Open': latest_data['Open'].iloc[0],
        'High': latest_data['High'].iloc[0],
        'Low': latest_data['Low'].iloc[0],
        'Volume': latest_data['Volume'].iloc[0]
    }])
    prediction = model.predict(input_features)
    return prediction.item()

# Function to check if the market is open
def is_market_open():
    market_open_time = datetime.strptime('09:30', '%H:%M').time()  # Market opens at 9:30 AM
    market_close_time = datetime.strptime('16:00', '%H:%M').time()  # Market closes at 4:00 PM
    now = datetime.now().time()  # Get the current time
    return market_open_time <= now <= market_close_time

# Title of the app
st.title("Stock Price Predictor")

# Initial stock tickers
initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Initialize session state variables
if 'tickers' not in st.session_state:
    st.session_state.tickers = initial_tickers.copy()

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

# Track the previous value of selected_tickers
if 'prev_selected_tickers' not in st.session_state:
    st.session_state.prev_selected_tickers = st.session_state.selected_tickers

# Update selected tickers callback
def update_selected_tickers(change):
    updated_selected_tickers = st.session_state.stock_multiselect
    st.session_state.selected_tickers = updated_selected_tickers

# Multiselect dropdown to select stocks
tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,
    default=st.session_state.selected_tickers,
    key='stock_multiselect',
    on_change=update_selected_tickers
)

# Search bar for new tickers
new_ticker_input = st.text_input("Search for a stock ticker:", key="new_ticker_input")
new_ticker_upper = new_ticker_input.upper()

# Process new ticker input
if new_ticker_upper:
    if new_ticker_upper not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker_upper)
        st.session_state.selected_tickers.append(new_ticker_upper)

    try:
        stock_data = yf.download(new_ticker_upper, period='1d')
        if stock_data.empty:
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            if new_ticker_upper not in st.session_state.selected_tickers:
                st.session_state.selected_tickers.append(new_ticker_upper)
                st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# Predict button logic
if st.button("Predict"):
    results = {}
    if is_market_open():
        st.warning("The market is currently open. Predictions may not be final.")

    for ticker in tickers:
        st.markdown(f"Processing {ticker}...")
        stock_data = fetch_stock_data(ticker)
        if stock_data is not None and not stock_data.empty:
            processed_data = preprocess_data(stock_data)
            X = processed_data[['Prev Close', 'Open', 'High', 'Low', 'Volume']]
            y = processed_data['Close']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Train model
            model = train_model(X_train, y_train)

            # Get the latest data for prediction
            latest_data = stock_data.iloc[-1].to_frame().T

            # Perform prediction
            prediction = predict_price(model, latest_data)
            results[ticker] = prediction
            st.write(f"Predicted closing price for {ticker}: ${prediction:.2f}")
        else:
            st.warning(f"No data available for {ticker}. Skipping prediction.")

    # Display results
    st.subheader("Predicted Closing Prices:")
    for ticker, price in results.items():
        if price is not None:
            st.write(f"{ticker}: ${price:.2f}")
        else:
            st.write(f"{ticker}: Data not available for prediction.")
