import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Title of the app
st.title("Stock Price Predictor")

# List of initial stock tickers
initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Initialize the list of tickers in session state if it doesn't exist yet
if 'tickers' not in st.session_state:
    st.session_state.tickers = initial_tickers.copy()

# Initialize the list of selected tickers in session state if it doesn't exist yet
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

# Multiselect dropdown to select stocks
tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,  # Available tickers (including dynamically added ones)
    default=st.session_state.selected_tickers  # Default selected tickers
)

# Add a search bar for ticker input
new_ticker = st.text_input("Search for a stock ticker:")

# If the user presses enter after entering a ticker
if new_ticker:
    try:
        # Try to fetch stock data for the ticker entered
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            # This will only show if ticker is not found, but it won't block adding
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            # If valid, add the ticker to the list (only if it's not already in the list)
            if new_ticker not in st.session_state.tickers:
                # Add the ticker to the list of available tickers
                st.session_state.tickers.insert(0, new_ticker)  # Add to the top of the list
                # Add the ticker to the selected tickers list as well
                st.session_state.selected_tickers.append(new_ticker)
                st.success(f"'{new_ticker}' has been added to your prediction list.")
                
            else:
                # If ticker is already added, inform the user
                st.warning(f"'{new_ticker}' is already in your prediction list.")
    except Exception as e:
        st.error(f"Error: {e}")

# When the user clicks the "Predict" button
if st.button("Predict"):
    results = {}
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    for ticker in tickers:
        st.write(f"Processing {ticker}...")

        # Fetch stock data
        stock_data = yf.download(ticker, start='2010-01-01', end=today.strftime('%Y-%m-%d'))
        if stock_data.empty:
            st.warning(f"No historical data found for {ticker}. Skipping prediction.")
            results[ticker] = None
            continue

        stock_data['Prev Close'] = stock_data['Close'].shift(1)
        stock_data.dropna(inplace=True)

        # Features and target
        try:
            X = stock_data[['Prev Close', 'Open', 'High', 'Low', 'Volume']]
            y = stock_data['Close']
        except KeyError as e:
            st.warning(f"Missing required columns for {ticker}. Skipping.")
            results[ticker] = None
            continue

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Fetch latest data for prediction
        latest_data = yf.download(ticker, start=(yesterday - timedelta(days=1)).strftime('%Y-%m-%d'), end=(today + timedelta(days=1)).strftime('%Y-%m-%d'))
        if latest_data.empty:
            st.warning(f"No data available for {ticker} on prediction dates. Skipping.")
            results[ticker] = None
            continue

        try:
            prev_close = latest_data.loc[yesterday.strftime('%Y-%m-%d'), 'Close']
            today_features = latest_data.loc[today.strftime('%Y-%m-%d'), ['Open', 'High', 'Low', 'Volume']]
            input_features = pd.DataFrame([{
                'Prev Close': prev_close,
                'Open': today_features['Open'],
                'High': today_features['High'],
                'Low': today_features['Low'],
                'Volume': today_features['Volume']
            }])
            prediction = model.predict(input_features)
            results[ticker] = prediction.item()
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
            results[ticker] = None

    # Display results
    st.subheader("Predicted Closing Prices:")
    for ticker, price in results.items():
        if price is not None:
            st.write(f"{ticker}: ${price:.2f}")
        else:
            st.write(f"{ticker}: Data not available")
