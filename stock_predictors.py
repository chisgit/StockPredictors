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

# Initialize session state variables
if 'tickers' not in st.session_state:
    st.session_state.tickers = initial_tickers.copy()

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

# Multiselect dropdown to select stocks
tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,
    default=st.session_state.selected_tickers,
    key='stock_multiselect'
)

# Update selected tickers based on multiselect
st.session_state.selected_tickers = tickers

# Search bar for new tickers
new_ticker = st.text_input("Search for a stock ticker:", key="new_ticker_input")

# Process new ticker input
if new_ticker:
    try:
        # Validate ticker
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            # Convert to uppercase
            new_ticker_upper = new_ticker.upper()
            
            # Check if ticker is already in the list
            if new_ticker_upper not in [t.upper() for t in st.session_state.tickers]:
                # Add to tickers list
                st.session_state.tickers.insert(0, new_ticker_upper)
                
                # Add to selected tickers
                st.session_state.selected_tickers.append(new_ticker_upper)
                
                # Clear the input by setting it to an empty string (not in session state)
                new_ticker = ""  # This clears the input field on the frontend
                
                # Force a rerun to update the interface
                st.rerun()
            else:
                # Ticker already exists, just select it if not already selected
                if new_ticker_upper not in [t.upper() for t in st.session_state.selected_tickers]:
                    st.session_state.selected_tickers.append(new_ticker_upper)
                    st.rerun()
                
    except Exception as e:
        st.error(f"Error: {e}")

# Rest of the code remains the same as in the previous version (prediction logic)
# ... [keep the entire Predict button and prediction logic from the previous version]
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
        
        # If no data is available for today (market might be closed)
        if latest_data.empty:
            # Fetch the last 7 days of data to display the most recent day available
            latest_data = yf.download(ticker, period='7d')
            last_day = latest_data.iloc[-1]  # Get the last available day of data
            
            # Display last available day's data
            st.write(f"Last available data for {ticker}:")
            st.write(f"Date: {last_day.name}")
            st.write(f"Open: {last_day['Open']}, High: {last_day['High']}, Low: {last_day['Low']}, Close: {last_day['Close']}, Volume: {last_day['Volume']}")
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
