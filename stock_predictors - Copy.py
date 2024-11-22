import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    default=st.session_state.selected_tickers,  # Default value is session state selected tickers
)

# Add a button to trigger the update of selected tickers
if st.button('Update Selected Tickers'):
    # Check if there are any changes in the selection
    if tickers != st.session_state.selected_tickers:
        # Get the newly added tickers
        newly_added = [ticker for ticker in tickers if ticker not in st.session_state.selected_tickers]
        newly_removed = [ticker for ticker in st.session_state.selected_tickers if ticker not in tickers]

        if newly_added:
            st.session_state.selected_tickers.extend(newly_added)  # Add newly selected tickers
        
        if newly_removed:
            for ticker in newly_removed:
                st.session_state.selected_tickers.remove(ticker)  # Remove deselected tickers

# Display selected tickers
st.write("Selected Tickers:", st.session_state.selected_tickers)

# Search bar for new tickers
new_ticker = st.text_input("Search for a stock ticker:")

# Process new ticker input
if new_ticker:
    try:
        # Validate ticker
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            new_ticker_upper = new_ticker.upper()

            # Add to tickers list if not already present
            if new_ticker_upper not in [t.upper() for t in st.session_state.tickers]:
                st.session_state.tickers.insert(0, new_ticker_upper)

            # Add to selected tickers if not already present
            if new_ticker_upper not in [t.upper() for t in st.session_state.selected_tickers]:
                st.session_state.selected_tickers.append(new_ticker_upper)

            # Clear the input field
            new_ticker = ""  # This clears the input field

    except Exception as e:
        st.error(f"Error: {e}")

# Predict button logic
if st.button("Predict"):
    results = {}
    output_lines = []  # Initialize output_lines list here
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    for ticker in tickers:
        st.markdown(f"Processing {ticker}...")

        # Fetch stock data for today and check availability
        latest_data = yf.download(ticker, start=yesterday.strftime('%Y-%m-%d'), end=(today + timedelta(days=1)).strftime('%Y-%m-%d'))

        # Check if data for today is available
        data_today = latest_data[latest_data.index.date == today]

        if data_today.empty:
            # If data for today is missing, show the last available day's data
            latest_data = yf.download(ticker, period='5d')  # Fetch the latest available data
            last_day = latest_data.iloc[-1] if not latest_data.empty else None

            if last_day is not None:
                output_line = (
                    f"Open: {last_day['Open']:.3f}, "
                    f"High: {last_day['High']:.3f}, "
                    f"Low: {last_day['Low']:.3f}, "
                    f"Close: {last_day['Close']:.3f}, "
                    f"Volume: {last_day['Volume']:,}"
                )
                st.markdown(f"The Market is not open right now.  \n{output_line}")
            else:
                st.warning(f"No data available for {ticker}. Skipping prediction.")
                results[ticker] = None
                continue

        else:
            # Proceed with prediction
            stock_data = yf.download(ticker, start='2010-01-01', end=today.strftime('%Y-%m-%d'))
            stock_data['Prev Close'] = stock_data['Close'].shift(1)
            stock_data.dropna(inplace=True)

            # Features and target
            X = stock_data[['Prev Close', 'Open', 'High', 'Low', 'Volume']]
            y = stock_data['Close']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            latest_data_for_prediction = yf.download(ticker, period='1d')
           
            if not latest_data_for_prediction.empty:
                latest_data_for_prediction = yf.download(ticker, period='5d')
                last_day = latest_data_for_prediction.iloc[-1]

                st.write(f"Market is not open right now. Here is the data from the last available day:")
                st.write(f"Date: {last_day.name}")
                st.write(f"Open: {last_day['Open']}, High: {last_day['High']}, Low: {last_day['Low']}, Close: {last_day['Close']}, Volume: {last_day['Volume']}")
                results[ticker] = None
                continue

            try:
                prev_close = latest_data_for_prediction.loc[yesterday.strftime('%Y-%m-%d'), 'Close']
                today_features = latest_data_for_prediction.loc[today.strftime('%Y-%m-%d'), ['Open', 'High', 'Low', 'Volume']]
                input_features = pd.DataFrame([{
                    'Prev Close': prev_close,
                    'Open': today_features['Open'],
                    'High': today_features['High'],
                    'Low': today_features['Low'],
                    'Volume': today_features['Volume']
                }])
                prediction = model.predict(input_features)
                results[ticker] = prediction.item()

            except KeyError as e:
                st.warning(f"Missing data for {ticker}, skipping prediction.")
                results[ticker] = None

    # Display predictions
    st.subheader("Predictions for selected stocks:")
    for ticker, prediction in results.items():
        if prediction is None:
            st.write(f"{ticker}: No data or prediction unavailable.")
        else:
            st.write(f"{ticker}: Predicted Close Price = {prediction:.2f}")
