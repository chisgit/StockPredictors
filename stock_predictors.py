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

# Predict button logic
if st.button("Predict"):
    results = {}
    output_lines = []  # Initialize output_lines list here
    today = today = datetime.now()
    yesterday = today - timedelta(days=1)
    

    for ticker in tickers:
        st.markdown(f"Processing {ticker}...")

        # Fetch stock data for today and check availability
        latest_data = yf.download(ticker, start=yesterday.strftime('%Y-%m-%d'), end=(today + timedelta(days=1)).strftime('%Y-%m-%d'))
        print(latest_data.head())

        # Check if data for today is available
        data_today = latest_data[latest_data.index.date == today]

        if data_today.empty:
            # If data for today is missing, show the last available day's data (if any)
            latest_data = yf.download(ticker, period='5d')  # Fetch the latest available data
            last_day = latest_data.iloc[-1] if not latest_data.empty else None
            print(latest_data.head())
            print(last_day)
          
            if last_day is not None:
                try:
                    print("Last day where last_day = latest_data.iloc[-1] ):")
                    print(last_day)
                    # Print column names to verify the correct indexing

                   # Formatting values for display
                    open_price = f"{last_day['Open'].iloc[0]:.3f}"
                    close_price = f"{last_day['Close'].iloc[0]:.3f}"
                    high_price = f"{last_day['High'].iloc[0]:.3f}"
                    low_price = f"{last_day['Low'].iloc[0]:.3f}"
                    adj_close_price = f"{last_day['Adj Close'].iloc[0]:.3f}"
                    volume = f"{last_day['Volume'].iloc[0]:,.0f}"  # No decimal places for volume

                    output_line = (
                        f"Open: {open_price}, "
                        f"High: {high_price}, "
                        f"Low: {low_price}, "
                        f"Close: {close_price}, "
                        f"Volume: {volume}"
                    )
                    st.markdown(f"The Market is not open right now.  \n{output_line}")

                except Exception:
                    output_line = f"{ticker}: Data not available"
                    output_lines.append(output_line)

                # Join and print each line followed by a newline character
                st.text("\n".join(output_lines))

                results[ticker] = None  # Skip prediction for this ticker

                continue

            else:
                st.warning(f"No data available for {ticker}. Skipping prediction.")
                results[ticker] = None
                continue
      
        else:
            # Otherwise, proceed with prediction
            stock_data = yf.download(ticker, start='2010-01-01', end=today.strftime('%Y-%m-%d'))
            stock_data['Prev Close'] = stock_data['Close'].shift(1)
            stock_data.dropna(inplace=True)

            # Features and target
            try:
                X = stock_data[['Prev Close', 'Open', 'High', 'Low', 'Volume']]
                y = stock_data['Close']
            except KeyError:
                st.warning(f"Missing required columns for {ticker}. Skipping.")
                results[ticker] = None
                continue

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            latest_data_for_prediction = yf.download(ticker, period='1d')
           
            if not latest_data_for_prediction.empty:
                # Use the most recent data available for prediction
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