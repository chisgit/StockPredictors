import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Title of the app
st.title("Stock Price Predictor")

# List of initial stock tickers
initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Initialize session state variables
if 'tickers' not in st.session_state:
    st.session_state.tickers = initial_tickers.copy()
    print(f"Initialized 'tickers' in session state: {st.session_state.tickers}")

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []
    print(f"Initialized 'selected_tickers' in session state: {st.session_state.selected_tickers}")

# Multiselect dropdown to select stocks
tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,
    default=st.session_state.selected_tickers,  # Default value is session state selected tickers
    key='stock_multiselect'
)

# Debug print inside multiselect to see what the widget returns
print(f"Multiselect tickers selected: {tickers}")

# Add a button to trigger the update of selected tickers
if st.button('Update Selected Tickers'):
    # Check if there are any changes in the selection
    if tickers != st.session_state.selected_tickers:
        # Get the newly added tickers
        newly_added = [ticker for ticker in tickers if ticker not in st.session_state.selected_tickers]
        newly_removed = [ticker for ticker in st.session_state.selected_tickers if ticker not in tickers]

        # Debug print for added and removed tickers
        if newly_added:
            print(f"Added tickers: {newly_added}")
            st.session_state.selected_tickers.extend(newly_added)  # Add newly selected tickers
        
        if newly_removed:
            print(f"Removed tickers: {newly_removed}")
            for ticker in newly_removed:
                st.session_state.selected_tickers.remove(ticker)  # Remove deselected tickers
        
        # Print the updated selected tickers list
        print(f"Updated selected_tickers: {st.session_state.selected_tickers}")

# Now we display the selected tickers
st.write("Selected Tickers:", st.session_state.selected_tickers)




# Search bar for new tickers
new_ticker = st.text_input("Search for a stock ticker:", key="new_ticker_input")

# Process new ticker input
if new_ticker:
    try:
        # Validate ticker
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
            print(f"Ticker '{new_ticker}' is invalid or does not exist.")
        else:
            # Convert to uppercase
            new_ticker_upper = new_ticker.upper()

            # Check if ticker is already in the list
            if new_ticker_upper not in [t.upper() for t in st.session_state.tickers]:
                # Add to tickers list
                st.session_state.tickers.insert(0, new_ticker_upper)
                print(f"Added {new_ticker_upper} to tickers list: {st.session_state.tickers}")

            # Add to selected tickers if it's not already in the selected list
            if new_ticker_upper not in [t.upper() for t in st.session_state.selected_tickers]:
                st.session_state.selected_tickers.append(new_ticker_upper)
                print(f"Added {new_ticker_upper} to selected_tickers: {st.session_state.selected_tickers}")

            # Clear the input by setting it to an empty string (not in session state)
            new_ticker = ""  # This clears the input field on the frontend
            # Force a rerun to update the interface
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        print(f"Error processing new ticker input: {e}")

# Predict button logic
if st.button("Predict"):
    results = {}
    output_lines = []  # Initialize output_lines list here
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        st.markdown(f"Processing {ticker}...")

        # Fetch stock data for today and check availability
        latest_data = yf.download(ticker, start=yesterday.strftime('%Y-%m-%d'), end=(today + timedelta(days=1)).strftime('%Y-%m-%d'))

        # Debugging print to show stock data for today
        print(f"Fetched stock data for {ticker}:\n{latest_data.head()}")

        # Check if data for today is available
        data_today = latest_data[latest_data.index.date == today]

        if data_today.empty:
            # If data for today is missing, show the last available day's data (if any)
            latest_data = yf.download(ticker, period='5d')  # Fetch the latest available data
            last_day = latest_data.iloc[-1] if not latest_data.empty else None

            if last_day is not None:
                try:
                    # Formatting values for display
                    open_price = f"{last_day['Open']:.3f}"
                    close_price = f"{last_day['Close']:.3f}"
                    high_price = f"{last_day['High']:.3f}"
                    low_price = f"{last_day['Low']:.3f}"
                    adj_close_price = f"{last_day['Adj Close']:.3f}"
                    volume = f"{last_day['Volume']:,}"  # No decimal places for volume

                    output_line = (
                        f"Open: {open_price}, "
                        f"High: {high_price}, "
                        f"Low: {low_price}, "
                        f"Close: {close_price}, "
                        f"Volume: {volume}"
                    )
                    print(f"Market is closed, using last available data: {output_line}")
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

                print(f"Using last available data for prediction: {last_day}")
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
                print(f"Prediction for {ticker}: {prediction.item()}")

            except KeyError as e:
                print(f"KeyError for {ticker}: {e}")
                st.warning(f"Missing data for {ticker}, skipping prediction.")
                results[ticker] = None

    # Display predictions
    st.subheader("Predictions for selected stocks:")
    for ticker, prediction in results.items():
        if prediction is None:
            st.write(f"{ticker}: No data or prediction unavailable.")
        else:
            st.write(f"{ticker}: Predicted Close Price = {prediction:.2f}")
