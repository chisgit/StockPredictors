import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Stock Price Predictor")

# Function to handle new ticker input
def handle_new_ticker(new_ticker):
    try:
        # Validate ticker by trying to download stock data
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            # Convert to uppercase
            new_ticker_upper = new_ticker.upper()

            # Check if ticker is already in the list of tickers
            if new_ticker_upper not in [t.upper() for t in st.session_state.tickers]:
                # Add to the available tickers list
                st.session_state.tickers.insert(0, new_ticker_upper)

            # Check if ticker is already selected
            if new_ticker_upper not in [t.upper() for t in st.session_state.selected_tickers]:
                # Add the ticker to selected_tickers
                st.session_state.selected_tickers.append(new_ticker_upper)

            # Clear the input field
            st.session_state.new_ticker_input = ""  # Clear the input field
            st.rerun()  # Trigger a rerun to update the dropdown
    except Exception as e:
        st.error(f"Error: {e}")

# List of initial stock tickers
initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Initialize session state variables
if 'tickers' not in st.session_state:
    st.session_state.tickers = initial_tickers.copy()

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

# Dropdown to select stocks
tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,
    default=st.session_state.selected_tickers,
    key='stock_multiselect'
)

# Update selected tickers based on multiselect
st.session_state.selected_tickers = tickers

# Search bar for new tickers (placed under the dropdown)
new_ticker = st.text_input("Search for a stock ticker:", key="new_ticker_input")

# Handle new ticker input automatically without a button
if new_ticker:
    handle_new_ticker(new_ticker)

# Predict button logic
if st.button("Predict"):
    results = {}
    output_lines = []  # Initialize output_lines list here
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    isIntraday = False

    for ticker in tickers:
        st.markdown(f"Processing {ticker}...")

        # Fetch stock data for today and check availability
        latest_data = yf.download(ticker, period='1d')

        # Check if it is intraday: 
        market_open = time(9, 30)
        market_close = time(16, 0)
        now = datetime.now().time()

        if market_open <= now <= market_close:
            isIntraday = True
        else:
            isIntraday = False

        today_features = latest_data.loc[today.strftime('%Y-%m-%d'), ['Open', 'High', 'Low', 'Volume', 'Close']]

        data_today = latest_data[latest_data.index.date == today.date()]

        if data_today.empty and not isIntraday:
            latest_data = yf.download(ticker, period='5d')
            last_day = latest_data.iloc[-1] if not latest_data.empty else None

            if last_day is not None:
                try:
                    open_price = f"{last_day['Open']:.3f}"
                    close_price = f"{last_day['Close']:.3f}"
                    high_price = f"{last_day['High']:.3f}"
                    low_price = f"{last_day['Low']:.3f}"
                    adj_close_price = f"{last_day['Adj Close']:.3f}"
                    volume = f"{last_day['Volume']:,}"

                    output_line = (
                        f"Open: {open_price}, "
                        f"High: {high_price}, "
                        f"Low: {low_price}, "
                        f"Close: {close_price}, "
                        f"Volume: {volume}"
                    )
                    st.markdown(f"Market is closed. Here is the data from the last available day: \n{output_line}")
                except Exception:
                    output_line = f"{ticker}: Data not available"
                    output_lines.append(output_line)

                st.text("\n".join(output_lines))

                results[ticker] = None  # Skip prediction for this ticker
                continue

        else:
            # Proceed with prediction
            stock_data = yf.download(ticker, start='2010-01-01', end=today.date())

            if stock_data.empty:
                st.warning(f"No data returned for {ticker}. Skipping.")
                continue

            stock_data['Prev Close'] = stock_data['Close'].shift(1)
            stock_data.dropna(inplace=True)

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

            latest_data_for_prediction = yf.download(ticker, period='5d')
            latest_data_for_prediction['Prev Close'] = latest_data_for_prediction['Close'].shift(1)
            latest_data_for_prediction.dropna(inplace=True)

            if not latest_data_for_prediction.empty:
                last_row = latest_data_for_prediction.iloc[-1]
                input_features = pd.DataFrame([{
                    'Prev Close': last_row['Prev Close'],
                    'Open': last_row['Open'],
                    'High': last_row['High'],
                    'Low': last_row['Low'],
                    'Volume': last_row['Volume']
                }])
                
                # Perform prediction
                prediction = model.predict(input_features)
                results[ticker] = prediction.item()
                st.write(f"Predicted closing price for {ticker}: ${prediction.item():.2f}")
            else:
                st.warning(f"No recent data available for {ticker}. Skipping prediction.")

    # Display results
    st.subheader("Predicted Closing Prices:")
    for ticker, price in results.items():
        if price is not None:
            st.write(f"{ticker}: ${price:.2f}")
        else:
            st.write(f"{ticker}: Data not available")
