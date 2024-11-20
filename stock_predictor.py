import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Title of the app
st.title("Stock Price Predictor")

# List of stock tickers
tickers = st.multiselect("Select stocks to predict:", ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'])

# Add a search bar for ticker input
new_ticker = st.text_input("Search for a stock ticker:")

# If the user enters a ticker, validate it and add to the list
if new_ticker:
    try:
        # Try to fetch stock data for the ticker entered
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            st.error(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            # If valid, add the ticker to the list
            tickers.append(new_ticker)
            st.success(f"'{new_ticker}' has been added to your prediction list.")
    except:
        # Handle invalid ticker input
        st.error(f"Ticker '{new_ticker}' could not be found.")

# When the user clicks the "Predict" button
if st.button("Predict"):
    results = {}
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    for ticker in tickers:
        st.write(f"Processing {ticker}...")

        # Fetch stock data
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

        # Fetch latest data for prediction
        latest_data = yf.download(ticker, start=yesterday.strftime('%Y-%m-%d'), end=(today + timedelta(days=1)).strftime('%Y-%m-%d'))

        if yesterday.strftime('%Y-%m-%d') in latest_data.index and today.strftime('%Y-%m-%d') in latest_data.index:
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
        else:
            results[ticker] = None

    # Display results
    st.subheader("Predicted Closing Prices:")
    for ticker, price in results.items():
        if price is not None:
            st.write(f"{ticker}: ${price:.2f}")
        else:
            st.write(f"{ticker}: Data not available")
