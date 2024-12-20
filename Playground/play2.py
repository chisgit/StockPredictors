import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, time

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
    # Ensure input features are a DataFrame with correct shape
    input_features = pd.DataFrame([{
        'Prev Close': latest_data['Prev Close'].iloc[0],  # Get scalar value from the series
        'Open': latest_data['Open'].iloc[0],  # Get scalar value from the series
        'High': latest_data['High'].iloc[0],  # Get scalar value from the series
        'Low': latest_data['Low'].iloc[0],  # Get scalar value from the series
        'Volume': latest_data['Volume'].iloc[0]  # Get scalar value from the series
    }])
    prediction = model.predict(input_features)
    return prediction.item()

    # Function to check if the market is open
def is_market_open():
        market_open = time(9, 30)
        market_close = time(16, 0)
        now = datetime.now().time()
        return market_open <= now <= market_close

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

print(f"[START] Current tickers: {st.session_state.tickers}")
print(f"[START] Current selected_tickers: {st.session_state.selected_tickers}")

def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    print(f"[UPDATE] Change Type: {type(change)}") # This will print <class 'str'>

    # Access the updated multiselect value directly
    updated_selected_tickers = st.session_state.stock_multiselect
    print(f"[UPDATE] Updated selected_tickers: {updated_selected_tickers}") 

    st.session_state.selected_tickers = updated_selected_tickers
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")

# Create a callback function that will be triggered when the selection changes
def on_change_multiselect():
    # Check if selected tickers have changed
    if st.session_state.selected_tickers != st.session_state.prev_selected_tickers:
        # Update the selected_tickers in session state if the value has changed
        st.session_state.prev_selected_tickers = st.session_state.selected_tickers
        print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")


    # Update session state
    #st.session_state.selected_tickers = updated_selected_tickers

# Multiselect dropdown to select stocks
tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,
    #default=[ticker for ticker in st.session_state.selected_tickers if ticker in st.session_state.tickers],
    default=st.session_state.selected_tickers,  # Preserve selected tickers
    key='stock_multiselect',
    on_change=update_selected_tickers,
    args=['stock_multiselect']
)



print(f"[AFTER MULTISELECT] Multiselect value: {tickers}")

# Search bar for new tickers
new_ticker_input = st.text_input(
    "Search for a stock ticker:", 
    key="new_ticker_input"
)
# Automatically add ticker if Enter is pressed
# Convert to uppercase
new_ticker_upper = new_ticker_input.upper()

print(f"[SEARCH] New ticker input: {new_ticker_upper}")

# Process new ticker input
if new_ticker_upper: 

    if new_ticker_upper not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker_upper)
        st.session_state.selected_tickers.append(new_ticker_upper)  # Add to selected list as well

    try:
        print(f"[PROCESSING] Attempting to fetch data for: {new_ticker_upper}")
        # Try to fetch stock data for the ticker entered
        stock_data = yf.download(new_ticker_upper, period='1d')
        # Validate ticker
        if stock_data.empty:
            # This will only show if ticker is not found, but it won't block adding
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
            print(f"[WARNING] Empty stock data for: {new_ticker_upper}")
        else:
            print(f"[BEFORE CHECK] Current tickers: {st.session_state.tickers}")
            print(f"[BEFORE CHECK] Current selected_tickers: {st.session_state.selected_tickers}")

            # Check if ticker is already in the list
            if new_ticker_upper not in [t.upper() for t in tickers]:
                print(f"[NEW TICKER] Adding new ticker to both lists")
                # Add to tickers list
                # st.session_state.tickers.append(new_ticker_upper)
                # Add to selected tickers
                 # After fetching and appending the new ticker
                if new_ticker_upper not in st.session_state.tickers:
                    st.session_state.tickers.append(new_ticker_upper)

                # Ensure the newly added ticker is included in selected_tickers
                if new_ticker_upper not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(new_ticker_upper)

                #tickers.append(new_ticker_upper)
                print(f"[AFTER ADD] Updated tickers: {st.session_state.tickers}")
                print(f"[AFTER ADD] Updated selected_tickers: {st.session_state.selected_tickers}")
                st.rerun()
            else:
                # Ticker already exists, just select it if not already selected
                if new_ticker_upper not in [t.upper() for t in st.session_state.selected_tickers]:
                    print(f"[EXISTING TICKER] Adding to selected_tickers")
                    tickers.append(new_ticker_upper)
                    print(f"[AFTER SELECT] Updated selected_tickers: {st.session_state.selected_tickers}")
                    st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")
        print(f"[ERROR] Exception occurred: {e}")

    print(f"[BEFORE SYNC] Current multiselect value: {tickers}")
    print(f"[BEFORE SYNC] Current selected_tickers: {st.session_state.selected_tickers}")

    # Synchronize the selected_tickers with the multiselect selections
    st.session_state.selected_tickers = tickers
    print(f"[AFTER FIRST SYNC] selected_tickers: {st.session_state.selected_tickers}")

    # Ensure the selected tickers reflect the current multiselect state
    # Update selected_tickers with only selected items from the multiselect
    st.session_state.selected_tickers = [ticker for ticker in tickers if ticker in st.session_state.tickers]
    print(f"[AFTER FINAL SYNC] selected_tickers: {st.session_state.selected_tickers}")
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
                latest_data = stock_data.iloc[-1].to_frame().T  # Get the last row as a DataFrame
                #latest_data['Prev Close'] = stock_data['Close'].shift(1).iloc[-1]  # Adjust for prediction

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