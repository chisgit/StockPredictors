import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta, time
import pytz  # Add this import for timezone handling

# Define NYSE market hours (Global constants)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
NYSE_TIMEZONE = pytz.timezone('America/New_York')

def get_nyse_datetime():
    """Get current datetime in NYSE timezone"""
    return datetime.now(pytz.UTC).astimezone(NYSE_TIMEZONE)

def get_nyse_date():
    """Get current date in NYSE timezone"""
    return get_nyse_datetime().date()

def get_nyse_time():
    """Get current time in NYSE timezone"""
    return get_nyse_datetime().time()

def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    print(f"[UPDATE] Change Type: {type(change)}") # This will print <class 'str'>

    # Access the updated multiselect value directly
    updated_sel_tickers = st.session_state.stock_multiselect
    print(f"[UPDATE] Updated selected_tickers: {updated_sel_tickers}") 

    st.session_state.selected_tickers = updated_sel_tickers
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")

def market_status():
    """Check the current market status based on NYSE time"""
    current_time = get_nyse_time()
    
    if current_time < MARKET_OPEN:
        return "BEFORE_MARKET_OPEN"
    elif MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return "MARKET_OPEN"
    else:
        return "AFTER_MARKET_CLOSE"

def fetch_data(ticker, end_date, start_date='2010-01-01'):
    """
    Fetch historical data for a given ticker
    """
    # Get current date in NYSE timezone
    end_date = get_nyse_date()
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def preprocess_data(stock_data):
    # """
    # Preprocesses stock data by adding the 'Prev Close' column and dropping rows with missing data.
    
    # Args:
    #     stock_data (pd.DataFrame): DataFrame containing stock data with at least a 'Close' column.
        
    # Returns:
    #     pd.DataFrame: Processed DataFrame with 'Prev Close' added and rows with missing data dropped.
    # """

    # Check if data is empty
    if stock_data.empty:
        st.warning("No data returned for the ticker. Skipping.")
        return stock_data
    
    # print(f"Stock Data in preprocess after fetch")
    # print(stock_data.head(5))
    # print(f"Stock Data Index")
    # print(stock_data.index)
    # print(f"Stock Data Columns")
    # print(stock_data.columns)
    
    
    ticker_value = stock_data.columns[1][1] #Get the ticker value from the dataframe

    # Add 'Prev Close' column by shifting 'Close' by 1 to the next row but ensuring to preserve the ticker value
    stock_data[('Prev Close', ticker_value)] = stock_data[('Close', ticker_value)].shift(1)
    stock_data.dropna(inplace=True) #removes the first record in the data set (since there is no close for the previous row)

    return stock_data

def train_model(train_ready_data, model_type):
    # """
    # Trains the model on the provided data.

    # Args:
    # - train_ready_data (pd.DataFrame): The preprocessed stock data.
    # - model_type (str): The type of model to train ("linear_regression", "decision_tree", "random_forest").

    # Returns:
    # - model: The trained model.
    # """

    # Features and target
    try:
        X = train_ready_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']]
        y = train_ready_data['Close']

    except KeyError:
        st.warning(f"Missing required columns. Skipping.")

    if model_type == "linear_regression":
        model = LinearRegression()
    # elif model_type == "decision_tree":
    #     model = DecisionTreeRegressor()
    # elif model_type == "random_forest":
    #     model = RandomForestRegressor()
    else:
        raise ValueError("Unknown model type")
    model.fit(X, y)
    return model

def fetch_features(ticker):
    """
    Fetch recent stock data for prediction features.
    """
    # Get current time in NYSE timezone
    current_time = get_nyse_datetime()
    
    # Download data using period to ensure we get the right trading days
    stock_data = yf.download(ticker, period='5d')
    print(f"Fetch Features stock_data")
    print(stock_data)
    
    prediction_data = preprocess_data(stock_data).tail(1)
    prediction_features = prediction_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']]
    
    print(f"Fetch DF to get Features prediction_data")
    print(prediction_features)

    return prediction_features

def execute_pipeline(tickers):
    results = []
    status = market_status()
    print(f"Market status: {status}")
    
    for each_ticker in tickers:
        st.markdown(f"Processing {each_ticker}...")

        try:
            # Get training data using NYSE timezone
            nyse_date = get_nyse_date()
            stock_data = fetch_data(each_ticker, nyse_date + timedelta(days=1))
            train_ready_data = preprocess_data(stock_data)
            
            # Train model
            model_type = "linear_regression"
            model = train_model(train_ready_data, model_type)
            
            # Get prediction features and make prediction
            prediction_features = fetch_features(each_ticker)
            prediction = model.predict(prediction_features)
            
            # Convert prediction to scalar if needed
            if isinstance(prediction, (list, np.ndarray)):
                prediction = prediction.flatten()[0] if isinstance(prediction, np.ndarray) else prediction[0]
            
            results.append((each_ticker, prediction))

        except Exception as e:
            print(f"Error processing {each_ticker}: {e}")
            results.append((each_ticker, None))
            continue

    return results

def display_market_status(last_available_date=None):
    """
    Display market status with icon and description.
    Args:
        last_available_date: Last available trading date (for BEFORE_MARKET_OPEN state)
    """
    # Get current time in Eastern Time
    current_time = get_nyse_datetime()
    status = market_status()
    
    # Format and display current date
    date_str = current_time.strftime('%A, %B %d, %Y')
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 0;'>{date_str}</h2>", unsafe_allow_html=True)

    last_date_str = last_available_date.strftime('%A, %B %d')
    time_str = current_time.strftime('%I:%M %p EST')  # Add EST to the time

    if status == "BEFORE_MARKET_OPEN":
        st.markdown(f"""<div style='text-align: center; margin-top: -10px;'>
            <h3 style='margin-bottom: 0;'>⏳ Market hasn't opened yet</h3>
            <div style='font-size: 10pt; margin-top: -10px;'>Predicted closing prices are for {last_date_str} based on the latest available data</div>
        </div>""", unsafe_allow_html=True)
    elif status == "MARKET_OPEN":
        st.markdown(f"""<div style='text-align: center; margin-top: -10px;'>
            <h3 style='margin-bottom: 0;'>🔔 Market is Open</h3>
            <div style='font-size: 10pt; margin-top: -10px;'>Predicted closing price for {last_date_str} based on current time: {time_str}</div>
        </div>""", unsafe_allow_html=True)
    else:  # AFTER_MARKET_CLOSE
        st.markdown(f"""<div style='text-align: center; margin-top: -10px;'>
            <h3 style='margin-bottom: 0;'>🔴 Market is Closed</h3>
            <div style='font-size: 10pt; margin-top: -10px;'>Predicted closing price for {last_date_str}. Today's closing prices are final.</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")  # Add a separator line

def display_results(results):
    """
    Display latest market data and predictions for each ticker.
    Args:
        results: List of tuples containing (ticker, prediction)
    """
    # Initialize last_available_date as None
    last_available_date = None
    
    for ticker, prediction in results:
        try:
            # Fetch latest data including previous day for previous close
            latest_data = yf.download(ticker, period='5d', interval='1d')
            
            if len(latest_data) < 2:  # Check length first
                st.warning(f"Insufficient data available for {ticker}. Need at least 2 days of data to show previous close.")
                continue
            
            # Get the last available date from the first valid ticker data
            if last_available_date is None:
                last_available_date = latest_data.index[-1].date()
                # Display market status after we have the last available date
                display_market_status(last_available_date)
            
            # Get current day's data
            current_data = latest_data.iloc[-1]
            prev_data = latest_data.iloc[-2]
            
            # Extract values using .item() for all Series
            try:
                current_close = current_data['Close'].item()
                prev_close = prev_data['Close'].item()
                open_price = current_data['Open'].item()
                high_price = current_data['High'].item()
                low_price = current_data['Low'].item()
                volume = current_data['Volume'].item()
            except Exception as e:
                st.error(f"Error extracting data for {ticker}: {str(e)}")
                continue

            # Calculate price difference if close price is available and market is not open
            diff_str = ""
            diff_color = ""  # Initialize with empty string
            if market_status() != "MARKET_OPEN":
                price_diff = prediction - current_close
                if not pd.isna(price_diff):
                    diff_color = "#4CAF50" if price_diff >= 0 else "#FF5252"  # Green if positive, red if negative
                    diff_sign = "+" if price_diff >= 0 else ""
                    diff_str = f'<span style="color: {diff_color}; margin-left: 8px;">({diff_sign}${price_diff:.2f})</span>'

            # Display ticker with predicted close and difference
            st.markdown(
                f'<div style="margin-bottom: 5px; font-size: 1.1em; font-weight: bold;">'
                f'{ticker} - Predicted Close: <span style="font-size: 1.15em;">${prediction:.2f}</span>{diff_str}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Format values
            open_val = f"${open_price:.2f}" if not pd.isna(open_price) else "N/A"
            high_val = f"${high_price:.2f}" if not pd.isna(high_price) else "N/A"
            low_val = f"${low_price:.2f}" if not pd.isna(low_price) else "N/A"
            prev_close_val = f"${prev_close:.2f}" if not pd.isna(prev_close) else "N/A"
            current_val = f"${current_data['Close'].item():.2f}" if market_status() == "MARKET_OPEN" else (f"${current_close:.2f}" if not pd.isna(current_close) else "N/A")
            volume_val = f"{int(volume):,}" if not pd.isna(volume) else "N/A"
            
            # Create grid display
            current_label = "Last Traded" if market_status() == "MARKET_OPEN" else "Close"
            metrics = ['Open', 'High', 'Low', 'Prev Close', current_label, 'Volume']
            values = [open_val, high_val, low_val, prev_close_val, current_val, volume_val]
            
            # Create HTML table with refined styling and theme-appropriate colors
            html = f"""
            <div style="margin: 10px 0;">
                <table style="width: 100%; text-align: center; border-collapse: collapse;">
                    <tr>
                        {''.join(f'<td style="width: 16.66%; padding: 2px;"><small style="opacity: 0.6;">{metric}</small></td>' for metric in metrics)}
                    </tr>
                    <tr>
                        {''.join(f'<td style="width: 16.66%; padding: 2px;"><span style="font-size: 1.1em; opacity: 0.8;">{value}</span></td>' for value in values)}
                    </tr>
                </table>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error displaying data for {ticker}: {str(e)}")
            continue


# Title of the app
st.title("Stock Price Predictor")

# List of initial stock tickers
initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Initialize session state variables
if 'tickers' not in st.session_state:
    st.session_state.tickers = initial_tickers.copy()  # Example: ['TSLA', 'NVDA', 'AAPL']

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

# Track the previous value of selected_tickers to detect changes
if 'prev_selected_tickers' not in st.session_state:
    st.session_state.prev_selected_tickers = st.session_state.selected_tickers

# Initialize new_ticker for search bar functionality if not already initialized
if 'new_ticker' not in st.session_state:
    st.session_state.new_ticker = ''

tickers = st.multiselect(
    "Select stocks to predict:",
    st.session_state.tickers,
    default=st.session_state.selected_tickers,  # Preserve selected tickers
    key='stock_multiselect',
    on_change=update_selected_tickers,
    args=['stock_multiselect']
)

# Update selected tickers based on multiselect
st.session_state.selected_tickers = tickers

# Search bar for new tickers
new_ticker = st.text_input("Search for a stock ticker:", key="new_ticker_input")

if new_ticker:
    try:
        stock_data = yf.download(new_ticker, period='1d')
        if stock_data.empty:
            st.warning(f"Ticker '{new_ticker}' is not valid or does not exist.")
        else:
            new_ticker_upper = new_ticker.upper()
            if new_ticker_upper not in [t.upper() for t in st.session_state.tickers]:
                st.session_state.tickers.insert(0, new_ticker_upper)
                st.session_state.selected_tickers.append(new_ticker_upper)
                new_ticker = ""
                st.rerun()
            else:
                if new_ticker_upper not in [t.upper() for t in st.session_state.selected_tickers]:
                    st.session_state.selected_tickers.append(new_ticker_upper)
                    st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# Predict button logic
if st.button("Predict"):
    # Directly assign selected tickers from session state to tickers
    tickers = st.session_state.selected_tickers
    print(f"Tickers {tickers}")
    
    close_price_prediction = execute_pipeline(tickers)
    display_results(close_price_prediction)
