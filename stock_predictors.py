import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta, time

# Define NYSE market hours (Global constants)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
# Define NOW as full datetime
NOW = datetime.now() - timedelta(hours=0)  # Adjust for your testing time offset
NOW_DATE = NOW.date()
NOW_TIME = NOW.time()

      
def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"
    print(f"[UPDATE] Change Type: {type(change)}") # This will print <class 'str'>

    # Access the updated multiselect value directly
    updated_sel_tickers = st.session_state.stock_multiselect
    print(f"[UPDATE] Updated selected_tickers: {updated_sel_tickers}") 

    st.session_state.selected_tickers = updated_sel_tickers
    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")

def market_status():
    # """
    # Determines the current market status:
    # 1. Before market opens
    # 2. During market hours
    # 3. After market closes
    # """

    # Get the current time in Eastern Time (or your server's time zone)
    #now = datetime.now().time()  # Current system time
    #testing times for
    # moved to global constant for testing now = (datetime.now() - timedelta(hours=1)).time()
    
    # 1. Check if it is between 12:00 AM and market open (before 9:30 AM)
    if NOW_TIME < MARKET_OPEN:
        return "BEFORE_MARKET_OPEN"  # Market has not opened yet (Midnight to 9:30 AM)
    
    # 2. Check if it is during market hours (between 9:30 AM and 4:00 PM)
    elif MARKET_OPEN <= NOW_TIME <= MARKET_CLOSE:
        return "MARKET_OPEN"  # Market is open (9:30 AM to 4:00 PM)
    
    # 3. Check if it is after market close but before 11:59 PM (4:00 PM to 11:59 PM)
    else:
        return "AFTER_MARKET_CLOSE"  # Market has closed, but it's still before midnight (4:00 PM to 11:59 PM)
    
    #Check to see if we are before today's market open
        #This means today's date has no data
        #If the market hasn't opened, we can predict today's close based on yesterdays closing values
    
    #Check to see if the market is open today
        #IF YES CONTINUE to predict today's close

    #Check to see if the market is after close today
        #IF it is after close this means two things
            # I should have the current day's data
            # In which case I can show today's prediction and today's closing prices

def fetch_data(ticker, end_date, start_date='2010-01-01'):
    if end_date is None:
        end_date = datetime.now().date()
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    return stock_data

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

def get_features_based_on_market_status(train_ready_data):
    # """
    # Returns the features (X) and date for model prediction based on market status and latest data.

    # :param train_ready_data: The data to use for feature selection
    # :return: The feature set (X) and corresponding date
    # """
    
    # Extract the date component
    #the_date = datetime.strptime(latest_data.name.date())
    #try this: data_date = datetime.strptime(str(latest_data.name), "%Y-%m-%d %H:%M:%S").date()

    the_date = (latest_data.name.date())
                                  
    print(f"Um the data date is {the_date}")
    #CODE GETS HERE

    # Determine market status
    status = market_status()

    #For now we test with status market open
    #status = "MARKET_OPEN"
    print(f"TEST assigned status to {status}")

    prediction_data = latest_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']]  # Ensure these columns are in the correct order
        # Ensure the prediction data is returned as a DataFrame
    input_features = pd.DataFrame([prediction_data])
    print(f"There are the input features {input_features}")
    
    # Select features based on market status
    if status == "MARKET_OPEN":
        print(f"I came into getfeatures and I'm in {status}")

        # prediction_data = latest_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']]  # Ensure these columns are in the correct order
        # # Ensure the prediction data is returned as a DataFrame
        # input_features = pd.DataFrame([prediction_data])
        # print(f"There are the input features {input_features}")

    elif status == "BEFORE_MARKET_OPEN":
        print(f"It is before Market Open for today.  We will use the previous day's closing values to predict today's closing price")

        #features = latest_data[['Prev Close']]

    elif status == "AFTER_MARKET_CLOSE":
        #features = latest_data[['Open', 'High', 'Low', 'Volume','Prev Close']]
        print(f"I'm in After Market Close")

    else:
        raise ValueError("Unknown market status")

    return input_features # Return features as DataFrame and date as scalar

def fetch_features(ticker):
    stock_data = yf.download(ticker, period='5d') #taking 5 days of data can only be 1 or 5- ensuring intra day non-close record doesn't get dropped
    #latest_data = 
    print(f"Fetch Features stock_data")
    print(stock_data)
    prediction_data = preprocess_data(stock_data).tail(1) #ensures that we get previous close and the last row in the df
    prediction_features = prediction_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']] #Select the features in the right order from the df
    
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
            # Get training data
            stock_data = fetch_data(each_ticker, NOW_DATE + timedelta(days=1))
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
    status = market_status()
    
    # Format and display current date
    date_str = NOW.strftime('%A, %B %d, %Y')
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 0;'>{date_str}</h2>", unsafe_allow_html=True)

    last_date_str = last_available_date.strftime('%A, %B %d')

    if status == "BEFORE_MARKET_OPEN":
        st.markdown(f"""<div style='text-align: center; margin-top: -10px;'>
            <h3 style='margin-bottom: 0;'>⏳ Market hasn't opened yet</h3>
            <div style='font-size: 10pt; margin-top: -10px;'>Predicted closing prices are for {last_date_str} based on the latest available data</div>
        </div>""", unsafe_allow_html=True)
    elif status == "MARKET_OPEN":
        time_str = NOW.strftime('%I:%M %p')  # Only format time when needed
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
                f'<div style="margin-bottom: 5px; font-size: 1.1em; color: #000; font-weight: bold;">'
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
            
            # Create HTML table with refined styling
            html = f"""
            <div style="margin: 10px 0;">
                <table style="width: 100%; text-align: center; border-collapse: collapse;">
                    <tr>
                        {''.join(f'<td style="width: 16.66%; padding: 2px;"><small style="color: #666;">{metric}</small></td>' for metric in metrics)}
                    </tr>
                    <tr>
                        {''.join(f'<td style="width: 16.66%; padding: 2px;"><span style="font-size: 1.1em; color: #999;">{value}</span></td>' for value in values)}
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
