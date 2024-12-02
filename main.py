import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from initializers.initializer import initialize_model
import streamlit as st
from app import render_ui  # Import the render_ui function from app.py
from initializers import initialize_session  # Import the session initialization function

# Define main function
def main():
    #ensure all the global variables are set
    '''
    gen_utils.init_session()
    app.render_ui()
    gen_utils.multiselector()
    gen_utils.serachbar()
    gen_utils.get_todays_date()
    gen_utils.convert_types(object1, object2)
    gen.utils.message_handler()
    data_processor.check_ticker()
    

    data_fetcher.check_ticker()


    # User input (optional)
    ticker = input("Enter stock ticker symbol (e.g., TSLA): ")

    # Fetch and preprocess data
    data = fetch_stock_data(ticker)
    processed_data = preprocess_data(data)

    # Train and evaluate model (optional)
    # You can comment out these lines if you pre-train a model and save it
    X_train, X_test, y_train, y_test = split_data(processed_data)  # Implement split_data function
    model = train_model(X_train, y_train)

    # Make prediction
    predictions = make_prediction(model, X_test)

    # Visualize results (optional)
    plot_data(processed_data)  # Plot historical data (optional)
    plot_predictions(y_test, predictions)  # Plot actual vs predicted prices (optional)

    # Display prediction (or integrate with Streamlit app)
    print(f"Predicted closing price for {ticker}: ${predictions[-1]:.2f}")
    '''
    # Teslapredictor/main.py
from app import render_ui  # Import the render_ui function from app.py
from initializers import initialize_session  # Import the session initialization function

# List of initial stock tickers
initial_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Initialize session with the default tickers
initialize_session(initial_tickers)

# Start the Streamlit UI
render_ui()

if __name__ == "__main__":
    main()
