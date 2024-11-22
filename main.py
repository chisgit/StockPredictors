import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta, time
from data_utils import fetch_stock_data, preprocess_data
from model_utils import train_model, make_prediction
from visualization_utils import plot_data, plot_predictions

# Define main function
def main():
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

# Execute main function
if __name__ == "__main__":
    main()