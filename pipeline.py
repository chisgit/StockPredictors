import streamlit as st
import numpy as np
import time as time_module
from datetime import timedelta
from utils import get_nyse_date, market_status, get_nyse_datetime, get_last_row
from data_handler import fetch_data, fetch_features
from data_processor import preprocess_data
from feature_engineering import add_technical_indicators, get_feature_columns
from model import train_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import traceback

def execute_pipeline(tickers):
    # Initialize variables before try block
    close_predictions = []
    next_day_close_predictions = []
    train_ready_data = None
    predictions = [close_predictions, next_day_close_predictions]
    
    status = market_status()
    print(f"Market status: {status}")
        
    for each_ticker in tickers:
        st.markdown(f"Processing {each_ticker}...")
        print(f"\nProcessing {each_ticker}...")

        try:
            # Get training data using NYSE timezone
            nyse_date = get_nyse_date()
            stock_data = fetch_data(each_ticker, nyse_date + timedelta(days=1))
            
            if stock_data.empty:
                print(f"No data returned for {each_ticker}")
                continue
            
            # Process data and add features
            processed_data = preprocess_data(stock_data)
            train_ready_data = add_technical_indicators(processed_data, each_ticker)
            
            # Train and predict using Linear Regression
            linear_model, (X_scaler, y_scaler) = train_model(train_ready_data, "linear_regression")
            print(f"\nType of X_scaler: {type(X_scaler)}\n")  
            print(f"\nType of y_scaler: {type(y_scaler)}\n")  
           
            # Make linear regression prediction on the latest row
            linear_feature_cols = get_feature_columns(train_ready_data, "linear_regression")
            prediction_input = []
            for feature in linear_feature_cols:
                prediction_input.append(train_ready_data.iloc[-1:][(feature, each_ticker)]) 
            linear_prediction_data = np.array(prediction_input).reshape(1, -1)

            # Scale the prediction data
            linear_prediction_scaled = X_scaler.transform(linear_prediction_data)

            # Ensure linear_prediction is assigned
            try:
                linear_prediction = linear_model.predict(linear_prediction_scaled)
                linear_prediction = y_scaler.inverse_transform(linear_prediction.reshape(-1, 1))
                prediction_value = float(linear_prediction[0])
                print(f"\nLinear Regression Prediction for {each_ticker}: ${prediction_value:.2f}")
                close_predictions.append((each_ticker, prediction_value))  # Store today's close prediction
            except Exception as e:
                print(f"Error processing {each_ticker}: {str(e)}")
                traceback.print_exc()  # Print the full stack trace

            # Train and predict using XGBoost
            xgb_model, scaler = train_model(train_ready_data, "xgboost")
            X_scaler, y_scaler = scaler  # Unpack the scalers from the tuple
            
            # Make XGBoost prediction on the latest row
            last_row = get_last_row(train_ready_data)
            xgb_feature_cols = get_feature_columns(last_row, "xgboost")
            prediction_input_xgb = []
        
            for feature in xgb_feature_cols:
                prediction_input_xgb.append(last_row[(feature, each_ticker)])
            xgb_prediction_data = np.array(prediction_input_xgb).reshape(1, -1)

            # Debug: Print shape and feature columns
            #print(f"\nDEBUG: xgb_prediction_data shape: {xgb_prediction_data.shape[0]}")  # Updated debug statement
            print(f"\nDEBUG: XGBoost Features: {xgb_feature_cols}\n")

            # Scale the input data for prediction
            xgb_prediction_scaled = X_scaler.transform(xgb_prediction_data)
            print(f"\nType of xgb_prediction_scaled: {type(xgb_prediction_scaled)}\n")
            # print(f"\nNumber of rows in xgb_prediction_scaled: {xgb_prediction_scaled.shape[0]}")  # Debugging line
            xgb_prediction = xgb_model.predict(xgb_prediction_scaled)
            xgb_prediction = y_scaler.inverse_transform(xgb_prediction.reshape(-1, 1))
            prediction_value = float(xgb_prediction[0])
            print(f"\nXGBoost Prediction for {each_ticker}: ${prediction_value:.2f}\n")
            next_day_close_predictions.append((each_ticker, prediction_value))

        except Exception as e:
            print(f"Error processing {each_ticker}: {str(e)}")
            continue
            
    return predictions