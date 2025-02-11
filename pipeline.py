import streamlit as st
import numpy as np
import pandas as pd  # Add pandas import
import time as time_module
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import get_nyse_date, market_status, get_nyse_datetime, get_last_row
from data_handler import fetch_data
from data_processor import preprocess_data
from feature_engineering import add_technical_indicators, get_feature_columns
from model import train_model
import traceback

def execute_pipeline(tickers):
    # Initialize variables before try block
    todays_close_predictions = []
    next_days_close_predictions = []
    predictions = [todays_close_predictions, next_days_close_predictions]
    
    status = market_status()
    print(f"\nMarket status: {status}\n")
        
    for each_ticker in tickers:
        st.markdown(f"Processing {each_ticker}...")
        print(f"\nProcessing {each_ticker}...\n")

        try:
            # Get training data using NYSE timezone
            nyse_date = get_nyse_date()
            stock_data = fetch_data(each_ticker, nyse_date + timedelta(days=2))
            print(f"\n\nFetched stock data for {nyse_date + timedelta(days=1)} {each_ticker}:\n{stock_data.tail()}\n\n")  # Debugging line
            
            if stock_data.empty:
                print(f"\n\nNo data returned for {each_ticker}\n\n")
                continue
            
            # Process base data (only cleans Prev Close NaNs)
            processed_data = preprocess_data(stock_data, "linear_regression")
            
            # Add technical indicators
            train_ready_data_xgb = add_technical_indicators(processed_data.copy(), each_ticker)
            
            # Add Next_Day_Close to both datasets
            processed_data[('Next_Day_Close', each_ticker)] = processed_data[('Close', each_ticker)].shift(-1)
            train_ready_data_xgb[('Next_Day_Close', each_ticker)] = train_ready_data_xgb[('Close', each_ticker)].shift(-1)
            
            # Save last rows AFTER adding Next_Day_Close (will have NaN in Next_Day_Close)
            last_row_basic = processed_data.iloc[-1:].copy()
            last_row_with_tech = train_ready_data_xgb.iloc[-1:].copy()
            
            # Create training datasets by dropping NaNs only once
            train_ready_data_linear = processed_data.dropna().copy()
            train_ready_data_xgb = train_ready_data_xgb.dropna().copy()
            
            # Train models on clean historical data
            linear_model, (X_scaler_linear, y_scaler_linear) = train_model(train_ready_data_linear, model_type="linear_regression")
            xgb_model, (X_scaler_xgb, y_scaler_xgb) = train_model(train_ready_data_xgb, model_type="xgboost")

            try:
                # --- Todays Close Linear Regression Prediction using last_row_basic ---
                feature_cols_linear = get_feature_columns(model_type="linear_regression")
                prediction_input_linear = [last_row_basic[(col, each_ticker)].iloc[0] for col in feature_cols_linear]
                linear_prediction_data = np.array(prediction_input_linear).reshape(1, -1)
                linear_prediction_scaled = X_scaler_linear.transform(linear_prediction_data)
                linear_prediction = linear_model.predict(linear_prediction_scaled)
                linear_prediction = y_scaler_linear.inverse_transform(linear_prediction.reshape(-1, 1))
                todays_close_predictions.append((each_ticker, float(linear_prediction[0])))

                # --- Today's CloseXGBoost Predictions (using data with technical indicators) ---
                feature_cols_xgb = get_feature_columns(model_type="xgboost")
                available_cols = set((col, each_ticker) for col in train_ready_data_xgb.columns.get_level_values(0))
                prediction_features = []
                for col in feature_cols_xgb:
                    if (col, each_ticker) in available_cols:
                        prediction_features.append(train_ready_data_xgb.iloc[-1:][(col, each_ticker)])
                    else:
                        print(f"Warning: Column {col} not found in data")
                
                xgb_prediction_data = np.array(prediction_features).reshape(1, -1)
                xgb_prediction_scaled = X_scaler_xgb.transform(xgb_prediction_data)
                xgb_prediction = xgb_model.predict(xgb_prediction_scaled)
                xgb_prediction = y_scaler_xgb.inverse_transform(xgb_prediction.reshape(-1, 1))
                todays_close_predictions.append((each_ticker, float(xgb_prediction[0])))

                # --- XGBoost Prediction using last_row_with_tech ---
                prediction_features_last_row = []
                for col in feature_cols_xgb:
                    if (col, each_ticker) in last_row_with_tech.columns:
                        value = last_row_with_tech[(col, each_ticker)].iloc[0]
                        if pd.isna(value):
                            print(f"Warning: NaN found in prediction data for {col}")
                            value = 0  # Use 0 for prediction only, not training
                        prediction_features_last_row.append(value)
                    else:
                        print(f"Warning: Missing column {col} in prediction data")
                        prediction_features_last_row.append(0)

                xgb_prediction_data_last_row = np.array(prediction_features_last_row).reshape(1, -1)
                xgb_prediction_scaled_last_row = X_scaler_xgb.transform(xgb_prediction_data_last_row)
                xgb_prediction_last_row = xgb_model.predict(xgb_prediction_scaled_last_row)
                xgb_prediction_last_row = y_scaler_xgb.inverse_transform(xgb_prediction_last_row.reshape(-1, 1))
                next_days_close_predictions.append((each_ticker, float(xgb_prediction_last_row[0])))
                print(f"XGBoost close for {each_ticker}: {xgb_prediction_last_row[0]}")
                print(f"XGBoost next day close for {each_ticker}: {xgb_prediction_last_row[0]}")
            except Exception as e:
                print(f"Error processing {each_ticker}: {str(e)}")
                # Log the current state of predictions even if an error occurs
                print(f"DEBUG (on error): Predictions - Close Predictions: {todays_close_predictions}")
                print(f"DEBUG (on error): Predictions - Next Day Close Predictions: {next_days_close_predictions}")
                continue  # Continue to the next ticker

        except Exception as e:
            print(f"\nError processing {each_ticker}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull stack trace:")
            traceback.print_exc()  # This will print the full stack trace
            
            # Optional: Print more detailed debugging info
            print("\nDebug info at time of error:")
            print(f"Current ticker: {each_ticker}")
            print(f"Last successful operation: {traceback.extract_tb(e.__traceback__)[-1].name}")
            
    # Log the final state of predictions
    print(f"DEBUG: Predictions - Close Predictions: {todays_close_predictions}")
    print(f"DEBUG: Predictions - Next Day Close Predictions: {next_days_close_predictions}")
    return predictions