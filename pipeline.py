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
    todays_close_prediction_lr = []    # Linear regression predictions for today
    todays_close_prediction_xgb = []   # XGBoost predictions for today
    next_days_close_prediction_xgb = []  # XGBoost predictions for tomorrow
    predictions = [todays_close_prediction_lr, todays_close_prediction_xgb, next_days_close_prediction_xgb]
    
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
            
            # Process base data and add Next_Day_Close (only as target)
            processed_data = preprocess_data(stock_data, "linear_regression")
                        
            # Add technical indicators
            train_ready_data_xgb = add_technical_indicators(processed_data.copy(), each_ticker)
            
            # Save last row for predictions
            prediction_row = train_ready_data_xgb.iloc[-1:].copy()
            
            # Clean training data (excluding prediction row)
            train_ready_data_linear = processed_data.copy().dropna()  # For today's close
            train_ready_data_xgb = train_ready_data_xgb.copy().dropna()  # For next day's close
            
            # Debug prints for data inspection
            print(f"\nLinear Regression Training Data for {each_ticker} (last 5 rows):")
            print(train_ready_data_linear.tail())
            print("\nFeature columns for Linear Regression:", get_feature_columns(model_type="linear_regression"))
            
            print(f"\nXGBoost Training Data for {each_ticker} (last 5 rows):")
            print(train_ready_data_xgb.tail())
            print("\nFeature columns for XGBoost:", get_feature_columns(model_type="xgboost"))
        
            # Train models for Today's Close - Updated target_column specification
            linear_model_todays_close, (X_scaler_linear, y_scaler_linear) = train_model(
                train_ready_data_linear, 
                model_type="linear_regression", 
                target_column='Close')
            
            # Calculate Linear Regression evaluation metrics
            feature_cols_linear = get_feature_columns(model_type="linear_regression")
            X_train_linear = np.array([train_ready_data_linear[(col, each_ticker)].values for col in feature_cols_linear]).T
            y_train_linear = train_ready_data_linear[('Close', each_ticker)].values
            y_pred_linear = linear_model_todays_close.predict(X_scaler_linear.transform(X_train_linear))
            mae_linear = mean_absolute_error(y_train_linear, y_scaler_linear.inverse_transform(y_pred_linear.reshape(-1, 1)))
            mse_linear = mean_squared_error(y_train_linear, y_scaler_linear.inverse_transform(y_pred_linear.reshape(-1, 1)))
            
            # Print Linear Regression metrics
            print(f"\nLinear Regression Model Evaluation for {each_ticker}:")
            print(f"MAE: ${mae_linear:.2f}")
            print(f"MSE: ${mse_linear:.2f}")
            print(f"RMSE: ${np.sqrt(mse_linear):.2f}")
            print("Feature Coefficients:")
            # Normalize coefficients to get relative importance
            coefficients = np.abs(linear_model_todays_close.coef_)
            normalized_coefficients = coefficients / np.sum(coefficients)
            for feat, coef in zip(feature_cols_linear, normalized_coefficients):
                print(f"{feat}: {coef:.4f}")

            xgb_model_todays_close, (X_scaler_xgb, y_scaler_xgb) = train_model(
                train_ready_data_xgb, 
                model_type="xgboost", 
                target_column='Close')

            # Calculate XGBoost evaluation metrics
            feature_cols_xgb = get_feature_columns(model_type="xgboost")
            X_train_xgb = np.array([train_ready_data_xgb[(col, each_ticker)].values for col in feature_cols_xgb]).T
            y_train_xgb = train_ready_data_xgb[('Close', each_ticker)].values
            y_pred_xgb = xgb_model_todays_close.predict(X_scaler_xgb.transform(X_train_xgb))
            mae_xgb = mean_absolute_error(y_train_xgb, y_scaler_xgb.inverse_transform(y_pred_xgb.reshape(-1, 1)))
            mse_xgb = mean_squared_error(y_train_xgb, y_scaler_xgb.inverse_transform(y_pred_xgb.reshape(-1, 1)))

            # Print XGBoost metrics
            print(f"\nXGBoost Model Evaluation for {each_ticker}:")
            print(f"MAE: ${mae_xgb:.2f}")
            print(f"MSE: ${mse_xgb:.2f}")
            print(f"RMSE: ${np.sqrt(mse_xgb):.2f}")
            print("Feature Importances:")
            for feat, imp in zip(feature_cols_xgb, xgb_model_todays_close.feature_importances_):
                print(f"{feat}: {imp:.4f}")

            # --- Linear Regression - Predict today's close ---
            feature_cols_linear = get_feature_columns(model_type="linear_regression")
            prediction_input_linear = [prediction_row[(col, each_ticker)].iloc[0] for col in feature_cols_linear]
            linear_prediction_data = np.array(prediction_input_linear).reshape(1, -1)
            linear_prediction_scaled = X_scaler_linear.transform(linear_prediction_data)
            linear_prediction_todays_close = linear_model_todays_close.predict(linear_prediction_scaled)
            linear_prediction_todays_close = y_scaler_linear.inverse_transform(linear_prediction_todays_close.reshape(-1, 1))
            todays_close_prediction_lr.append((each_ticker, float(linear_prediction_todays_close[0])))
            
            # --- XGBoost - Predict today's close ---
            feature_cols_xgb = get_feature_columns(model_type="xgboost")
            prediction_features = []
            for col in feature_cols_xgb:
                value = prediction_row[(col, each_ticker)].iloc[0]
                prediction_features.append(value if not pd.isna(value) else 0)
            
            xgb_prediction_today = xgb_model_todays_close.predict(X_scaler_xgb.transform(np.array(prediction_features).reshape(1, -1)))
            xgb_prediction_today = y_scaler_xgb.inverse_transform(xgb_prediction_today.reshape(-1, 1))
            todays_close_prediction_xgb.append((each_ticker, float(xgb_prediction_today[0])))

            # --- XGBoost - Predict tomorrow's close ---
            xgb_prediction_tomorrow = xgb_model_todays_close.predict(X_scaler_xgb.transform(np.array(prediction_features).reshape(1, -1)))
            xgb_prediction_tomorrow = y_scaler_xgb.inverse_transform(xgb_prediction_tomorrow.reshape(-1, 1))
            next_days_close_prediction_xgb.append((each_ticker, float(xgb_prediction_tomorrow[0])))

            print(f"\nPredictions for {each_ticker}:")
            print(f"Today's close (Linear Regression): ${float(linear_prediction_todays_close[0]):.2f}")
            print(f"Today's close (XGBoost): ${float(xgb_prediction_today[0]):.2f}")
            print(f"Tomorrow's close (XGBoost): ${float(xgb_prediction_tomorrow[0]):.2f}")

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
            
    return predictions