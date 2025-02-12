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
    """Process each ticker independently with models trained specifically for that ticker"""
    # Initialize storage for predictions and trained models
    todays_close_predictions = []
    next_days_close_predictions = []
    predictions = [todays_close_predictions, next_days_close_predictions]
    
    # Store models for each ticker
    models = {}
    
    for ticker in tickers:
        st.markdown(f"Processing {ticker}...")
        print(f"\nProcessing {ticker}...\n")
        
        try:
            # Get training data using NYSE timezone
            nyse_date = get_nyse_date()
            stock_data = fetch_data(ticker, nyse_date + timedelta(days=2))
            print(f"\n\nFetched stock data for {nyse_date + timedelta(days=1)} {ticker}:\n{stock_data.tail()}\n\n")  # Debugging line
            
            if stock_data.empty:
                print(f"\n\nNo data returned for {ticker}\n\n")
                continue
            
            # Process base data (only cleans Prev Close NaNs)
            processed_data = preprocess_data(stock_data, "linear_regression")
            
            # Add technical indicators
            train_ready_data_xgb = add_technical_indicators(processed_data.copy(), ticker)
            
            # Get ticker from data columns
            ticker_level = 1  # Index of ticker in MultiIndex
            current_ticker = processed_data.columns[0][ticker_level]
            
            # Replace all direct [0][1] accesses with current_ticker
            processed_data[('Next_Day_Close', current_ticker)] = processed_data[('Close', current_ticker)].shift(-1)
            train_ready_data_xgb[('Next_Day_Close', current_ticker)] = train_ready_data_xgb[('Close', current_ticker)].shift(-1)
            
            print("\nVerifying Next_Day_Close data:")
            print(f"Sample of Next_Day_Close values:\n{train_ready_data_xgb[('Next_Day_Close', current_ticker)].tail()}")
            print(f"Sample of Close values:\n{train_ready_data_xgb[('Close', current_ticker)].tail()}")

            # Save last rows AFTER adding Next_Day_Close (will have NaN in Next_Day_Close)
            last_row_basic = processed_data.iloc[-1:].copy()
            last_row_with_tech = train_ready_data_xgb.iloc[-1:].copy()
            
            # Create training datasets by dropping NaNs only once
            train_ready_data_linear = processed_data.dropna().copy()
            train_ready_data_xgb = train_ready_data_xgb.dropna().copy()
            
            # Train separate models for this specific ticker
            models[ticker] = {
                'linear_today': {
                    'model': None,
                    'scalers': None
                },
                'linear_next_day': {
                    'model': None,
                    'scalers': None
                },
                'xgboost_today': {
                    'model': None,
                    'scalers': None
                },
                'xgboost_next_day': {
                    'model': None,
                    'scalers': None
                }
            }
            
            # Train linear regression for today's close for this ticker
            models[ticker]['linear_today']['model'], models[ticker]['linear_today']['scalers'] = train_model(
                train_ready_data_linear,
                model_type="linear_regression",
                target="today"
            )

            # Train linear regression for next day's close for this ticker
            models[ticker]['linear_next_day']['model'], models[ticker]['linear_next_day']['scalers'] = train_model(
                train_ready_data_linear,
                model_type="linear_regression",
                target="next_day"
            )
            
            # Train XGBoost for today's close for this ticker
            models[ticker]['xgboost_today']['model'], models[ticker]['xgboost_today']['scalers'] = train_model(
                train_ready_data_xgb,
                model_type="xgboost",
                target="today"
            )
            
            # Train XGBoost for next day's close for this ticker
            models[ticker]['xgboost_next_day']['model'], models[ticker]['xgboost_next_day']['scalers'] = train_model(
                train_ready_data_xgb,
                model_type="xgboost",
                target="next_day"
            )
            
            try:
                # Make predictions using this ticker's specific models
                linear_model_today = models[ticker]['linear_today']['model']
                X_scaler_linear_today, y_scaler_linear_today = models[ticker]['linear_today']['scalers']
                
                linear_model_next_day = models[ticker]['linear_next_day']['model']
                X_scaler_linear_next_day, y_scaler_linear_next_day = models[ticker]['linear_next_day']['scalers']
                
                xgb_model_today = models[ticker]['xgboost_today']['model']
                X_scaler_xgb_today, y_scaler_xgb_today = models[ticker]['xgboost_today']['scalers']
                
                xgb_model_next_day = models[ticker]['xgboost_next_day']['model']
                X_scaler_xgb_next_day, y_scaler_xgb_next_day = models[ticker]['xgboost_next_day']['scalers']
                
                # --- Todays Close Linear Regression Prediction using last_row_basic ---
                feature_cols_linear_today = get_feature_columns(model_type="linear_regression", target="today")
                prediction_input_linear_today = [last_row_basic[(col, ticker)].iloc[0] for col in feature_cols_linear_today]
                linear_prediction_data_today = np.array(prediction_input_linear_today).reshape(1, -1)
                linear_prediction_scaled_today = X_scaler_linear_today.transform(linear_prediction_data_today)
                linear_prediction_today = linear_model_today.predict(linear_prediction_scaled_today)
                linear_prediction_today = y_scaler_linear_today.inverse_transform(linear_prediction_today.reshape(-1, 1))
                todays_close_predictions.append((ticker, float(linear_prediction_today[0])))

                # Linear Regression prediction for next day
                feature_cols_linear_next = get_feature_columns(model_type="linear_regression", target="next_day")
                prediction_input_linear_next = [last_row_basic[(col, ticker)].iloc[0] for col in feature_cols_linear_next]
                linear_prediction_data_next = np.array(prediction_input_linear_next).reshape(1, -1)
                linear_prediction_scaled_next = X_scaler_linear_next_day.transform(linear_prediction_data_next)
                linear_prediction_next = linear_model_next_day.predict(linear_prediction_scaled_next)
                linear_prediction_next = y_scaler_linear_next_day.inverse_transform(linear_prediction_next.reshape(-1, 1))
                next_days_close_predictions.append((ticker, float(linear_prediction_next[0])))

                # XGBoost prediction for today's close
                feature_cols_xgb_today = get_feature_columns(model_type="xgboost", target="today")
                prediction_features_today = []
                for col in feature_cols_xgb_today:
                    if (col, ticker) in last_row_with_tech.columns:
                        value = last_row_with_tech[(col, ticker)].iloc[0]
                        prediction_features_today.append(value if not pd.isna(value) else 0)
                    else:
                        print(f"Warning: Missing column {col} in prediction data")
                        prediction_features_today.append(0)
                
                # Today's close prediction
                xgb_prediction_today = make_prediction(
                    prediction_features_today,
                    xgb_model_today,
                    X_scaler_xgb_today,
                    y_scaler_xgb_today
                )
                todays_close_predictions.append((ticker, float(xgb_prediction_today[0])))

                # Next day's close prediction
                feature_cols_xgb_next = get_feature_columns(model_type="xgboost", target="next_day")
                prediction_features_next = []
                for col in feature_cols_xgb_next:
                    if (col, ticker) in last_row_with_tech.columns:
                        value = last_row_with_tech[(col, ticker)].iloc[0]
                        prediction_features_next.append(value if not pd.isna(value) else 0)
                    else:
                        print(f"Warning: Missing column {col} in prediction data")
                        prediction_features_next.append(0)
                
                xgb_prediction_next_day = make_prediction(
                    prediction_features_next,
                    xgb_model_next_day,
                    X_scaler_xgb_next_day,
                    y_scaler_xgb_next_day
                )
                next_days_close_predictions.append((ticker, float(xgb_prediction_next_day[0])))
                
                # Fix debugging output formatting
                print(f"DEBUG XGBoost today's close for {ticker}: {float(xgb_prediction_today[0]):.2f}")
                print(f"DEBUG Linear Regression next day close for {ticker}: {float(linear_prediction_next[0]):.2f}")
                print(f"DEBUG XGBoost next day close for {ticker}: {float(xgb_prediction_next_day[0]):.2f}")
                
            except Exception as e:
                print(f"Error making predictions for {ticker}: {str(e)}")
                continue

        except Exception as e:
            print(f"\nError processing {ticker}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull stack trace:")
            traceback.print_exc()  # This will print the full stack trace
            
            # Optional: Print more detailed debugging info
            print("\nDebug info at time of error:")
            print(f"Current ticker: {ticker}")
            print(f"Last successful operation: {traceback.extract_tb(e.__traceback__)[-1].name}")
            
    # Log the final state of predictions
    print(f"DEBUG: Predictions - Close Predictions: {todays_close_predictions}")
    print(f"DEBUG: Predictions - Next Day Close Predictions: {next_days_close_predictions}")
    return predictions

def make_prediction(features, model, X_scaler, y_scaler):
    """Helper function to make predictions using a model"""
    prediction_data = np.array(features).reshape(1, -1)
    prediction_scaled = X_scaler.transform(prediction_data)
    prediction = model.predict(prediction_scaled)
    return y_scaler.inverse_transform(prediction.reshape(-1, 1))